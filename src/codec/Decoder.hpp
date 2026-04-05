#pragma once

#include "reed_solomon.hpp"
#include "fountain_code.hpp"
#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "util/errors.hpp"
#include "util/ZstdCompressor.hpp"

class Decoder {
public:
    struct ProcessPacketStats {
        uint32_t rs_blocks_ok = 0;
        uint32_t rs_blocks_fail = 0;
        uint32_t fountain_packets_crc_ok = 0;
        uint32_t fountain_packets_crc_fail = 0;
        uint32_t fountain_blocks_added = 0;
        uint32_t fountain_blocks_completed = 0;
        uint32_t fountain_blocks_duplicate = 0;
        uint32_t fountain_blocks_file_mismatch = 0;
        uint32_t fountain_blocks_decode_error = 0;
    };

    Decoder() : fountain_decoder_(std::make_unique<FountainDecoder>()) {
        if (!fountain_decoder_) {
            throw DecoderInitError("Failed to create fountain decoder");
        }
    }

    // 处理一整帧接受到的图像数据
    bool process_packet(const std::vector<uint8_t>& frame_data, ProcessPacketStats* stats = nullptr){
        if (frame_data.empty()) {
            return false;
        }
        
        if (frame_data.size() < Config::PACKET_CAPACITY) {
            if (stats) stats->rs_blocks_fail++;
            return false;
        }

        std::vector<uint8_t> decoded_data;
        bool accepted_any = false;

        size_t offset = 0;

        // 逐个提取帧内所有的 Fountain 包
        for(uint32_t i = 0; i < Config::FOUNTAIN_PACKETS_PER_FRAME; ++i){
            std::vector<uint8_t> decoded_payload;
            decoded_payload.reserve(Config::FOUNTAIN_PAYLOAD_SIZE);
            bool packet_valid = true;

            for(uint32_t j = 0; j < Config::RS_BLOCKS_PER_FOUNTAIN_CHUNK; ++j){
                if(offset + Config::RS_BLOCK_SIZE > frame_data.size()){
                    packet_valid = false;
                    break;
                }

                std::vector<uint8_t> rs_block(
                    frame_data.begin() + offset,
                    frame_data.begin() + offset + Config::RS_BLOCK_SIZE
                );
                offset += Config::RS_BLOCK_SIZE;

                if(packet_valid){
                    std::vector<uint8_t> decoded;
                    try {
                        decoded = rs_decoder.decode(rs_block);
                    } catch (const RSError& e) {
                        packet_valid = false;
                        if(stats) stats->rs_blocks_fail++;
                        continue;
                    }
                    
                    if(decoded.empty()){
                        packet_valid = false;
                        if(stats) stats->rs_blocks_fail++;
                    }
                    else{
                        if(stats) stats->rs_blocks_ok++;
                        decoded_payload.insert(
                            decoded_payload.end(),
                            decoded.begin(), decoded.end()
                        );
                    }
                }
            }

            if(packet_valid && decoded_payload.size() == Config::FOUNTAIN_PAYLOAD_SIZE){
                DataPacket packet;
                try {
                    if(!packet.deserialize(decoded_payload.data(), decoded_payload.size())){
                        if(stats) stats->fountain_packets_crc_fail++;
                        continue;
                    }
                } catch (const DataPacketError& e) {
                    if(stats) stats->fountain_packets_crc_fail++;
                    continue;
                }
                
                if(stats) stats->fountain_packets_crc_ok++;
                
                FountainDecoder::AddBlockResult add_result;
                try {
                    add_result = fountain_decoder_->add_block_ex(packet);
                } catch (const FountainError& e) {
                    if(stats) stats->fountain_blocks_decode_error++;
                    continue;
                }
                
                switch (add_result) {
                    case FountainDecoder::AddBlockResult::NeedMore:
                        accepted_any = true;
                        if(stats) stats->fountain_blocks_added++;
                        break;
                    case FountainDecoder::AddBlockResult::Complete:
                        accepted_any = true;
                        if(stats) {
                            stats->fountain_blocks_added++;
                            stats->fountain_blocks_completed++;
                        }
                        break;
                    case FountainDecoder::AddBlockResult::Duplicate:
                        if(stats) stats->fountain_blocks_duplicate++;
                        break;
                    case FountainDecoder::AddBlockResult::FileMismatch:
                        if(stats) stats->fountain_blocks_file_mismatch++;
                        break;
                    case FountainDecoder::AddBlockResult::DecodeError:
                        if(stats) stats->fountain_blocks_decode_error++;
                        break;
                    case FountainDecoder::AddBlockResult::InitFailed:
                        if(stats) stats->fountain_blocks_decode_error++;
                        break;
                }
            } else if(stats && decoded_payload.size() == Config::FOUNTAIN_PAYLOAD_SIZE){
                stats->fountain_packets_crc_fail++;
            }
        }
        return accepted_any;
    }

    bool is_complete() const {
        return fountain_decoder_->is_complete();
    }

    uint32_t blocks_needed() const {
        return fountain_decoder_->blocks_needed();
    }

    uint32_t blocks_required() const {
        return fountain_decoder_->blocks_required();
    }

    uint32_t blocks_received() const {
        return fountain_decoder_->blocks_received();
    }

    void save_to_file(const std::string& filename){
        if(!is_complete()) {
            throw DecoderRuntimeError("Decoding not complete, cannot save file");
        }
        
        std::vector<uint8_t> data;
        try {
            data = fountain_decoder_->decode();
        } catch (const FountainError& e) {
            throw DecoderRuntimeError(std::string("Fountain decode failed: ") + e.what());
        }
        
        if(data.empty()) {
            throw DecoderRuntimeError("Fountain decoder returned empty data");
        }
      
        ZstdDecompressor decompressor;
        std::pair<std::vector<uint8_t>, std::string> decompressed;
        
        try {
            decompressed = decompressor.decompress(data.data(), data.size());
        } catch (const DecompressError& e) {
            throw DecoderRuntimeError(std::string("Decompression failed: ") + e.what());
        }
        
        auto& decompressed_data = decompressed.first;
        if(decompressed_data.empty()) {
            throw DecoderRuntimeError("Decompressed data is empty");
        }
        
        FileWriter writer(filename);
        try {
            writer.write(decompressed_data);
        } catch (const FileWriteError& e) {
            throw DecoderRuntimeError(std::string("Failed to write output file: ") + e.what());
        }
    }

private:
    std::unique_ptr<FountainDecoder> fountain_decoder_;
    RSDecoder rs_decoder;
};
