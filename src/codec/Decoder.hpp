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

    Decoder() : fountain_decoder_(std::make_unique<FountainDecoder>()) {}

    // 处理一整帧接受到的图像数据
    // TODO: 函数改名 process_frame_data()
    bool process_packet(const std::vector<uint8_t>& frame_data, ProcessPacketStats* stats = nullptr){
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
                    std::vector<uint8_t> decoded = rs_decoder.decode(rs_block);
                    if(decoded.empty()){
                        packet_valid = false;
                        if(stats) stats->rs_blocks_fail++;
                    }
                    else{
                        if(stats) stats->rs_blocks_ok++;
                        decoded_payload.insert(
                            decoded_payload.end(),
                            decoded.begin(),
                            decoded.end()
                        );
                    }
                }
            }

            if(packet_valid and decoded_payload.size() == Config::FOUNTAIN_PAYLOAD_SIZE){
                DataPacket packet;
                if(packet.deserialize(decoded_payload.data(), decoded_payload.size())){
                    if(stats) stats->fountain_packets_crc_ok++;
                    const auto add_result = fountain_decoder_->add_block_ex(packet);
                    if(add_result == FountainDecoder::AddBlockResult::NeedMore){
                        accepted_any = true;
                        if(stats) stats->fountain_blocks_added++;
                    }
                    else if(add_result == FountainDecoder::AddBlockResult::Complete){
                        accepted_any = true;
                        if(stats) {
                            stats->fountain_blocks_added++;
                            stats->fountain_blocks_completed++;
                        }
                    }
                    else if(stats){
                        if(add_result == FountainDecoder::AddBlockResult::Duplicate) stats->fountain_blocks_duplicate++;
                        else if(add_result == FountainDecoder::AddBlockResult::FileMismatch) stats->fountain_blocks_file_mismatch++;
                        else if(add_result == FountainDecoder::AddBlockResult::DecodeError) stats->fountain_blocks_decode_error++;
                    }
                } else if(stats){
                    stats->fountain_packets_crc_fail++;
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

    bool save_to_file(const std::string& filename){
        if(!is_complete()) return false;
        std::vector<uint8_t> data = fountain_decoder_->decode();
      
        ZstdDecompressor decompressor;
        auto decompressed = decompressor.decompress(data.data(), data.size());
        auto decompressed_data = decompressed.first;
        //  if(data.empty()) return false;
        FileWriter writer(filename);
        if(!writer.is_open()) return false;
        return writer.write(decompressed_data);
    }

private:
    std::unique_ptr<FountainDecoder> fountain_decoder_;
    RSDecoder rs_decoder;
};
