#pragma once

#include "reed_solomon.hpp"
#include "fountain_code.hpp"
#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "util/errors.hpp"
#include "util/ZstdCompressor.hpp"

class Encoder {
public:
    Encoder(const std::string& filename){
        FileReader reader(filename);
        
        size_t fsize = reader.file_size();
        if(fsize > Config::MAX_FILE_SIZE){
            throw EncoderInitError("File size exceeds limit: " + std::to_string(fsize));
        }
        
        std::vector<uint8_t> file_data = reader.read_all();
        if(fsize != 0 and file_data.empty()){
            throw EncoderInitError("Failed to read file: " + filename);
        }

        ZstdCompressor compressor(Config::COMPRESSION_LEVEL);
        try {
            data_ = compressor.compress(file_data.data(), file_data.size(), filename);
        } catch (const CompressionError& e) {
            throw EncoderInitError(std::string("Compression failed: ") + e.what());
        }
        
        if(data_.empty()){
            throw EncoderInitError("Failed to compress file: " + filename);
        }
        
        // Wirehair 要求分块至少为 2，故短文件需 padding。
        uint32_t original_size = static_cast<uint32_t>(data_.size());
        if(data_.size() <= Config::FOUNTAIN_PAYLOAD_SIZE){
            data_.resize(Config::FOUNTAIN_PAYLOAD_SIZE + 1, 0);
        }

        try {
            fountain_encoder_ = std::make_unique<FountainEncoder>(data_, original_size);
        } catch (const FountainError& e) {
            throw EncoderInitError(std::string("Fountain encoder creation failed: ") + e.what());
        }
    }

    bool is_valid() const {
        return fountain_encoder_ && fountain_encoder_->is_valid();
    }

    // 生成下一个一整帧画面的数据
    std::vector<uint8_t> get_packet(){
        if(!is_valid()) {
            throw EncoderRuntimeError("Encoder is not valid");
        }
        
        std::vector<uint8_t> result;
        result.reserve(Config::PACKET_CAPACITY);
        RSEncoder rs_encoder;
        
        // 循环生成当前帧需要的所有喷泉包
        for(uint32_t i = 0; i < Config::FOUNTAIN_PACKETS_PER_FRAME; ++i){
            DataPacket fountain_packet;
            try {
                fountain_packet = fountain_encoder_->encode_block();
            } catch (const FountainError& e) {
                throw EncoderRuntimeError(std::string("Fountain encode failed: ") + e.what());
            }
            
            std::vector<uint8_t> fountain_data = fountain_packet.serialize();
        
            // 对这个 Fountain 包进行 RS 分块（N个）
            size_t offset = 0;
            while(offset < fountain_data.size()){
                size_t chunk_size = std::min((size_t)Config::RS_DATA_SIZE, fountain_data.size() - offset);
                std::vector<uint8_t> chunk(
                    fountain_data.begin() + offset,
                    fountain_data.begin() + offset + chunk_size
                );

                if(chunk.size() < Config::RS_DATA_SIZE){
                    chunk.resize(Config::RS_DATA_SIZE, 0); 
                }

                std::vector<uint8_t> encoded;
                try {
                    encoded = rs_encoder.encode(chunk);
                } catch (const RSError& e) {
                    throw EncoderRuntimeError(std::string("RS encode failed: ") + e.what());
                }
                
                if(encoded.empty()) {
                    throw EncoderRuntimeError("RS encoder returned empty data");
                }
                
                result.insert(result.end(), encoded.begin(), encoded.end());
                offset += chunk_size;
            }
        }

        if(result.size() < Config::PACKET_CAPACITY){
            result.resize(Config::PACKET_CAPACITY, 0);
        }
        return result;
    }

    uint32_t packet_count_recommended() const {
        if(!is_valid()) {
            throw EncoderRuntimeError("Encoder is not valid");
        }
        return fountain_encoder_->blocks_recommended();
    }
    
private:
    std::vector<uint8_t> data_;
    std::unique_ptr<FountainEncoder> fountain_encoder_;
};
