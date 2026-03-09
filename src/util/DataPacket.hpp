#pragma once
#include "config.hpp"
#include <vector>
#include <cstring>
#include <cstdint>

inline uint32_t calculate_crc32(const uint8_t* data, size_t length){
    uint32_t crc = 0xFFFFFFFF;
    for(size_t i = 0; i < length; ++i){
        crc ^= data[i];
        for(int j = 0; j < 8; ++j){
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

struct FountainMetadata {
    uint32_t file_size;
    uint32_t original_size;
    uint16_t block_id;

    void serialize(uint8_t* buffer) const {
        std::memcpy(buffer, &file_size, 4);
        std::memcpy(buffer + 4, &original_size, 4);
        std::memcpy(buffer + 8, &block_id, 2);
    }
 
    void deserialize(const uint8_t* buffer){
        std::memcpy(&file_size, buffer, 4);
        std::memcpy(&original_size, buffer + 4, 4);
        std::memcpy(&block_id, buffer + 8, 2);
    }
};

class DataPacket {
public:
    DataPacket() : metadata_(), data_() {}

    void set_metadata(const FountainMetadata& meta){
        metadata_ = meta;
    }

    const FountainMetadata& metadata() const {
        return metadata_;
    }

    void set_data(const std::vector<uint8_t>& data){
        data_ = data;
    }

    const std::vector<uint8_t>& data() const {
        return data_;
    }

    // 序列化为字节流
    // Header(10) + Payload + CRC32(4)
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result(Config::FOUNTAIN_HEADER_SIZE + data_.size() + 4);
        metadata_.serialize(result.data());
        std::memcpy(result.data() + Config::FOUNTAIN_HEADER_SIZE, data_.data(), data_.size());
        
        uint32_t crc = calculate_crc32(result.data(), result.size() - 4);
        std::memcpy(result.data() + result.size() - 4, &crc, 4);
        return result;
    }

    // 从字节流反序列化
    // 检验 CRC 并剥离
    bool deserialize(const uint8_t* buffer, size_t size){
        size_t expected_size = Config::FOUNTAIN_PAYLOAD_SIZE;
        
        if(size < expected_size) return false;
        
        // check crc32
        uint32_t actual_crc = calculate_crc32(buffer, expected_size - 4);
        uint32_t expected_crc;
        std::memcpy(&expected_crc, buffer + expected_size - 4, 4);
        if(actual_crc != expected_crc) return false;

        metadata_.deserialize(buffer);

        data_.resize(Config::FOUNTAIN_CHUNK_SIZE);
        std::memcpy(data_.data(), buffer + Config::FOUNTAIN_HEADER_SIZE, Config::FOUNTAIN_CHUNK_SIZE);
        return true;
    }

private:
    FountainMetadata metadata_;
    std::vector<uint8_t> data_;
};
