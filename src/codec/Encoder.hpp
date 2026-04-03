#pragma once

#include "reed_solomon.hpp"
#include "fountain_code.hpp"
#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "util/errors.hpp"
#include "util/ZstdCompressor.hpp"

struct EncoderOptions {
    uint32_t logical_frame_limit = 0;
};

class Encoder {
public:
    Encoder(const std::string& filename, const EncoderOptions& options = {}){
        FileReader reader(filename);
        
        size_t fsize = reader.file_size();
        if(fsize > Config::MAX_FILE_SIZE){
            throw EncoderInitError("File size exceeds limit: " + std::to_string(fsize));
        }

        source_file_size_ = fsize;
        if (options.logical_frame_limit > 0) {
            const uint64_t max_packets = static_cast<uint64_t>(options.logical_frame_limit)
                                       * static_cast<uint64_t>(Config::FOUNTAIN_PACKETS_PER_FRAME);
            if (max_packets == 0) {
                throw EncoderInitError("logical frame limit is too small for current layout");
            }
            const uint64_t max_compressed_bytes_u64 = max_packets * static_cast<uint64_t>(Config::FOUNTAIN_CHUNK_SIZE);
            if (max_compressed_bytes_u64 > static_cast<uint64_t>(Config::MAX_FILE_SIZE)) {
                throw EncoderInitError("logical frame limit exceeds max compressed size");
            }
            const PrefixCompressResult limited = compress_best_fit_prefix(
                reader,
                filename,
                fsize,
                static_cast<size_t>(max_compressed_bytes_u64));
            source_bytes_read_ = limited.source_bytes;
            input_truncated_ = source_bytes_read_ < source_file_size_;
            data_ = limited.compressed;
        } else {
            std::vector<uint8_t> file_data = reader.read_all();
            if(fsize != 0 and file_data.empty()){
                throw EncoderInitError("Failed to read file: " + filename);
            }
            source_bytes_read_ = file_data.size();
            data_ = compress_bytes(file_data, filename);
        }
        
        if(data_.empty()){
            throw EncoderInitError("Failed to compress file: " + filename);
        }
        compressed_size_ = data_.size();
        
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

    uint32_t packet_count_required() const {
        if(!is_valid()) {
            throw EncoderRuntimeError("Encoder is not valid");
        }
        return fountain_encoder_->blocks_required();
    }

    size_t source_file_size() const { return source_file_size_; }
    size_t source_bytes_read() const { return source_bytes_read_; }
    size_t compressed_size() const { return compressed_size_; }
    bool input_truncated() const { return input_truncated_; }
    
private:
    struct PrefixCompressResult {
        std::vector<uint8_t> compressed;
        size_t source_bytes = 0;
    };

    static std::vector<uint8_t> compress_bytes(const std::vector<uint8_t>& file_data,
                                               const std::string& filename) {
        ZstdCompressor compressor(Config::COMPRESSION_LEVEL);
        try {
            return compressor.compress(file_data.data(), file_data.size(), filename);
        } catch (const CompressionError& e) {
            throw EncoderInitError(std::string("Compression failed: ") + e.what());
        }
    }

    static PrefixCompressResult compress_prefix(FileReader& reader,
                                                const std::string& filename,
                                                size_t prefix_size) {
        reader.reset();
        std::vector<uint8_t> file_data = reader.read(prefix_size);
        if (file_data.size() != prefix_size) {
            throw EncoderInitError("Failed to read requested prefix from file: " + filename);
        }
        PrefixCompressResult result;
        result.source_bytes = prefix_size;
        result.compressed = compress_bytes(file_data, filename);
        return result;
    }

    static PrefixCompressResult compress_best_fit_prefix(FileReader& reader,
                                                         const std::string& filename,
                                                         size_t file_size,
                                                         size_t max_compressed_bytes) {
        PrefixCompressResult best = compress_prefix(reader, filename, 0);
        if (best.compressed.size() > max_compressed_bytes) {
            throw EncoderInitError("logical frame limit is too small to fit compressed metadata");
        }

        size_t low = 0;
        size_t high = file_size;
        while (low <= high) {
            const size_t mid = low + ((high - low) / 2);
            PrefixCompressResult current = compress_prefix(reader, filename, mid);
            if (current.compressed.size() <= max_compressed_bytes) {
                best = std::move(current);
                low = mid + 1;
            } else {
                if (mid == 0) {
                    break;
                }
                high = mid - 1;
            }
        }
        return best;
    }

    std::vector<uint8_t> data_;
    std::unique_ptr<FountainEncoder> fountain_encoder_;
    size_t source_file_size_ = 0;
    size_t source_bytes_read_ = 0;
    size_t compressed_size_ = 0;
    bool input_truncated_ = false;
};
