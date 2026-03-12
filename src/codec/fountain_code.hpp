#pragma once

#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/errors.hpp"
extern "C" {
    #include <wirehair/wirehair.h>
}

#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <unordered_map>

// 确保 wirehair_init 全局只被调用一次
inline void wirehair_init_once(){
    static std::once_flag flag;
    std::call_once(flag, [](){
        if(wirehair_init() != Wirehair_Success) {
            throw FountainInitError("Failed to initialize Wirehair library");
        }
    });
}

class FountainEncoder {
public:
    FountainEncoder(const std::vector<uint8_t>& data, uint32_t original_size, uint8_t encode_id = 0) 
        : data_(data), encode_id_(encode_id), block_cnt_(0), original_size_(original_size) {
        
        if (data.empty()) {
            throw FountainInitError("Cannot create encoder with empty data");
        }
        
        if (Config::FOUNTAIN_CHUNK_SIZE == 0) {
            throw FountainInitError("FOUNTAIN_CHUNK_SIZE not configured, call Config::auto_config() first");
        }
        
        wirehair_init_once();

        codec_ = wirehair_encoder_create(
            nullptr, data_.data(), data_.size(), Config::FOUNTAIN_CHUNK_SIZE);
            
        if (!codec_) {
            throw FountainInitError("Failed to create Wirehair encoder");
        }
    }
    
    ~FountainEncoder(){
        if(codec_) wirehair_free(codec_);
    }

    bool is_valid() const {
        return codec_ != nullptr;
    }

    // 编码一个块
    DataPacket encode_block(){
        if (!codec_) {
            throw FountainEncodeError("Encoder not initialized");
        }
        
        DataPacket packet;
        FountainMetadata meta;
        meta.file_size = static_cast<uint32_t>(data_.size());
        meta.block_id = block_cnt_++;
        meta.original_size = original_size_;
        packet.set_metadata(meta);
        std::vector<uint8_t> chunk(Config::FOUNTAIN_CHUNK_SIZE);
        uint32_t written = 0;

        WirehairResult result = wirehair_encode(
            codec_,
            meta.block_id,
            chunk.data(),
            Config::FOUNTAIN_CHUNK_SIZE,
            &written
        );

        if(result != Wirehair_Success){
            throw FountainEncodeError("Wirehair encode failed with error: " + std::to_string(result));
        }

        packet.set_data(chunk);
        return packet;
    }

    // 计算需要的最少块数
    uint32_t blocks_required() const {
        if (Config::FOUNTAIN_CHUNK_SIZE == 0) {
            throw FountainError("FOUNTAIN_CHUNK_SIZE not configured");
        }
        return static_cast<uint32_t>((data_.size() + Config::FOUNTAIN_CHUNK_SIZE - 1) / Config::FOUNTAIN_CHUNK_SIZE);
    }

    // 丢包率下的建议块数
    uint32_t blocks_recommended() const {
        return static_cast<uint32_t>(blocks_required() * Config::REDUNDANCY_FACTOR);
    }

private:
    std::vector<uint8_t> data_;
    uint8_t encode_id_;
    uint32_t original_size_;
    uint32_t block_cnt_;
    WirehairCodec codec_;
};

class FountainDecoder {
public:
    enum class AddBlockResult {
        NeedMore,
        Complete,
        Duplicate,
        FileMismatch,
        InitFailed,
        DecodeError,
    };

    FountainDecoder()
        : codec_(nullptr), file_size_(0), original_size_(0), init_(false), is_complete_(false) {
            wirehair_init_once();
    }
    
    ~FountainDecoder(){
        if(codec_) wirehair_free(codec_);
    }

    // 禁止拷贝
    FountainDecoder(const FountainDecoder&) = delete;
    FountainDecoder& operator = (const FountainDecoder&) = delete;

    // 添加新接收的数据块，返回详细结果
    AddBlockResult add_block_ex(const DataPacket& packet){
        if(is_complete_) return AddBlockResult::Complete;

        const FountainMetadata& meta = packet.metadata();
        
        if (meta.file_size == 0) {
            throw FountainDecodeError("Invalid packet: file_size is 0");
        }
        
        if(!init_){
            if (Config::FOUNTAIN_CHUNK_SIZE == 0) {
                throw FountainInitError("FOUNTAIN_CHUNK_SIZE not configured");
            }
            
            file_size_ = meta.file_size;
            original_size_ = meta.original_size;
            codec_ = wirehair_decoder_create(
                nullptr, file_size_, Config::FOUNTAIN_CHUNK_SIZE);
            if(!codec_) return AddBlockResult::InitFailed;
            init_ = true;
        }

        if(meta.file_size != file_size_ || meta.original_size != original_size_) return AddBlockResult::FileMismatch;
        if(block_feeded_.count(meta.block_id)) return AddBlockResult::Duplicate;

        WirehairResult result = wirehair_decode(
            codec_,
            meta.block_id,
            packet.data().data(),
            packet.data().size()
        );
        block_feeded_[meta.block_id] = true;

        if(result == Wirehair_Success) {
            is_complete_ = true;
            return AddBlockResult::Complete;
        }
        if(result == Wirehair_NeedMore) {
            return AddBlockResult::NeedMore;
        }
        block_feeded_.erase(meta.block_id);
        return AddBlockResult::DecodeError;
    }

    // 添加新接收的数据块，返回是否成功（向后兼容）
    bool add_block(const DataPacket& packet){
        const AddBlockResult result = add_block_ex(packet);
        return result == AddBlockResult::NeedMore || result == AddBlockResult::Complete;
    }

    // 判断是否已经接收到足够的块并完成解码
    bool is_complete() const {
        return is_complete_;
    }

    // 获取解码后的数据
    std::vector<uint8_t> decode() const {
        if(!is_complete()) {
            throw FountainDecodeError("Decoding not complete");
        }
        
        if(file_size_ == 0) {
            throw FountainDecodeError("Invalid file size (0)");
        }

        std::vector<uint8_t> recovered(file_size_);
        WirehairResult result = wirehair_recover(
            codec_,
            recovered.data(),
            file_size_
        );

        if(result != Wirehair_Success) {
            throw FountainDecodeError("Wirehair recover failed with error: " + std::to_string(result));
        }
       
        recovered.resize(original_size_);
        return recovered;
    }

private:
    WirehairCodec codec_;
    uint32_t file_size_;
    uint32_t original_size_;
    bool init_;
    bool is_complete_;
    std::unordered_map<uint32_t, bool> block_feeded_;
};
