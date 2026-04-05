#pragma once

#include "util/config.hpp"
#include "util/errors.hpp"

#include <zstd.h>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

class ZstdCompressor {
public:
    explicit ZstdCompressor(int compression_level = Config::COMPRESSION_LEVEL)
        : compression_level_(compression_level) {
        if (compression_level < 1 || compression_level > 22) {
            throw CompressionError("Invalid compression level: " + std::to_string(compression_level) +
                                  ", must be in range [1, 22]");
        }
    }

    // 压缩数据，文件名会添加到 skippable frame。
    // 若压缩后更大，则直接存原始数据（解码端通过 zstd magic 判断是否压缩）。
    std::vector<uint8_t> compress(const uint8_t* data, size_t size, const std::string& filename = ""){
        if (!data && size > 0) {
            throw CompressError("Null data pointer with non-zero size");
        }

        std::vector<uint8_t> result;
        if(!filename.empty()) write_skippable_frame(result, filename);

        size_t bound = ZSTD_compressBound(size);
        std::vector<uint8_t> compressed(bound);
        size_t compressed_size = ZSTD_compress(
            compressed.data(), bound, data, size, compression_level_);

        if(ZSTD_isError(compressed_size)) {
            throw CompressError(std::string("ZSTD compression failed: ") + ZSTD_getErrorName(compressed_size));
        }

        // 压缩后更大则直接存原始数据
        if (compressed_size >= size) {
            result.insert(result.end(), data, data + size);
        } else {
            result.insert(result.end(), compressed.data(), compressed.data() + compressed_size);
        }
        return result;
    }

private:
    int compression_level_;

    void write_skippable_frame(std::vector<uint8_t>& output, const std::string& filename){
        const uint32_t magic = 0x184D2A50;
        const uint32_t frame_size = static_cast<uint32_t>(filename.size());
        output.resize(8 + frame_size);
        memcpy(output.data(), &magic, 4);
        memcpy(output.data() + 4, &frame_size, 4);
        memcpy(output.data() + 8, filename.data(), frame_size);
    }
};

class ZstdDecompressor {
public:

    // (data, filename)
    std::pair<std::vector<uint8_t>, std::string> decompress(const uint8_t* compressed, size_t compressed_size){
        if (!compressed && compressed_size > 0) {
            throw DecompressError("Null compressed data pointer with non-zero size");
        }

        std::pair<std::vector<uint8_t>, std::string> result;
        size_t offset = 0;

        // 解析 skippable frame
        if(compressed_size >= 8){
            uint32_t magic;
            memcpy(&magic, compressed, 4);
            if ((magic & 0xFFFFFFF0) == 0x184D2A50){
                uint32_t frame_size;
                memcpy(&frame_size, compressed + 4, 4);
                if (8 + frame_size <= compressed_size){
                    result.second = std::string(
                        reinterpret_cast<const char*>(compressed + 8),
                        frame_size
                    );
                    offset = 8 + frame_size;
                }
            }
        }

        if (offset >= compressed_size) {
            throw DecompressError("No data after skippable frame");
        }

        // 检查是否是 zstd frame（magic = 0xFD2FB528）
        const uint8_t* payload = compressed + offset;
        const size_t payload_size = compressed_size - offset;

        bool is_zstd = false;
        if (payload_size >= 4) {
            uint32_t magic;
            memcpy(&magic, payload, 4);
            is_zstd = (magic == 0xFD2FB528u);
        }

        if (!is_zstd) {
            // 未压缩，直接返回原始数据
            result.first.assign(payload, payload + payload_size);
            return result;
        }

        unsigned long long decompressed_size = ZSTD_getFrameContentSize(payload, payload_size);

        if(decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
            throw DecompressError("ZSTD content size error - data may be corrupted");
        }
        if(decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw DecompressError("ZSTD content size unknown");
        }

        result.first.resize(static_cast<size_t>(decompressed_size));
        size_t actual_size = ZSTD_decompress(
            result.first.data(), static_cast<size_t>(decompressed_size),
            payload, payload_size
        );

        if(ZSTD_isError(actual_size)){
            throw DecompressError(std::string("ZSTD decompression failed: ") + ZSTD_getErrorName(actual_size));
        }

        if (actual_size != decompressed_size) {
            result.first.resize(actual_size);
        }

        return result;
    }
};
