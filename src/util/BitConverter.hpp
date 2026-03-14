#pragma once

#include "errors.hpp"
#include <vector>
#include <cstdint>

class BitConverter {
public:
    static std::vector<uint8_t> convert_826(const uint8_t* data, size_t size){
        if(!data && size > 0) {
            throw BitConverterError("Null input pointer with non-zero size");
        }
        
        if(!size) return {};
        
        size_t new_size = (size * 8 + 5) / 6;
        std::vector<uint8_t> result(new_size, 0);
        
        size_t actual_size = convert_826(data, size, result.data(), new_size);
        result.resize(actual_size);
        return result;
    }

    static std::vector<uint8_t> convert_826(const std::vector<uint8_t>& data){
        return convert_826(data.data(), data.size());
    }

    // 返回实际写入的字节数
    static size_t convert_826(const uint8_t* input, size_t in_size, uint8_t* output, size_t out_size){
        if(!input && in_size > 0) {
            throw BitConverterError("Null input pointer with non-zero size");
        }
        
        if(!output && out_size > 0) {
            throw BitConverterError("Null output pointer with non-zero size");
        }
        
        if(!in_size || !out_size) return 0;
        
        size_t need = (in_size * 8 + 5) / 6;
        if(out_size < need) {
            throw BitConverterSizeError("Output buffer too small: need " + std::to_string(need) + 
                                        ", got " + std::to_string(out_size));
        }
        
        size_t i = 0;
        size_t out_idx = 0;

        for(; i + 2 < in_size; i += 3){
            uint32_t v = (input[i] << 16) | (input[i + 1] << 8) | input[i + 2];
            
            output[out_idx++] = (v >> 18) & 0x3f;
            output[out_idx++] = (v >> 12) & 0x3f;
            output[out_idx++] = (v >> 6) & 0x3f;
            output[out_idx++] = v & 0x3f;
        }

        size_t rem = in_size - i;
        if(rem == 1){
            uint32_t v = input[i] << 16;
            output[out_idx++] = (v >> 18) & 0x3f;
            output[out_idx++] = (v >> 12) & 0x3f;
        }
        else if(rem == 2){
            uint32_t v = (input[i] << 16) | (input[i + 1] << 8);
            output[out_idx++] = (v >> 18) & 0x3f;
            output[out_idx++] = (v >> 12) & 0x3f;
            output[out_idx++] = (v >> 6) & 0x3f;
        }

        return out_idx;
    }
    // =======================================================

    static std::vector<uint8_t> convert_628(const uint8_t* data, size_t size){
        if(!data && size > 0) {
            throw BitConverterError("Null input pointer with non-zero size");
        }
        
        if(!size) return {};
        
        size_t new_size = (size * 3) / 4;
        std::vector<uint8_t> result(new_size, 0);
        
        size_t actual_size = convert_628(data, size, result.data(), new_size);
        result.resize(actual_size);
        return result;
    }

    static std::vector<uint8_t> convert_628(const std::vector<uint8_t>& data){
        return convert_628(data.data(), data.size());
    }

    static size_t convert_628(const uint8_t* input, size_t in_size, uint8_t* output, size_t out_size){
        if(!input && in_size > 0) {
            throw BitConverterError("Null input pointer with non-zero size");
        }
        
        if(!output && out_size > 0) {
            throw BitConverterError("Null output pointer with non-zero size");
        }
        
        if(!in_size || !out_size) return 0;
        
        size_t need = (in_size * 3) / 4;
        if(out_size < need) {
            throw BitConverterSizeError("Output buffer too small: need " + std::to_string(need) + 
                                        ", got " + std::to_string(out_size));
        }
        
        size_t i = 0;
        size_t out_idx = 0;

        for(; i + 3 < in_size; i += 4){
            uint32_t v = ((input[i] & 0x3f) << 18) | 
                         ((input[i + 1] & 0x3f) << 12) | 
                         ((input[i + 2] & 0x3f) << 6) | 
                         (input[i + 3] & 0x3f);
            
            output[out_idx++] = (v >> 16) & 0xff;
            output[out_idx++] = (v >> 8) & 0xff;
            output[out_idx++] = v & 0xff;
        }

        size_t rem = in_size - i;
        if(rem == 2){
            uint32_t v = ((input[i] & 0x3f) << 18) | ((input[i + 1] & 0x3f) << 12);
            output[out_idx++] = (v >> 16) & 0xff;
        }
        else if(rem == 3){
            uint32_t v = ((input[i] & 0x3f) << 18) | ((input[i + 1] & 0x3f) << 12) | ((input[i + 2] & 0x3f) << 6);
            output[out_idx++] = (v >> 16) & 0xff;
            output[out_idx++] = (v >> 8) & 0xff;
        }

        return out_idx;
    }
};
