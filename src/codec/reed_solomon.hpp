#pragma once
#include "util/config.hpp"
#include "util/errors.hpp"
extern "C" {
    #include <correct.h>
}

#include <vector>
#include <memory>

class RSEncoder {
public:
    RSEncoder(){
        if (Config::RS_PARITY_SIZE == 0 || Config::RS_BLOCK_SIZE == 0) {
            throw RSInitError("RS configuration not initialized, call Config::auto_config() first");
        }
        
        if (Config::RS_PARITY_SIZE >= Config::RS_BLOCK_SIZE) {
            throw RSInitError("Invalid RS configuration: parity size >= block size");
        }
        
        rs_ = correct_reed_solomon_create(
            correct_rs_primitive_polynomial_8_7_2_1_0,
            1, 1, Config::RS_PARITY_SIZE
        );
        
        if (!rs_) {
            throw RSInitError("Failed to create Reed-Solomon encoder");
        }
    }
    
    ~RSEncoder(){
        if(rs_) correct_reed_solomon_destroy(rs_);
    }

    // 编码一个数据块
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data){
        if(!rs_) {
            throw RSEncodeError("RS encoder not initialized");
        }
        
        if(data.size() > Config::RS_DATA_SIZE) {
            throw RSEncodeError("Input data size " + std::to_string(data.size()) + 
                               " exceeds RS_DATA_SIZE " + std::to_string(Config::RS_DATA_SIZE));
        }
        
        std::vector<uint8_t> encoded(Config::RS_BLOCK_SIZE);   
        ssize_t res = correct_reed_solomon_encode(rs_, data.data(), data.size(), encoded.data());
        
        if (res < 0) {
            throw RSEncodeError("Reed-Solomon encoding failed");
        }
        
        return encoded;
    }
    
private:
    correct_reed_solomon* rs_;
};

class RSDecoder {
public:
    RSDecoder(){
        if (Config::RS_PARITY_SIZE == 0 || Config::RS_BLOCK_SIZE == 0) {
            throw RSInitError("RS configuration not initialized, call Config::auto_config() first");
        }
        
        if (Config::RS_PARITY_SIZE >= Config::RS_BLOCK_SIZE) {
            throw RSInitError("Invalid RS configuration: parity size >= block size");
        }
        
        rs_ = correct_reed_solomon_create(
            correct_rs_primitive_polynomial_8_7_2_1_0,
            1, 1, Config::RS_PARITY_SIZE
        );
        
        if (!rs_) {
            throw RSInitError("Failed to create Reed-Solomon decoder");
        }
    }
    
    ~RSDecoder(){
        if(rs_) correct_reed_solomon_destroy(rs_);
    }

    // 解码一个数据块。解码失败返回空
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded){
        if(!rs_) {
            throw RSDecodeError("RS decoder not initialized");
        }
        
        if(encoded.size() != Config::RS_BLOCK_SIZE) {
            throw RSDecodeError("Input size " + std::to_string(encoded.size()) + 
                               " != RS_BLOCK_SIZE " + std::to_string(Config::RS_BLOCK_SIZE));
        }
        
        std::vector<uint8_t> decoded(Config::RS_DATA_SIZE);
        ssize_t res = correct_reed_solomon_decode(rs_, encoded.data(), Config::RS_BLOCK_SIZE, decoded.data());
        
        if(res < 0) {
            // 解码失败是正常情况（数据损坏），不抛异常，返回空
            return {};
        }

        return decoded;
    }

private:
    correct_reed_solomon* rs_;
};
