#pragma once

#include "util/config.hpp"
#include "util/errors.hpp"

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>

class Interleaver {
public:
    // 获取单例
    static const Interleaver& get_instance(){
        static Interleaver instance;
        return instance;
    }

    void interleave(uint8_t* data_6bits, size_t size) const {
        if(!data_6bits) {
            throw InterleaverError("Null data pointer in interleave");
        }
        
        if(size != keymap_.size()){
            throw InterleaverSizeError("Input size " + std::to_string(size) + 
                                       " != expected " + std::to_string(keymap_.size()));
        }
        
        std::vector<uint8_t> tmp(data_6bits, data_6bits + size);
        for(size_t i = 0; i < size; ++i){
            data_6bits[keymap_[i]] = tmp[i];
        }
    }

    void deinterleave(uint8_t* data_6bits, size_t size) const {
        if(!data_6bits) {
            throw InterleaverError("Null data pointer in deinterleave");
        }
        
        if(size != keymap_.size()){
            throw InterleaverSizeError("Input size " + std::to_string(size) + 
                                       " != expected " + std::to_string(keymap_.size()));
        }

        std::vector<uint8_t> tmp(data_6bits, data_6bits + size);
        for(size_t i = 0; i < size; ++i){
            data_6bits[i] = tmp[keymap_[i]];
        }
    }

private:
    Interleaver(){
        size_t size = Config::UINTS_COUNT;
        
        if (size == 0) {
            throw InterleaverError("Config::UINTS_COUNT is 0, call Config::auto_config() first");
        }
        
        keymap_.resize(size);

        std::iota(keymap_.begin(), keymap_.end(), 0);

        // 随机映射比什么转置好多了
        std::mt19937 rng(0x114514);
        std::shuffle(keymap_.begin(), keymap_.end(), rng);
    }

    std::vector<uint32_t> keymap_;
};
