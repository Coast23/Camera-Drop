#pragma once

#include "util/config.hpp"
#include "util/errors.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

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
    static uint32_t mingw_downscale_u32(std::mt19937& rng, uint32_t range) {
        if (range == 0) {
            throw InterleaverError("Invalid zero range in MinGW-equivalent shuffle");
        }

        uint64_t product = static_cast<uint64_t>(rng()) * static_cast<uint64_t>(range);
        uint32_t low = static_cast<uint32_t>(product);
        if (low < range) {
            const uint32_t threshold = static_cast<uint32_t>(0u - range) % range;
            while (low < threshold) {
                product = static_cast<uint64_t>(rng()) * static_cast<uint64_t>(range);
                low = static_cast<uint32_t>(product);
            }
        }
        return static_cast<uint32_t>(product >> 32);
    }

    static uint64_t mingw_uniform_inclusive(std::mt19937& rng, uint64_t upper) {
        constexpr uint64_t rng_min = static_cast<uint64_t>(std::mt19937::min());
        constexpr uint64_t rng_max = static_cast<uint64_t>(std::mt19937::max());
        constexpr uint64_t rng_range = rng_max - rng_min;

        if (rng_range > upper) {
            const uint64_t inclusive_range = upper + 1;
            if (inclusive_range <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return mingw_downscale_u32(rng, static_cast<uint32_t>(inclusive_range));
            }

            const uint64_t scaling = rng_range / inclusive_range;
            const uint64_t past = inclusive_range * scaling;
            uint64_t value = 0;
            do {
                value = static_cast<uint64_t>(rng()) - rng_min;
            } while (value >= past);
            return value / scaling;
        }

        if (rng_range == upper) {
            return static_cast<uint64_t>(rng()) - rng_min;
        }

        const uint64_t expanded_rng_range = rng_range + 1;
        uint64_t combined_base = 0;
        uint64_t value = 0;
        do {
            combined_base = expanded_rng_range *
                            mingw_uniform_inclusive(rng, upper / expanded_rng_range);
            value = combined_base + (static_cast<uint64_t>(rng()) - rng_min);
        } while (value > upper || value < combined_base);
        return value;
    }

    static std::pair<uint64_t, uint64_t> mingw_gen_two_uniform_ints(uint64_t b0,
                                                                    uint64_t b1,
                                                                    std::mt19937& rng) {
        const uint64_t combined = mingw_uniform_inclusive(rng, (b0 * b1) - 1);
        return {combined / b1, combined % b1};
    }

    static void mingw_equivalent_shuffle(std::vector<uint32_t>& values) {
        if (values.empty()) {
            return;
        }

        std::mt19937 rng(0x114514);
        const uint64_t rng_range = static_cast<uint64_t>(std::mt19937::max())
                                 - static_cast<uint64_t>(std::mt19937::min());
        const uint64_t value_count = static_cast<uint64_t>(values.size());

        if (rng_range / value_count >= value_count) {
            size_t i = 1;

            if ((value_count % 2) == 0) {
                const size_t pos = static_cast<size_t>(mingw_uniform_inclusive(rng, 1));
                std::swap(values[i], values[pos]);
                ++i;
            }

            while (i < values.size()) {
                const uint64_t swap_range = static_cast<uint64_t>(i) + 1;
                const auto [pos0, pos1] =
                    mingw_gen_two_uniform_ints(swap_range, swap_range + 1, rng);
                std::swap(values[i], values[static_cast<size_t>(pos0)]);
                ++i;
                std::swap(values[i], values[static_cast<size_t>(pos1)]);
                ++i;
            }
            return;
        }

        for (size_t i = 1; i < values.size(); ++i) {
            const size_t pos = static_cast<size_t>(
                mingw_uniform_inclusive(rng, static_cast<uint64_t>(i)));
            std::swap(values[i], values[pos]);
        }
    }

    Interleaver(){
        size_t size = Config::UINTS_COUNT;
        
        if (size == 0) {
            throw InterleaverError("Config::UINTS_COUNT is 0, call Config::auto_config() first");
        }
        
        keymap_.resize(size);

        std::iota(keymap_.begin(), keymap_.end(), 0);

        // Protocol compatibility requires the exact permutation produced by the
        // historical MinGW g++ 15.2.0 / libstdc++ shuffle path.
        mingw_equivalent_shuffle(keymap_);
    }

    std::vector<uint32_t> keymap_;
};
