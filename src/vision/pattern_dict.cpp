#include "vision/pattern_dict.hpp"

#include <cstdio>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace camdrop::vision {
namespace {

int pattern_bits_for_count(int count) {
    int bits = 0;
    while ((1 << bits) < count) {
        ++bits;
    }
    return bits;
}

std::pair<uint32_t, uint32_t> split_mask64(uint64_t mask) {
    return {
        static_cast<uint32_t>(mask & 0xFFFFFFFFULL),
        static_cast<uint32_t>((mask >> 32) & 0xFFFFFFFFULL),
    };
}

bool mask_is_on(uint32_t lo, uint32_t hi, int bit) {
    if (bit < 32) {
        return ((lo >> bit) & 1U) != 0U;
    }
    return ((hi >> (bit - 32)) & 1U) != 0U;
}

uint16_t compress_mask64_to_16(uint32_t lo, uint32_t hi) {
    uint16_t out = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const int base = (r << 4) + (c << 1);
            int on = 0;
            on += mask_is_on(lo, hi, base) ? 1 : 0;
            on += mask_is_on(lo, hi, base + 1) ? 1 : 0;
            on += mask_is_on(lo, hi, base + 8) ? 1 : 0;
            on += mask_is_on(lo, hi, base + 9) ? 1 : 0;
            if (on >= 2) {
                out = static_cast<uint16_t>(out | (1U << (r * 4 + c)));
            }
        }
    }
    return out;
}

uint64_t mask_from_8x8_gray(const cv::Mat& gray8) {
    uint64_t mask = 0;
    for (int r = 0; r < 8; ++r) {
        const uint8_t* row = gray8.ptr<uint8_t>(r);
        for (int c = 0; c < 8; ++c) {
            if (row[c] < 128) {
                mask |= (1ULL << (r * 8 + c));
            }
        }
    }
    return mask;
}

}  // namespace

int PatternDictionary::pattern_bits() const {
    return pattern_bits_for_count(size());
}

PatternDictionary PatternDictionary::LoadFromDirectory(const std::string& dir) {
    PatternDictionary dict;
    dict.masks64.resize(16);
    dict.lo.resize(16);
    dict.hi.resize(16);
    dict.masks16.resize(16);

    for (int i = 0; i < 16; ++i) {
        char name[16];
        std::snprintf(name, sizeof(name), "%02x.png", i);
        const std::string path = dir + "/" + name;
        cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            throw std::runtime_error("failed to load pattern image: " + path);
        }
        cv::Mat gray8;
        cv::resize(gray, gray8, cv::Size(8, 8), 0.0, 0.0, cv::INTER_NEAREST);
        const uint64_t mask = mask_from_8x8_gray(gray8);
        const auto parts = split_mask64(mask);
        dict.masks64[i] = mask;
        dict.lo[i] = parts.first;
        dict.hi[i] = parts.second;
        dict.masks16[i] = compress_mask64_to_16(parts.first, parts.second);
    }

    return dict;
}

}  // namespace camdrop::vision
