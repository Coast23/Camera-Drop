#include "vision/pattern_dict.hpp"

#include <cstdio>
#include <filesystem>
#include <map>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util/errors.hpp"

namespace camdrop::vision {
namespace {

namespace fs = std::filesystem;

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
    if (gray8.empty()) {
        throw PatternDictError("Empty image in mask_from_8x8_gray");
    }
    
    if (gray8.rows != 8 || gray8.cols != 8) {
        throw PatternDictError("Expected 8x8 image, got " + std::to_string(gray8.cols) + "x" + std::to_string(gray8.rows));
    }
    
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

/**
 * @brief 从目录加载模式字典
 * @param dir 包含模式图像的目录路径
 * @return 加载的模式字典
 * @throws std::runtime_error 如果无法加载图像文件
 */
PatternDictionary PatternDictionary::LoadFromDirectory(const std::string& dir) {
    if (dir.empty()) {
        throw PatternDictLoadError("Empty directory path");
    }

    std::map<int, fs::path> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path path = entry.path();
            if (path.extension() != ".png") {
                continue;
            }
            const std::string stem = path.stem().string();
            if (stem.empty()) {
                continue;
            }
            size_t parsed = 0;
            int idx = -1;
            try {
                idx = std::stoi(stem, &parsed, 16);
            } catch (const std::exception&) {
                continue;
            }
            if (parsed != stem.size() || idx < 0) {
                continue;
            }
            files[idx] = path;
        }
    } catch (const std::exception& e) {
        throw PatternDictLoadError("Failed to enumerate pattern directory: " + dir + " (" + e.what() + ")");
    }

    if (files.empty()) {
        throw PatternDictLoadError("No hex-named pattern images found in: " + dir);
    }

    const int expected_count = static_cast<int>(files.size());
    int expected_index = 0;
    for (const auto& [idx, path] : files) {
        (void)path;
        if (idx != expected_index) {
            throw PatternDictLoadError(
                "Pattern directory must contain contiguous hex indices starting at 00; missing index " +
                std::to_string(expected_index));
        }
        ++expected_index;
    }

    PatternDictionary dict;
    dict.masks64.resize(expected_count);
    dict.lo.resize(expected_count);
    dict.hi.resize(expected_count);
    dict.masks16.resize(expected_count);

    for (const auto& [idx, path] : files) {
        cv::Mat gray = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            throw PatternDictLoadError("Failed to load pattern image: " + path.string());
        }
        cv::Mat gray8;
        cv::resize(gray, gray8, cv::Size(8, 8), 0.0, 0.0, cv::INTER_NEAREST);

        uint64_t mask;
        try {
            mask = mask_from_8x8_gray(gray8);
        } catch (const PatternDictError& e) {
            throw PatternDictLoadError(std::string("Failed to process pattern ") + path.filename().string() + ": " + e.what());
        }

        const auto parts = split_mask64(mask);
        dict.masks64[idx] = mask;
        dict.lo[idx] = parts.first;
        dict.hi[idx] = parts.second;
        dict.masks16[idx] = compress_mask64_to_16(parts.first, parts.second);
    }

    if ((expected_count & (expected_count - 1)) != 0) {
        throw PatternDictLoadError(
            "Pattern count must be a power of two, got " + std::to_string(expected_count));
    }

    return dict;
}

}  // namespace camdrop::vision
