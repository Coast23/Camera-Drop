#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace camdrop::vision {

struct PatternDictionary {
    std::vector<uint64_t> masks64;
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    std::vector<uint16_t> masks16;

    bool empty() const { return masks64.empty(); }
    int size() const { return static_cast<int>(masks64.size()); }
    int pattern_bits() const;

    static PatternDictionary LoadFromDirectory(const std::string& dir);
};

}  // namespace camdrop::vision
