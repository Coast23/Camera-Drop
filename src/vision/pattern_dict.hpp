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

    [[nodiscard]] bool empty() const { return masks64.empty(); }
    [[nodiscard]] int size() const { return static_cast<int>(masks64.size()); }
    [[nodiscard]] int pattern_bits() const;

    [[nodiscard]] static PatternDictionary LoadFromDirectory(const std::string& dir);
};

}  // namespace camdrop::vision
