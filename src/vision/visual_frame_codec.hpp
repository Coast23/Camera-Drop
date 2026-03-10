#pragma once

#include <cstdint>
#include <vector>

#include "vision/types.hpp"

namespace camdrop::vision {

std::vector<uint8_t> FrameBytesToInterleavedSymbols(const std::vector<uint8_t>& frame_bytes);
std::vector<uint8_t> InterleavedSymbolsToFrameBytes(const std::vector<uint8_t>& interleaved_symbols);
std::vector<uint8_t> RecognizeResultToInterleavedSymbols(const RecognizeResult& result);
std::vector<uint8_t> RecognizeResultToFrameBytes(const RecognizeResult& result);

}  // namespace camdrop::vision
