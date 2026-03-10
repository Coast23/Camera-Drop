#include "vision/visual_frame_codec.hpp"

#include <stdexcept>

#include "codec/interleaver.hpp"
#include "util/BitConverter.hpp"
#include "util/config.hpp"

namespace camdrop::vision {
namespace {

void require_frame_bytes_size(size_t size) {
    if (size != Config::PACKET_CAPACITY) {
        throw std::runtime_error("frame byte count mismatch");
    }
}

void require_interleaved_symbol_count(size_t size) {
    if (size != Config::UINTS_COUNT) {
        throw std::runtime_error("interleaved symbol count mismatch");
    }
}

void require_recognized_layout(const RecognizeResult& result) {
    if (result.header_symbols.size() != Config::HEADER_SYMBOL_COUNT) {
        throw std::runtime_error("recognized header symbol count mismatch");
    }
    if (result.payload_symbols.size() != Config::PAYLOAD_SYMBOL_COUNT) {
        throw std::runtime_error("recognized payload symbol count mismatch");
    }
}

}  // namespace

std::vector<uint8_t> FrameBytesToInterleavedSymbols(const std::vector<uint8_t>& frame_bytes) {
    require_frame_bytes_size(frame_bytes.size());
    std::vector<uint8_t> symbols = BitConverter::convert_826(frame_bytes);
    require_interleaved_symbol_count(symbols.size());
    Interleaver::get_instance().interleave(symbols.data(), symbols.size());
    return symbols;
}

std::vector<uint8_t> InterleavedSymbolsToFrameBytes(const std::vector<uint8_t>& interleaved_symbols) {
    require_interleaved_symbol_count(interleaved_symbols.size());
    std::vector<uint8_t> symbols = interleaved_symbols;
    Interleaver::get_instance().deinterleave(symbols.data(), symbols.size());
    std::vector<uint8_t> frame_bytes = BitConverter::convert_628(symbols);
    require_frame_bytes_size(frame_bytes.size());
    return frame_bytes;
}

std::vector<uint8_t> RecognizeResultToInterleavedSymbols(const RecognizeResult& result) {
    require_recognized_layout(result);
    std::vector<uint8_t> symbols;
    symbols.reserve(Config::UINTS_COUNT);
    symbols.insert(symbols.end(), result.header_symbols.begin(), result.header_symbols.end());
    symbols.insert(symbols.end(), result.payload_symbols.begin(), result.payload_symbols.end());
    require_interleaved_symbol_count(symbols.size());
    return symbols;
}

std::vector<uint8_t> RecognizeResultToFrameBytes(const RecognizeResult& result) {
    return InterleavedSymbolsToFrameBytes(RecognizeResultToInterleavedSymbols(result));
}

}  // namespace camdrop::vision
