#include "vision/visual_frame_codec.hpp"

#include <stdexcept>

#include "codec/interleaver.hpp"
#include "util/BitConverter.hpp"
#include "util/config.hpp"
#include "util/errors.hpp"

namespace camdrop::vision {
namespace {

void require_frame_bytes_size(size_t size) {
    if (size != Config::PACKET_CAPACITY) {
        throw ImageSizeError("Frame bytes size " + std::to_string(size) + 
                            " != PACKET_CAPACITY " + std::to_string(Config::PACKET_CAPACITY));
    }
}

void require_interleaved_symbol_count(size_t size) {
    if (size != Config::UINTS_COUNT) {
        throw ImageSizeError("Interleaved symbol count " + std::to_string(size) + 
                            " != UINTS_COUNT " + std::to_string(Config::UINTS_COUNT));
    }
}

void require_recognized_layout(const RecognizeResult& result) {
    if (result.header_symbols.size() != Config::HEADER_SYMBOL_COUNT) {
        throw ImageSizeError("Header symbol count " + std::to_string(result.header_symbols.size()) + 
                            " != HEADER_SYMBOL_COUNT " + std::to_string(Config::HEADER_SYMBOL_COUNT));
    }
    if (result.payload_symbols.size() != Config::PAYLOAD_SYMBOL_COUNT) {
        throw ImageSizeError("Payload symbol count " + std::to_string(result.payload_symbols.size()) + 
                            " != PAYLOAD_SYMBOL_COUNT " + std::to_string(Config::PAYLOAD_SYMBOL_COUNT));
    }
}

}  // namespace

std::vector<uint8_t> FrameBytesToInterleavedSymbols(const std::vector<uint8_t>& frame_bytes) {
    require_frame_bytes_size(frame_bytes.size());
    
    std::vector<uint8_t> symbols;
    try {
        symbols = BitConverter::convert_826(frame_bytes);
    } catch (const BitConverterError& e) {
        throw ImageFormatError(std::string("Bit conversion failed: ") + e.what());
    }
    
    require_interleaved_symbol_count(symbols.size());
    
    try {
        Interleaver::get_instance().interleave(symbols.data(), symbols.size());
    } catch (const InterleaverError& e) {
        throw ImageFormatError(std::string("Interleave failed: ") + e.what());
    }
    
    return symbols;
}

std::vector<uint8_t> InterleavedSymbolsToFrameBytes(const std::vector<uint8_t>& interleaved_symbols) {
    require_interleaved_symbol_count(interleaved_symbols.size());
    
    std::vector<uint8_t> symbols = interleaved_symbols;
    try {
        Interleaver::get_instance().deinterleave(symbols.data(), symbols.size());
    } catch (const InterleaverError& e) {
        throw ImageFormatError(std::string("Deinterleave failed: ") + e.what());
    }
    
    std::vector<uint8_t> frame_bytes;
    try {
        frame_bytes = BitConverter::convert_628(symbols);
    } catch (const BitConverterError& e) {
        throw ImageFormatError(std::string("Bit conversion failed: ") + e.what());
    }
    
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
