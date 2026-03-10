#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "vision/pattern_dict.hpp"

namespace camdrop::vision {

class PatternFrameRenderer {
public:
    explicit PatternFrameRenderer(PatternDictionary dict);

    const PatternDictionary& dict() const { return dict_; }

    cv::Mat Render(const std::vector<uint8_t>& frame_bytes) const;
    cv::Mat RenderInterleavedSymbols(const std::vector<uint8_t>& interleaved_symbols) const;

private:
    PatternDictionary dict_;
};

}  // namespace camdrop::vision
