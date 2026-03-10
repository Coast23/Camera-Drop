#pragma once

#include "vision/pattern_dict.hpp"
#include "vision/types.hpp"

namespace camdrop::vision {

struct RecognizerOptions {
    bool enable_luma_recheck = true;
    bool enable_bitgrid_recheck = true;
    bool sharpen_hint = false;
    double sharpen_strength = 0.6;
};

class PatternRecognizer {
public:
    explicit PatternRecognizer(PatternDictionary dict, RecognizerOptions options = {});

    const PatternDictionary& dict() const { return dict_; }
    RecognizeResult Decode(const cv::Mat& deskewed) const;

private:
    PatternDictionary dict_;
    RecognizerOptions options_;
};

}  // namespace camdrop::vision
