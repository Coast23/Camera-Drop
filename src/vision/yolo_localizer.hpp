#pragma once

#include <memory>
#include <optional>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "vision/types.hpp"

namespace camdrop::vision {

struct YoloLocalizerOptions {
    int input_size = 640;
    float conf_threshold = 0.35f;
    float progressive_min_conf_threshold = 0.005f;
    float progressive_conf_step = 0.05f;
    float anchor_expand = 0.05f;
    int ort_threads = 1;
    std::string input_name = "images";
    std::string output_name = "output0";
};

class YoloLocalizer {
public:
    explicit YoloLocalizer(const std::string& model_path, YoloLocalizerOptions options = {});

    const YoloLocalizerOptions& options() const { return options_; }
    std::optional<LocalizeResult> Locate(const cv::Mat& frame);

private:
    YoloLocalizerOptions options_;
    Ort::Env env_;
    Ort::Session session_;
};

}  // namespace camdrop::vision
