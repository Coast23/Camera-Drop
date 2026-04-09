#pragma once

#include <functional>
#include <memory>
#include <string>

#include "vision/color_cnn.hpp"
#include "vision/pattern_dict.hpp"
#include "vision/pattern_cnn.hpp"
#include "vision/recognizer.hpp"
#include "vision/types.hpp"
#include "vision/yolo_localizer.hpp"

namespace camdrop::vision {

struct FramePipelineConfig {
    std::string model_path;
    std::string pattern_dir;
    std::string pattern_cnn_model_path;
    std::string color_cnn_model_path;
    YoloLocalizerOptions localizer_options;
    RecognizerOptions recognizer_options;
    int deskew_width = kImageWidth;
    int deskew_height = kImageHeight;
    float deskew_expand = 1.0f;
    int deskew_canonical_inset = 2;
    bool patch_track_enabled = true;
    int patch_size = 16;
    int patch_search = 10;
    float patch_sad_max = 0.18f;
};

class FramePipeline {
public:
    explicit FramePipeline(const FramePipelineConfig& config);
    ~FramePipeline();
    FramePipeline(const FramePipeline&) = delete;
    FramePipeline& operator=(const FramePipeline&) = delete;
    FramePipeline(FramePipeline&&) noexcept = default;
    FramePipeline& operator=(FramePipeline&&) noexcept = default;

    const PatternDictionary& dict() const { return recognizer_.dict(); }
    PipelineResult Process(const cv::Mat& frame,
                           const std::function<bool(const cv::Mat&)>& skip_recognition = {});

private:
    struct PatchState {
        cv::Point2f center;
        std::vector<float> gray;
    };

    FramePipelineConfig config_;
    YoloLocalizer localizer_;
    PatternRecognizer recognizer_;
    std::unique_ptr<PatternCnnClassifier> pattern_cnn_;
    std::unique_ptr<ColorCnnClassifier> color_cnn_;
    std::vector<PatchState> patches_;
    CornerQuad last_corners_;
    bool patch_ready_ = false;

    bool init_patches(const cv::Mat& frame, const CornerQuad& corners);
    bool track_patches(const cv::Mat& frame, CornerQuad* out);
    void clear_patches();
};

}  // namespace camdrop::vision
