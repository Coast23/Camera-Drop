#include "vision/frame_pipeline.hpp"

#include <stdexcept>
#include <limits>
#include <array>

#include <opencv2/imgproc.hpp>

#include "vision/deskew.hpp"
#include "util/errors.hpp"

namespace camdrop::vision {

FramePipeline::FramePipeline(const FramePipelineConfig& config)
    : config_(config),
      localizer_(config.model_path, config.localizer_options),
      recognizer_(PatternDictionary::LoadFromDirectory(config.pattern_dir), config.recognizer_options) {
    if (config.pattern_dir.empty()) {
        throw VisionInitError("Empty pattern directory");
    }
}

namespace {

inline float gray_at(const cv::Mat& frame, int x, int y) {
    if (x < 0 || y < 0 || x >= frame.cols || y >= frame.rows) {
        return 0.0f;
    }
    const cv::Vec3b& px = frame.at<cv::Vec3b>(y, x);
    return static_cast<float>(px[0] + px[1] + px[2]) / 765.0f;
}

}  // namespace

void FramePipeline::clear_patches() {
    patches_.clear();
    patch_ready_ = false;
}

bool FramePipeline::init_patches(const cv::Mat& frame, const CornerQuad& corners) {
    if (frame.empty() || !corners.valid()) {
        return false;
    }
    const int patch_size = std::max(4, config_.patch_size);
    const int half = patch_size / 2;
    patches_.clear();
    patches_.reserve(4);
    const std::array<cv::Point2f, 4> points = {corners.tl, corners.tr, corners.bl, corners.br};

    for (const auto& pt : points) {
        PatchState patch;
        patch.center = pt;
        patch.gray.assign(static_cast<size_t>(patch_size * patch_size), 0.0f);
        const int x0 = static_cast<int>(std::lround(pt.x)) - half;
        const int y0 = static_cast<int>(std::lround(pt.y)) - half;
        size_t k = 0;
        for (int py = 0; py < patch_size; ++py) {
            for (int px = 0; px < patch_size; ++px) {
                patch.gray[k++] = gray_at(frame, x0 + px, y0 + py);
            }
        }
        patches_.push_back(std::move(patch));
    }
    last_corners_ = corners;
    patch_ready_ = true;
    return true;
}

bool FramePipeline::track_patches(const cv::Mat& frame, CornerQuad* out) {
    if (!patch_ready_ || patches_.size() != 4 || frame.empty()) {
        return false;
    }
    const int patch_size = std::max(4, config_.patch_size);
    const int search = std::max(1, config_.patch_search);
    const float sad_max = std::max(0.0f, config_.patch_sad_max);
    const int half = patch_size / 2;
    std::array<cv::Point2f, 4> points;

    for (size_t i = 0; i < patches_.size(); ++i) {
        PatchState& patch = patches_[i];
        const int base_x0 = static_cast<int>(std::lround(patch.center.x)) - half - search;
        const int base_y0 = static_cast<int>(std::lround(patch.center.y)) - half - search;
        float best_sad = std::numeric_limits<float>::infinity();
        int best_dx = 0;
        int best_dy = 0;

        for (int dy = -search; dy <= search; ++dy) {
            for (int dx = -search; dx <= search; ++dx) {
                float sad = 0.0f;
                const int ox = search + dx;
                const int oy = search + dy;
                size_t k = 0;
                for (int py = 0; py < patch_size; ++py) {
                    const int sy = base_y0 + oy + py;
                    for (int px = 0; px < patch_size; ++px) {
                        const int sx = base_x0 + ox + px;
                        const float g = gray_at(frame, sx, sy);
                        sad += std::abs(g - patch.gray[k++]);
                    }
                }
                if (sad < best_sad) {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        const float sad_norm = best_sad / static_cast<float>(patch_size * patch_size);
        if (sad_norm > sad_max) {
            clear_patches();
            return false;
        }

        patch.center.x += static_cast<float>(best_dx);
        patch.center.y += static_cast<float>(best_dy);
        points[i] = patch.center;
    }

    if (out) {
        out->tl = points[0];
        out->tr = points[1];
        out->bl = points[2];
        out->br = points[3];
        out->out_size = last_corners_.out_size;
    }
    last_corners_ = *out;
    return out->valid();
}

PipelineResult FramePipeline::Process(const cv::Mat& frame) {
    PipelineResult result;
    if (frame.empty()) {
        return result;
    }

    bool localized = false;
    CornerQuad corners;
    if (config_.patch_track_enabled && patch_ready_) {
        if (track_patches(frame, &corners)) {
            localized = true;
            result.localize.ok = true;
            result.localize.corners = corners;
            result.localize.inference_ms = 0.0;
            result.localize.source = "patch-track";
        }
    }

    if (!localized) {
        const std::optional<LocalizeResult> localize = localizer_.Locate(frame);
        if (!localize.has_value() || !localize->ok || !localize->corners.valid()) {
            clear_patches();
            return result;
        }
        result.localize = *localize;
        result.localized = true;
        corners = localize->corners;
        if (config_.patch_track_enabled) {
            init_patches(frame, corners);
        }
    }

    result.localized = true;
    
    try {
        result.deskewed_image = Deskewer::Deskew(
            frame,
            corners,
            config_.deskew_width,
            config_.deskew_height,
            config_.deskew_expand,
            config_.deskew_canonical_inset,
            cv::INTER_NEAREST);
    } catch (const VisionDeskewError& e) {
        clear_patches();
        return result;
    }
    
    result.deskewed = !result.deskewed_image.empty();
    if (!result.deskewed) {
        return result;
    }

    result.recognize = recognizer_.Decode(result.deskewed_image);
    result.recognized = result.recognize.ok;
    return result;
}

}  // namespace camdrop::vision
