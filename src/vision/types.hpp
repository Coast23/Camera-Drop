#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "util/config.hpp"

namespace camdrop::vision {

constexpr int kImageWidth = Config::IMG_WIDTH;
constexpr int kImageHeight = Config::IMG_HEIGHT;
constexpr int kGridRows = Config::GRID_R;
constexpr int kGridCols = Config::GRID_C;
constexpr int kStride = Config::STRIDE;
constexpr int kMargin = Config::MARGIN;
constexpr int kTileSize = Config::TILE_SIZE;
constexpr int kNumPatterns = 16;
constexpr int kNumColors = 4;
constexpr int kAnchorOutStart = Config::ANCHOR_OUT_START;
constexpr int kAnchorL1Size = Config::ANCHOR_L1_SIZE;
constexpr int kAnchorL2Inset = Config::ANCHOR_L2_INSET;
constexpr int kAnchorL2Size = Config::ANCHOR_L2_SIZE;
constexpr int kAnchorL3Inset = Config::ANCHOR_L3_INSET;
constexpr int kAnchorL3Size = Config::ANCHOR_L3_SIZE;
constexpr int kAnchorL4Inset = Config::ANCHOR_L4_INSET;
constexpr int kAnchorL4Size = Config::ANCHOR_L4_SIZE;

struct Detection {
    cv::Rect2f box;
    float score = 0.0f;
    int cls = -1;
};

struct CornerQuad {
    cv::Point2f tl;
    cv::Point2f tr;
    cv::Point2f bl;
    cv::Point2f br;
    int out_size = std::max(kImageWidth, kImageHeight);

    [[nodiscard]] bool valid() const {
        return std::isfinite(tl.x) && std::isfinite(tl.y)
            && std::isfinite(tr.x) && std::isfinite(tr.y)
            && std::isfinite(bl.x) && std::isfinite(bl.y)
            && std::isfinite(br.x) && std::isfinite(br.y);
    }
};

struct LocalizeResult {
    bool ok = false;
    CornerQuad corners;
    std::vector<Detection> detections;
    cv::Rect2f frame_box;
    bool has_frame = false;
    bool used_center_fallback = false;
    double inference_ms = 0.0;
    std::string source = "yolo";
};

struct RecognizeResult {
    bool ok = false;
    int pattern_bits = 0;
    int header_tail_bits = 0;
    int payload_tail_bits = 0;
    double avg_pattern_dist = 0.0;
    std::vector<uint8_t> header_symbols;
    std::vector<uint8_t> payload_symbols;
    std::vector<uint8_t> header_bytes;
    std::vector<uint8_t> payload_bytes;
};

struct PipelineResult {
    bool localized = false;
    bool deskewed = false;
    bool recognized = false;
    LocalizeResult localize;
    cv::Mat deskewed_image;
    RecognizeResult recognize;
};

}  // namespace camdrop::vision
