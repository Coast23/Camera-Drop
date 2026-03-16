#include "vision/frame_renderer.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "util/config.hpp"
#include "util/errors.hpp"
#include "vision/visual_frame_codec.hpp"

namespace camdrop::vision {
namespace {

cv::Vec3b get_color_bgr(int color_idx) {
    switch (color_idx) {
        case 0: return cv::Vec3b(0, 255, 255);
        case 1: return cv::Vec3b(0, 255, 0);
        case 2: return cv::Vec3b(255, 255, 0);
        case 3: return cv::Vec3b(255, 0, 255);
        default: return cv::Vec3b(255, 255, 255);
    }
}

bool is_anchor_reserved(int r, int c) {
    if (r < Config::ANCHOR_RESERVED_CELLS && c < Config::ANCHOR_RESERVED_CELLS) return true;
    if (r < Config::ANCHOR_RESERVED_CELLS && c >= Config::GRID_C - Config::ANCHOR_RESERVED_CELLS) return true;
    if (r >= Config::GRID_R - Config::ANCHOR_RESERVED_CELLS && c < Config::ANCHOR_RESERVED_CELLS) return true;
    if (r >= Config::GRID_R - Config::ANCHOR_RESERVED_CELLS && c >= Config::GRID_C - Config::ANCHOR_RESERVED_CELLS) return true;
    return false;
}

bool is_calibration_cell(int r, int c) {
    return r == Config::CALIB_ROW && c >= Config::CALIB_COL_BEGIN && c < Config::CALIB_COL_END;
}

uint8_t calibration_symbol_for_cell(int col, int pattern_bits) {
    return static_cast<uint8_t>((((col - Config::CALIB_COL_BEGIN) & 3) << pattern_bits) | 0);
}

void draw_normal_anchor(cv::Mat& img, int x0, int y0) {
    cv::rectangle(img, cv::Rect(x0, y0, Config::ANCHOR_L1_SIZE, Config::ANCHOR_L1_SIZE), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + Config::ANCHOR_L2_INSET, y0 + Config::ANCHOR_L2_INSET, Config::ANCHOR_L2_SIZE, Config::ANCHOR_L2_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + Config::ANCHOR_L3_INSET, y0 + Config::ANCHOR_L3_INSET, Config::ANCHOR_L3_SIZE, Config::ANCHOR_L3_SIZE), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + Config::ANCHOR_L4_INSET, y0 + Config::ANCHOR_L4_INSET, Config::ANCHOR_L4_SIZE, Config::ANCHOR_L4_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
}

void draw_br_anchor(cv::Mat& img, int x0, int y0) {
    const int h1 = Config::ANCHOR_L1_SIZE / 2;
    cv::rectangle(img, cv::Rect(x0, y0, h1, h1), cv::Scalar(0, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + h1, y0, h1, h1), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, y0 + h1, h1, h1), cv::Scalar(255, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + h1, y0 + h1, h1, h1), cv::Scalar(255, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + Config::ANCHOR_L2_INSET, y0 + Config::ANCHOR_L2_INSET, Config::ANCHOR_L2_SIZE, Config::ANCHOR_L2_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
    const int h3 = Config::ANCHOR_L3_SIZE / 2;
    const int ix = x0 + Config::ANCHOR_L3_INSET;
    const int iy = y0 + Config::ANCHOR_L3_INSET;
    cv::rectangle(img, cv::Rect(ix, iy, h3, h3), cv::Scalar(0, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix + h3, iy, h3, h3), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix, iy + h3, h3, h3), cv::Scalar(255, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix + h3, iy + h3, h3, h3), cv::Scalar(255, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + Config::ANCHOR_L4_INSET, y0 + Config::ANCHOR_L4_INSET, Config::ANCHOR_L4_SIZE, Config::ANCHOR_L4_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
}

/**
 * @brief 在图像上绘制符号
 * @param img 要绘制的图像
 * @param grid_r 网格行索引
 * @param grid_c 网格列索引
 * @param symbol 要绘制的符号值
 * @param dict 字典
 */
void draw_symbol_tile(cv::Mat& img,
                      int grid_r,
                      int grid_c,
                      uint8_t symbol,
                      const PatternDictionary& dict) {
    const int pat_mask = dict.size() - 1;
    const int pat_idx = symbol & pat_mask;
    const int color_idx = symbol >> dict.pattern_bits();
    if (pat_idx < 0 || pat_idx >= dict.size()) {
        throw ImageFormatError("Symbol pattern index " + std::to_string(pat_idx) + 
                              " out of range [0, " + std::to_string(dict.size()) + ")");
    }
    const uint64_t mask = dict.masks64[pat_idx];
    const cv::Vec3b color = get_color_bgr(color_idx);
    const int start_x = Config::MARGIN + grid_c * Config::STRIDE;
    const int start_y = Config::MARGIN + grid_r * Config::STRIDE;
    for (int pr = 0; pr < Config::TILE_SIZE; ++pr) {
        for (int pc = 0; pc < Config::TILE_SIZE; ++pc) {
            if ((mask >> (pr * 8 + pc)) & 1ULL) {
                img.at<cv::Vec3b>(start_y + pr, start_x + pc) = color;
            }
        }
    }
}

void validate_dict(const PatternDictionary& dict) {
    if (dict.empty()) {
        throw PatternDictInvalidError("Pattern dictionary is empty");
    }
    const int size = dict.size();
    if ((size & (size - 1)) != 0) {
        throw PatternDictInvalidError("Pattern dictionary size " + std::to_string(size) + 
                                     " is not a power of two");
    }
}

}  // namespace

/**
 * @brief PatternFrameRenderer构造函数
 * @param dict 模式字典，用于渲染符号
 */
PatternFrameRenderer::PatternFrameRenderer(PatternDictionary dict)
    : dict_(std::move(dict)) {
    validate_dict(dict_);
}

/**
 * @brief 渲染帧字节为图像
 * @param frame_bytes 要渲染的帧字节数据
 * @return 渲染后的BGR图像
 */
cv::Mat PatternFrameRenderer::Render(const std::vector<uint8_t>& frame_bytes) const {
    std::vector<uint8_t> symbols;
    try {
        symbols = FrameBytesToInterleavedSymbols(frame_bytes);
    } catch (const ImageError& e) {
        throw ImageFormatError(std::string("Failed to convert frame bytes: ") + e.what());
    }
    return RenderInterleavedSymbols(symbols);
}

/**
 * @brief 渲染交织符号为图像
 * @param interleaved_symbols 要渲染的交织符号数据
 * @return 渲染后的BGR图像，包含锚点、校准单元和数据符号
 */
cv::Mat PatternFrameRenderer::RenderInterleavedSymbols(const std::vector<uint8_t>& interleaved_symbols) const {
    if (interleaved_symbols.size() != Config::UINTS_COUNT) {
        throw ImageSizeError("Render symbol count " + std::to_string(interleaved_symbols.size()) + 
                            " != UINTS_COUNT " + std::to_string(Config::UINTS_COUNT));
    }

    cv::Mat img = cv::Mat::zeros(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);
    draw_normal_anchor(img, Config::ANCHOR_OUT_START, Config::ANCHOR_OUT_START);
    draw_normal_anchor(img, Config::IMG_WIDTH - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE, Config::ANCHOR_OUT_START);
    draw_normal_anchor(img, Config::ANCHOR_OUT_START, Config::IMG_HEIGHT - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE);
    draw_br_anchor(img, Config::IMG_WIDTH - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE, Config::IMG_HEIGHT - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE);

    size_t idx = 0;
    for (int r = 0; r < Config::GRID_R; ++r) {
        for (int c = 0; c < Config::GRID_C; ++c) {
            if (is_anchor_reserved(r, c)) {
                continue;
            }
            if (is_calibration_cell(r, c)) {
                draw_symbol_tile(img, r, c, calibration_symbol_for_cell(c, dict_.pattern_bits()), dict_);
                continue;
            }
            draw_symbol_tile(img, r, c, interleaved_symbols[idx++], dict_);
        }
    }

    if (idx != interleaved_symbols.size()) {
        throw ImageFormatError("Render symbol layout mismatch: processed " + std::to_string(idx) + 
                              " symbols, expected " + std::to_string(interleaved_symbols.size()));
    }
    return img;
}

}  // namespace camdrop::vision
