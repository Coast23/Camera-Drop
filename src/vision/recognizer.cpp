#include "vision/recognizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

namespace camdrop::vision {
namespace {
#ifdef _MSC_VER
#  include <intrin.h>
static inline int popcount32(uint32_t x) { return static_cast<int>(__popcnt(x)); }
static inline int popcount64(uint64_t x) { return static_cast<int>(__popcnt64(x)); }
#else
static inline int popcount32(uint32_t x) { return __builtin_popcount(x); }
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
#endif

static constexpr int GRID_ROWS = Config::GRID_R;
static constexpr int GRID_COLS = Config::GRID_C;
static constexpr int STRIDE = Config::STRIDE;
static constexpr int MARGIN = Config::MARGIN;
static constexpr int IMG_W = Config::IMG_WIDTH;
static constexpr int IMG_H = Config::IMG_HEIGHT;
static constexpr int TILE_SIZE = Config::TILE_SIZE;
static constexpr int CELL_SAMPLE_SIZE = TILE_SIZE + 2;
static constexpr int SAMPLE_AREA = CELL_SAMPLE_SIZE * CELL_SAMPLE_SIZE;
static constexpr int NUM_COLORS = 4;
static constexpr int ANCHOR_OUT_START = Config::ANCHOR_OUT_START;
static constexpr int ANCHOR_L1_SIZE = Config::ANCHOR_L1_SIZE;
static constexpr int ANCHOR_L2_INSET = Config::ANCHOR_L2_INSET;
static constexpr int ANCHOR_L2_SIZE = Config::ANCHOR_L2_SIZE;
static constexpr int ANCHOR_L3_INSET = Config::ANCHOR_L3_INSET;
static constexpr int ANCHOR_L3_SIZE = Config::ANCHOR_L3_SIZE;
static constexpr int ANCHOR_L4_INSET = Config::ANCHOR_L4_INSET;
static constexpr int ANCHOR_L4_SIZE = Config::ANCHOR_L4_SIZE;
static constexpr int DRIFT_MAX = 7;
static constexpr uint8_t COOL_INIT = 0xFE;
static constexpr uint8_t COOL_NONE = 0xFF;
static constexpr uint16_t PRIO_INIT = 0xFFFE;
static constexpr int HASH_FAST_N = 5;
static constexpr int HEAP_IDX_BITS = 14;
static constexpr int HEAP_IDX_MASK = (1 << HEAP_IDX_BITS) - 1;
static constexpr double BEST_COLOR_FLOOR = 48.0;
static constexpr double COLOR_VOTE_MIN_SPAN = 12.0;
static constexpr double COLOR_VOTE_MIN_GAP = 6.0;
static constexpr double COLOR_VOTE_STRONG_SPAN = 32.0;
static constexpr double COLOR_VOTE_STRONG_GAP = 12.0;
static constexpr double COLOR_VOTE_ABS_WEIGHT = 0.35;
static constexpr double COLOR_VOTE_REL_WEIGHT = 0.65;
static constexpr int LUMA_RECHECK_DIST64 = 5;
static constexpr int LUMA_RECHECK_DIST16 = 1;
static constexpr int BINARY_BLOCK_SIZE = 5;
static constexpr int BINARY_SHARP_BLOCK_SIZE = 7;
static constexpr int BINARY_THRESHOLD_BIAS = 0;
static constexpr int BITGRID_RECHECK_DIST64 = 8;
static constexpr int BITGRID_RECHECK_DIST16 = 1;
static constexpr int BITGRID_ACCEPT_GAIN = 2;
static constexpr int BITGRID_ACCEPT_GAIN_HINT = 1;

enum CellKind : uint8_t {
    CELL_KIND_CAL = 0,
    CELL_KIND_HEADER = 1,
    CELL_KIND_PAYLOAD = 2,
};

struct Rgb {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

struct ColorMatch {
    int idx = 0;
    double dist = 0.0;
    double second_dist = 0.0;
    double span = 0.0;
};

struct ColorCalibration {
    std::array<double, 3> bias {{0.0, 0.0, 0.0}};
    std::array<double, 9> matrix {{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    }};
    bool matrix_active = false;
    std::array<Rgb, NUM_COLORS> refs;
    std::array<Rgb, NUM_COLORS> vote_refs;
};

struct PatternDict {
    std::vector<uint64_t> masks64;
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    std::vector<uint16_t> masks16;
};

struct EncodedFrame {
    cv::Mat img;
    std::vector<uint8_t> raw;
};

struct DecodeLayout {
    int count = 0;
    std::vector<int16_t> x;
    std::vector<int16_t> y;
    std::vector<int16_t> row;
    std::vector<int16_t> col;
    std::vector<uint8_t> kind;
    std::vector<int32_t> neighbors;
    std::vector<uint32_t> seeds;
};

struct DecodeBuffers {
    std::vector<uint8_t> pending;
    std::vector<int8_t> drift_x;
    std::vector<int8_t> drift_y;
    std::vector<uint16_t> priority;
    std::vector<uint8_t> cooldown;
    std::vector<uint8_t> symbol;
    std::vector<uint8_t> gray_frame;
    std::vector<uint8_t> gray_temp;
    std::vector<uint8_t> luma_frame;
    std::vector<uint8_t> bin_frame;
    std::vector<uint32_t> sat_frame;
    std::array<uint8_t, SAMPLE_AREA> cell10 {};
    std::array<uint16_t, 16> block16 {};
};

struct SignalFrameStats {
    double luma_mean = 0.0;
    double luma_std = 0.0;
    double gray_mean = 0.0;
    double hi_clip_ratio = 0.0;
    double lo_clip_ratio = 0.0;
    bool washed_out = false;
    bool low_contrast = false;
};

struct PreprocessedFrames {
    const std::vector<uint8_t>* primary_frame = nullptr;
    const std::vector<uint8_t>* luma_frame = nullptr;
    const std::vector<uint8_t>* bitgrid_frame = nullptr;
    bool luma_hint = false;
    bool bitgrid_hint = false;
    SignalFrameStats frame_stats;
};

struct CandidateHit {
    int best_pat = 0;
    int best_dist16 = 17;
    int best_dist64 = 65;
    int best_dx = 0;
    int best_dy = 0;
    int best_sample_x = 0;
    int best_sample_y = 0;
    int best_radius = 0;
};

struct DecodedCell {
    uint8_t symbol = 0;
    uint16_t best_dist = 0;
    uint8_t drift_idx = 4;
    int8_t drift_x = 0;
    int8_t drift_y = 0;
};

struct DecodeStats {
    int symbol_correct = 0;
    int pattern_correct = 0;
    int color_correct = 0;
    int total = 0;
};

struct WeightedStats {
    double symbol_correct = 0.0;
    double pattern_correct = 0.0;
    double color_correct = 0.0;
    double total = 0.0;
};

struct ScoreConfig {
    double symbol_weight = 0.74;
    double pattern_weight = 0.14;
    double color_weight = 0.12;
    double sparse_penalty_weight = 0.06;
    double balance_penalty_weight = 0.03;
    double min_fill = 20.0;
    double max_fill = 44.0;
    double fragility_penalty_weight = 0.04;
    double distance64_penalty_weight = 0.08;
    double distance16_penalty_weight = 0.06;
    double shift_penalty_weight = 0.09;
};

struct OffsetCase {
    double dx = 0.0;
    double dy = 0.0;
    double weight = 1.0;
};

static const std::array<cv::Vec3b, NUM_COLORS> COLORS_BGR = {{
    cv::Vec3b(0, 255, 255),
    cv::Vec3b(0, 255, 0),
    cv::Vec3b(255, 255, 0),
    cv::Vec3b(255, 0, 255),
}};

static const std::array<std::array<int, 2>, 9> DRIFT_PAIRS = {{
    {{-1, -1}}, {{0, -1}}, {{1, -1}},
    {{-1, 0}},  {{0, 0}},  {{1, 0}},
    {{-1, 1}},  {{0, 1}},  {{1, 1}},
}};

static const std::array<int, 9> HASH_ORDER = {{4, 5, 7, 3, 1, 8, 0, 2, 6}};
static const std::array<std::array<uint8_t, 64>, 9> SUBWINDOW_MAP = []() {
    std::array<std::array<uint8_t, 64>, 9> out {};
    for (int idx = 0; idx < 9; ++idx) {
        const int ox = idx % 3;
        const int oy = idx / 3;
        int k = 0;
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                out[idx][k++] = static_cast<uint8_t>((oy + r) * CELL_SAMPLE_SIZE + (ox + c));
            }
        }
    }
    return out;
}();
static const std::array<uint8_t, 64> BLOCK16_MAP = []() {
    std::array<uint8_t, 64> out {};
    for (int i = 0; i < 64; ++i) {
        const int r = i >> 3;
        const int c = i & 7;
        out[i] = static_cast<uint8_t>(((r >> 1) * 4) + (c >> 1));
    }
    return out;
}();
static const std::array<std::array<int, 2>, 25> SEARCH_EXTENDED = {{
    {{0, 0}}, {{1, 0}}, {{0, 1}}, {{-1, 0}}, {{0, -1}},
    {{1, 1}}, {{-1, -1}}, {{1, -1}}, {{-1, 1}},
    {{2, 0}}, {{0, 2}}, {{-2, 0}}, {{0, -2}},
    {{2, 1}}, {{1, 2}}, {{-1, 2}}, {{-2, 1}},
    {{-2, -1}}, {{-1, -2}}, {{1, -2}}, {{2, -1}},
    {{2, 2}}, {{-2, -2}}, {{2, -2}}, {{-2, 2}},
}};

static const std::array<OffsetCase, 6> OFFSET_CASES = {{
    {0.0, 0.0, 0.75},
    {0.75, 0.0, 1.00},
    {0.0, 0.75, 1.00},
    {1.25, 0.5, 1.18},
    {0.5, 1.25, 1.18},
    {1.5, 1.5, 1.28},
}};

/**
 * @brief 根据字典大小计算模式位数
 * @param n 字典大小
 * @return 所需的位数
 */
static inline int pattern_bits_for_dict(int n) {
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    return bits;
}

/**
 * @brief 将整数值限制在指定范围内
 * @param v 输入值
 * @param lo 下限
 * @param hi 上限
 * @return 限制后的值
 */
static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * @brief 将堆节点索引和优先级打包为32位值
 * @param idx 节点索引
 * @param prio 优先级
 * @return 打包后的32位值
 */
static inline uint32_t pack_heap_node(int idx, uint16_t prio) {
    return (static_cast<uint32_t>(prio) << HEAP_IDX_BITS) | static_cast<uint32_t>(idx & HEAP_IDX_MASK);
}

/**
 * @brief 从打包的堆节点值中解包索引
 * @param node 打包的堆节点值
 * @return 节点索引
 */
static inline int unpack_heap_idx(uint32_t node) {
    return static_cast<int>(node & HEAP_IDX_MASK);
}

/**
 * @brief 检查指定网格位置是否为锚点保留单元
 * @param r 行索引
 * @param c 列索引
 * @return 如果是保留单元则返回true，否则false
 */
static inline bool is_anchor_reserved(int r, int c) {
    if (r < Config::ANCHOR_RESERVED_CELLS && c < Config::ANCHOR_RESERVED_CELLS) return true;
    if (r < Config::ANCHOR_RESERVED_CELLS && c >= GRID_COLS - Config::ANCHOR_RESERVED_CELLS) return true;
    if (r >= GRID_ROWS - Config::ANCHOR_RESERVED_CELLS && c < Config::ANCHOR_RESERVED_CELLS) return true;
    if (r >= GRID_ROWS - Config::ANCHOR_RESERVED_CELLS && c >= GRID_COLS - Config::ANCHOR_RESERVED_CELLS) return true;
    return false;
}

/**
 * @brief 检查指定网格位置是否为校准单元
 * @param r 行索引
 * @param c 列索引
 * @return 如果是校准单元则返回true，否则false
 */
static inline bool is_calibration_cell(int r, int c) {
    return r == Config::CALIB_ROW && c >= Config::CALIB_COL_BEGIN && c < Config::CALIB_COL_END;
}

/**
 * @brief 检查指定网格位置是否为头部单元
 * @param r 行索引
 * @param c 列索引
 * @return 如果是头部单元则返回true，否则false
 */
static inline bool is_header_cell(int r, int c) {
    return r == Config::HEADER_ROW && c >= Config::HEADER_COL_BEGIN && c < Config::HEADER_COL_END;
}

/**
 * @brief 检查指定网格位置是否为有效载荷单元
 * @param r 行索引
 * @param c 列索引
 * @return 如果是有效载荷单元则返回true，否则false
 */
static inline bool is_payload_cell(int r, int c) {
    return !is_anchor_reserved(r, c) && !is_calibration_cell(r, c) && !is_header_cell(r, c);
}

/**
 * @brief 检查掩码中的指定位是否开启
 * @param mask_lo 低32位掩码
 * @param mask_hi 高32位掩码
 * @param bit 位索引 (0-63)
 * @return 如果位开启则返回true，否则false
 */
static inline bool mask_is_on(uint32_t mask_lo, uint32_t mask_hi, int bit) {
    if (bit < 32) return ((mask_lo >> bit) & 1U) != 0;
    return ((mask_hi >> (bit - 32)) & 1U) != 0;
}

/**
 * @brief 将64位掩码分割为两个32位部分
 * @param mask 64位掩码
 * @return 包含低32位和高32位的pair
 */
static inline std::pair<uint32_t, uint32_t> split_mask64(uint64_t mask) {
    return {
        static_cast<uint32_t>(mask & 0xFFFFFFFFULL),
        static_cast<uint32_t>((mask >> 32) & 0xFFFFFFFFULL)
    };
}

/**
 * @brief 将64位掩码压缩为16位掩码
 * @param mask_lo 低32位掩码
 * @param mask_hi 高32位掩码
 * @return 压缩后的16位掩码
 */
static uint16_t compress_mask64_to_16(uint32_t mask_lo, uint32_t mask_hi) {
    uint16_t out = 0;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const int base = (r << 4) + (c << 1);
            int on = 0;
            on += mask_is_on(mask_lo, mask_hi, base) ? 1 : 0;
            on += mask_is_on(mask_lo, mask_hi, base + 1) ? 1 : 0;
            on += mask_is_on(mask_lo, mask_hi, base + 8) ? 1 : 0;
            on += mask_is_on(mask_lo, mask_hi, base + 9) ? 1 : 0;
            if (on >= 2) out = static_cast<uint16_t>(out | (1U << (r * 4 + c)));
        }
    }
    return out;
}

/**
 * @brief 从64位掩码向量构建模式字典
 * @param masks64 64位掩码向量
 * @return 构建的模式字典
 */
static PatternDict build_pattern_dict(const std::vector<uint64_t>& masks64) {
    PatternDict dict;
    dict.masks64 = masks64;
    dict.lo.resize(masks64.size());
    dict.hi.resize(masks64.size());
    dict.masks16.resize(masks64.size());
    for (size_t i = 0; i < masks64.size(); ++i) {
        const auto parts = split_mask64(masks64[i]);
        dict.lo[i] = parts.first;
        dict.hi[i] = parts.second;
        dict.masks16[i] = compress_mask64_to_16(parts.first, parts.second);
    }
    return dict;
}

/**
 * @brief 将64位掩码压缩为16位掩码
 * @param mask 64位掩码
 * @return 压缩后的16位掩码
 */
static inline uint16_t compress_mask64_to_16(uint64_t mask) {
    const auto parts = split_mask64(mask);
    return compress_mask64_to_16(parts.first, parts.second);
}

/**
 * @brief 检查64位掩码中的指定位置是否开启
 * @param mask 64位掩码
 * @param x x坐标 (0-7)
 * @param y y坐标 (0-7)
 * @return 如果位置开启则返回true，否则false
 */
static inline bool mask64_is_on(uint64_t mask, int x, int y) {
    return ((mask >> (y * 8 + x)) & 1ULL) != 0ULL;
}

/**
 * @brief 平移64位掩码
 * @param mask 原始64位掩码
 * @param dx x方向偏移
 * @param dy y方向偏移
 * @return 平移后的64位掩码
 */
static uint64_t translate_mask64(uint64_t mask, int dx, int dy) {
    uint64_t out = 0;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            if (!mask64_is_on(mask, x, y)) continue;
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8) continue;
            out |= (1ULL << (ny * 8 + nx));
        }
    }
    return out;
}

/**
 * @brief 采样矩形区域的平均RGB值
 * @param img 输入图像
 * @param x0 矩形左上角x坐标
 * @param y0 矩形左上角y坐标
 * @param size 矩形边长
 * @return 平均RGB值
 */
static Rgb sample_rect_mean_rgb(const cv::Mat& img, int x0, int y0, int size) {
    const int sx = clamp_int(x0, 0, IMG_W - size);
    const int sy = clamp_int(y0, 0, IMG_H - size);
    double sr = 0.0;
    double sg = 0.0;
    double sb = 0.0;
    double n = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const cv::Vec3b px = img.at<cv::Vec3b>(sy + y, sx + x);
            sr += static_cast<double>(px[2]);
            sg += static_cast<double>(px[1]);
            sb += static_cast<double>(px[0]);
            n += 1.0;
        }
    }
    return {sr / n, sg / n, sb / n};
}

/**
 * @brief 采样矩形区域的选择性RGB值（根据评分函数选择像素）
 * @tparam ScoreFn 评分函数类型
 * @param img 输入图像
 * @param x0 矩形左上角x坐标
 * @param y0 矩形左上角y坐标
 * @param size 矩形边长
 * @param score_pixel 像素评分函数
 * @param keep_ratio 保留比例
 * @return 选择性平均RGB值
 */
template <typename ScoreFn>
static Rgb sample_rect_selective_rgb(const cv::Mat& img,
                                     int x0,
                                     int y0,
                                     int size,
                                     ScoreFn score_pixel,
                                     double keep_ratio) {
    const int sx = clamp_int(x0, 0, IMG_W - size);
    const int sy = clamp_int(y0, 0, IMG_H - size);
    std::vector<std::array<double, 4>> samples;
    samples.reserve(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const cv::Vec3b px = img.at<cv::Vec3b>(sy + y, sx + x);
            const double r = static_cast<double>(px[2]);
            const double g = static_cast<double>(px[1]);
            const double b = static_cast<double>(px[0]);
            samples.push_back({score_pixel(r, g, b), r, g, b});
        }
    }
    std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) {
        return a[0] > b[0];
    });
    const int total = static_cast<int>(samples.size());
    const int keep = std::max(4, std::min(total, static_cast<int>(std::lround(total * keep_ratio))));
    double sr = 0.0;
    double sg = 0.0;
    double sb = 0.0;
    for (int i = 0; i < keep; ++i) {
        sr += samples[i][1];
        sg += samples[i][2];
        sb += samples[i][3];
    }
    return {sr / keep, sg / keep, sb / keep};
}

/**
 * @brief 计算两个RGB值的平均值
 * @param a 第一个RGB值
 * @param b 第二个RGB值
 * @return 平均RGB值
 */
static Rgb average_rgb(const Rgb& a, const Rgb& b) {
    return {(a.r + b.r) * 0.5, (a.g + b.g) * 0.5, (a.b + b.b) * 0.5};
}

/**
 * @brief 计算RGB值向量的平均值
 * @param samples RGB值向量
 * @return 平均RGB值
 */
static Rgb average_rgbs(const std::vector<Rgb>& samples) {
    if (samples.empty()) return {};
    double sr = 0.0;
    double sg = 0.0;
    double sb = 0.0;
    for (const auto& rgb : samples) {
        sr += rgb.r;
        sg += rgb.g;
        sb += rgb.b;
    }
    const double inv = 1.0 / static_cast<double>(samples.size());
    return {sr * inv, sg * inv, sb * inv};
}

/**
 * @brief 计算两个RGB值的差值（非负）
 * @param a 被减数RGB值
 * @param b 减数RGB值
 * @return 差值RGB值（各分量不小于0）
 */
static Rgb subtract_rgb(const Rgb& a, const Rgb& b) {
    return {
        std::max(0.0, a.r - b.r),
        std::max(0.0, a.g - b.g),
        std::max(0.0, a.b - b.b),
    };
}

/**
 * @brief 计算像素颜色与指定颜色索引的匹配评分
 * @param r 红色分量
 * @param g 绿色分量
 * @param b 蓝色分量
 * @param color_idx 颜色索引 (0-3)
 * @return 匹配评分
 */
static double color_rect_score(double r, double g, double b, int color_idx) {
    switch (color_idx) {
        case 0: return (r + g) - (b * 2.0);
        case 1: return (g * 2.0) - (r + b);
        case 2: return (g + b) - (r * 2.0);
        case 3: return (r + b) - (g * 2.0);
        default: return 0.0;
    }
}

/**
 * @brief 采样矩形区域中指定颜色的强颜色RGB值
 * @param img 输入图像
 * @param x0 矩形左上角x坐标
 * @param y0 矩形左上角y坐标
 * @param size 矩形边长
 * @param color_idx 颜色索引
 * @return 强颜色平均RGB值
 */
static Rgb sample_rect_strong_color_rgb(const cv::Mat& img, int x0, int y0, int size, int color_idx) {
    return sample_rect_selective_rgb(
        img, x0, y0, size,
        [color_idx](double r, double g, double b) { return color_rect_score(r, g, b, color_idx); },
        0.45
    );
}

static Rgb sample_rect_dark_rgb(const cv::Mat& img, int x0, int y0, int size) {
    return sample_rect_selective_rgb(
        img, x0, y0, size,
        [](double r, double g, double b) { return -(r + g + b); },
        0.45
    );
}

static Rgb sample_rect_bright_rgb(const cv::Mat& img, int x0, int y0, int size) {
    return sample_rect_selective_rgb(
        img, x0, y0, size,
        [](double r, double g, double b) { return r + g + b; },
        0.45
    );
}

static Rgb sample_anchor_white_rgb(const cv::Mat& img, int base_x, int base_y) {
    const int outer_inset = 2;
    const int outer_size = 8;
    const int inner_base_x = base_x + ANCHOR_L3_INSET;
    const int inner_base_y = base_y + ANCHOR_L3_INSET;
    const int inner_half = ANCHOR_L3_SIZE >> 1;
    const int inner_inset = 1;
    const int inner_size = 6;
    return average_rgbs({
        sample_rect_bright_rgb(img, base_x + outer_inset, base_y + outer_inset, outer_size),
        sample_rect_bright_rgb(img, base_x + ANCHOR_L1_SIZE - outer_inset - outer_size, base_y + outer_inset, outer_size),
        sample_rect_bright_rgb(img, base_x + outer_inset, base_y + ANCHOR_L1_SIZE - outer_inset - outer_size, outer_size),
        sample_rect_bright_rgb(img, base_x + ANCHOR_L1_SIZE - outer_inset - outer_size, base_y + ANCHOR_L1_SIZE - outer_inset - outer_size, outer_size),
        sample_rect_bright_rgb(img, inner_base_x + inner_inset, inner_base_y + inner_inset, inner_size),
        sample_rect_bright_rgb(img, inner_base_x + inner_half + inner_inset, inner_base_y + inner_inset, inner_size),
        sample_rect_bright_rgb(img, inner_base_x + inner_inset, inner_base_y + inner_half + inner_inset, inner_size),
        sample_rect_bright_rgb(img, inner_base_x + inner_half + inner_inset, inner_base_y + inner_half + inner_inset, inner_size),
    });
}

static Rgb sample_anchor_black_rgb(const cv::Mat& img, int base_x, int base_y) {
    return average_rgbs({
        sample_rect_dark_rgb(img, base_x + ANCHOR_L2_INSET + 7, base_y + ANCHOR_L2_INSET + 7, 8),
        sample_rect_dark_rgb(img, base_x + ANCHOR_L4_INSET + 3, base_y + ANCHOR_L4_INSET + 3, 8),
    });
}

static bool invert3x3(const std::array<double, 9>& m, std::array<double, 9>& inv) {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];
    const double A = (e * i) - (f * h);
    const double B = -((d * i) - (f * g));
    const double C = (d * h) - (e * g);
    const double D = -((b * i) - (c * h));
    const double E = (a * i) - (c * g);
    const double F = -((a * h) - (b * g));
    const double G = (b * f) - (c * e);
    const double H = -((a * f) - (c * d));
    const double I = (a * e) - (b * d);
    const double det = (a * A) + (b * B) + (c * C);
    if (!std::isfinite(det) || std::abs(det) < 1e-6) return false;
    const double inv_det = 1.0 / det;
    inv = {{
        A * inv_det, D * inv_det, G * inv_det,
        B * inv_det, E * inv_det, H * inv_det,
        C * inv_det, F * inv_det, I * inv_det,
    }};
    return true;
}

static bool fit_linear_color_matrix(const std::vector<Rgb>& actual_rows,
                                    const std::vector<Rgb>& desired_rows,
                                    std::array<double, 9>& matrix) {
    std::array<double, 9> ata {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 3> atb_r {{0.0, 0.0, 0.0}};
    std::array<double, 3> atb_g {{0.0, 0.0, 0.0}};
    std::array<double, 3> atb_b {{0.0, 0.0, 0.0}};
    for (size_t i = 0; i < actual_rows.size(); ++i) {
        const double x = actual_rows[i].r;
        const double y = actual_rows[i].g;
        const double z = actual_rows[i].b;
        const Rgb& want = desired_rows[i];
        ata[0] += x * x;
        ata[1] += x * y;
        ata[2] += x * z;
        ata[3] += y * x;
        ata[4] += y * y;
        ata[5] += y * z;
        ata[6] += z * x;
        ata[7] += z * y;
        ata[8] += z * z;
        atb_r[0] += x * want.r;
        atb_r[1] += y * want.r;
        atb_r[2] += z * want.r;
        atb_g[0] += x * want.g;
        atb_g[1] += y * want.g;
        atb_g[2] += z * want.g;
        atb_b[0] += x * want.b;
        atb_b[1] += y * want.b;
        atb_b[2] += z * want.b;
    }
    std::array<double, 9> inv {};
    if (!invert3x3(ata, inv)) return false;
    auto solve = [&](const std::array<double, 3>& rhs) -> std::array<double, 3> {
        return {{
            (inv[0] * rhs[0]) + (inv[1] * rhs[1]) + (inv[2] * rhs[2]),
            (inv[3] * rhs[0]) + (inv[4] * rhs[1]) + (inv[5] * rhs[2]),
            (inv[6] * rhs[0]) + (inv[7] * rhs[1]) + (inv[8] * rhs[2]),
        }};
    };
    const auto row_r = solve(atb_r);
    const auto row_g = solve(atb_g);
    const auto row_b = solve(atb_b);
    matrix = {{
        row_r[0], row_r[1], row_r[2],
        row_g[0], row_g[1], row_g[2],
        row_b[0], row_b[1], row_b[2],
    }};
    return true;
}

static double compute_matrix_residual(const std::array<double, 9>& matrix,
                                      const std::vector<Rgb>& actual_rows,
                                      const std::vector<Rgb>& desired_rows) {
    double total = 0.0;
    int count = 0;
    for (size_t i = 0; i < actual_rows.size(); ++i) {
        const Rgb& row = actual_rows[i];
        const Rgb& want = desired_rows[i];
        const double r = (matrix[0] * row.r) + (matrix[1] * row.g) + (matrix[2] * row.b);
        const double g = (matrix[3] * row.r) + (matrix[4] * row.g) + (matrix[5] * row.b);
        const double b = (matrix[6] * row.r) + (matrix[7] * row.g) + (matrix[8] * row.b);
        total += (r - want.r) * (r - want.r);
        total += (g - want.g) * (g - want.g);
        total += (b - want.b) * (b - want.b);
        count += 3;
    }
    return count > 0 ? (total / static_cast<double>(count)) : std::numeric_limits<double>::infinity();
}

static void set_fallback_color_calibration(ColorCalibration& calib, const Rgb& black, const Rgb& white) {
    calib.bias = {{black.r, black.g, black.b}};
    calib.matrix = {{
        255.0 / std::max(16.0, white.r - black.r), 0.0, 0.0,
        0.0, 255.0 / std::max(16.0, white.g - black.g), 0.0,
        0.0, 0.0, 255.0 / std::max(16.0, white.b - black.b),
    }};
    calib.matrix_active = false;
}

static inline double clamp_byte(double v) {
    return v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
}

static Rgb apply_color_transform(double r, double g, double b, const ColorCalibration& calib) {
    const double x = r - calib.bias[0];
    const double y = g - calib.bias[1];
    const double z = b - calib.bias[2];
    return {
        clamp_byte((calib.matrix[0] * x) + (calib.matrix[1] * y) + (calib.matrix[2] * z)),
        clamp_byte((calib.matrix[3] * x) + (calib.matrix[4] * y) + (calib.matrix[5] * z)),
        clamp_byte((calib.matrix[6] * x) + (calib.matrix[7] * y) + (calib.matrix[8] * z)),
    };
}

static Rgb normalize_rgb_sample(const Rgb& rgb, const ColorCalibration& calib) {
    return apply_color_transform(rgb.r, rgb.g, rgb.b, calib);
}

static Rgb stretch_normalized_color_sample(const Rgb& rgb) {
    const double maxv = std::max({rgb.r, rgb.g, rgb.b, 1.0});
    double minv = std::min({rgb.r, rgb.g, rgb.b, BEST_COLOR_FLOOR});
    if (minv >= maxv) minv = 0.0;
    const double adjust = 255.0 / std::max(1.0, maxv - minv);
    return {
        clamp_byte((rgb.r - minv) * adjust),
        clamp_byte((rgb.g - minv) * adjust),
        clamp_byte((rgb.b - minv) * adjust),
    };
}

static double relative_color_dist(const Rgb& a, const Rgb& b) {
    const double arg = a.r - a.g;
    const double agb = a.g - a.b;
    const double abr = a.b - a.r;
    const double brg = b.r - b.g;
    const double bgb = b.g - b.b;
    const double bbr = b.b - b.r;
    const double d0 = arg - brg;
    const double d1 = agb - bgb;
    const double d2 = abr - bbr;
    return (d0 * d0) + (d1 * d1) + (d2 * d2);
}

static ColorCalibration estimate_color_calibration(const cv::Mat& img) {
    ColorCalibration calib;
    const int base_x = IMG_W - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int base_y = IMG_H - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int outer_inset = 2;
    const int outer_size = 8;
    const int inner_base_x = base_x + ANCHOR_L3_INSET;
    const int inner_base_y = base_y + ANCHOR_L3_INSET;
    const int inner_half = ANCHOR_L3_SIZE >> 1;
    const int inner_inset = 1;
    const int inner_size = 6;
    const int tl_base_x = ANCHOR_OUT_START;
    const int tl_base_y = ANCHOR_OUT_START;
    const int tr_base_x = IMG_W - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int tr_base_y = ANCHOR_OUT_START;
    const int bl_base_x = ANCHOR_OUT_START;
    const int bl_base_y = IMG_H - ANCHOR_OUT_START - ANCHOR_L1_SIZE;

    std::array<Rgb, NUM_COLORS> refs {};
    refs[0] = average_rgb(
        sample_rect_strong_color_rgb(img, base_x + outer_inset, base_y + outer_inset, outer_size, 0),
        sample_rect_strong_color_rgb(img, inner_base_x + inner_inset, inner_base_y + inner_inset, inner_size, 0)
    );
    refs[1] = average_rgb(
        sample_rect_strong_color_rgb(img, base_x + ANCHOR_L1_SIZE - outer_inset - outer_size, base_y + outer_inset, outer_size, 1),
        sample_rect_strong_color_rgb(img, inner_base_x + inner_half + inner_inset, inner_base_y + inner_inset, inner_size, 1)
    );
    refs[2] = average_rgb(
        sample_rect_strong_color_rgb(img, base_x + ANCHOR_L1_SIZE - outer_inset - outer_size, base_y + ANCHOR_L1_SIZE - outer_inset - outer_size, outer_size, 2),
        sample_rect_strong_color_rgb(img, inner_base_x + inner_half + inner_inset, inner_base_y + inner_half + inner_inset, inner_size, 2)
    );
    refs[3] = average_rgb(
        sample_rect_strong_color_rgb(img, base_x + outer_inset, base_y + ANCHOR_L1_SIZE - outer_inset - outer_size, outer_size, 3),
        sample_rect_strong_color_rgb(img, inner_base_x + inner_inset, inner_base_y + inner_half + inner_inset, inner_size, 3)
    );

    const Rgb white = average_rgbs({
        sample_anchor_white_rgb(img, tl_base_x, tl_base_y),
        sample_anchor_white_rgb(img, tr_base_x, tr_base_y),
        sample_anchor_white_rgb(img, bl_base_x, bl_base_y),
    });
    const Rgb black = average_rgbs({
        sample_anchor_black_rgb(img, tl_base_x, tl_base_y),
        sample_anchor_black_rgb(img, tr_base_x, tr_base_y),
        sample_anchor_black_rgb(img, bl_base_x, bl_base_y),
        sample_anchor_black_rgb(img, base_x, base_y),
    });

    set_fallback_color_calibration(calib, black, white);
    std::vector<Rgb> actual_rows;
    actual_rows.reserve(NUM_COLORS + 1);
    for (const auto& ref : refs) actual_rows.push_back(subtract_rgb(ref, black));
    actual_rows.push_back(subtract_rgb(white, black));
    const std::vector<Rgb> desired_rows = {
        {255.0, 255.0, 0.0},
        {0.0, 255.0, 0.0},
        {0.0, 255.0, 255.0},
        {255.0, 0.0, 255.0},
        {255.0, 255.0, 255.0},
    };
    std::array<double, 9> matrix {};
    if (fit_linear_color_matrix(actual_rows, desired_rows, matrix)) {
        const double residual = compute_matrix_residual(matrix, actual_rows, desired_rows);
        double max_coeff = 0.0;
        for (double coeff : matrix) max_coeff = std::max(max_coeff, std::abs(coeff));
        if (std::isfinite(residual) && residual <= 4800.0 && max_coeff <= 4.0) {
            calib.bias = {{black.r, black.g, black.b}};
            calib.matrix = matrix;
            calib.matrix_active = true;
        }
    }
    for (int i = 0; i < NUM_COLORS; ++i) {
        calib.refs[i] = Rgb{
            static_cast<double>(COLORS_BGR[i][2]),
            static_cast<double>(COLORS_BGR[i][1]),
            static_cast<double>(COLORS_BGR[i][0]),
        };
        calib.vote_refs[i] = stretch_normalized_color_sample(calib.refs[i]);
    }
    return calib;
}

static inline uint8_t compute_color_signal(double r, double g, double b) {
    const double maxv = std::max({r, g, b});
    const double minv = std::min({r, g, b});
    const double out = maxv - minv;
    return static_cast<uint8_t>(out < 0.0 ? 0.0 : (out > 255.0 ? 255.0 : out));
}

static SignalFrameStats build_signal_frames(const cv::Mat& img,
                                            const ColorCalibration& calib,
                                            std::vector<uint8_t>& gray_frame,
                                            std::vector<uint8_t>& luma_frame) {
    gray_frame.resize(IMG_W * IMG_H);
    luma_frame.resize(IMG_W * IMG_H);
    double sum_gray = 0.0;
    double sum_luma = 0.0;
    double sum_luma_sq = 0.0;
    int hi_clip = 0;
    int lo_clip = 0;
    for (int y = 0; y < IMG_H; ++y) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < IMG_W; ++x) {
            const cv::Vec3b& px = row[x];
            const Rgb rgb = apply_color_transform(
                static_cast<double>(px[2]),
                static_cast<double>(px[1]),
                static_cast<double>(px[0]),
                calib
            );
            const uint8_t gray = compute_color_signal(rgb.r, rgb.g, rgb.b);
            const uint8_t luma = static_cast<uint8_t>(
                (((rgb.r * 77.0) + (rgb.g * 150.0) + (rgb.b * 29.0)) / 256.0)
            );
            const int idx = y * IMG_W + x;
            gray_frame[idx] = gray;
            luma_frame[idx] = luma;
            sum_gray += static_cast<double>(gray);
            sum_luma += static_cast<double>(luma);
            sum_luma_sq += static_cast<double>(luma) * static_cast<double>(luma);
            const double maxv = std::max({rgb.r, rgb.g, rgb.b});
            if (maxv >= 248.0) ++hi_clip;
            if (maxv <= 16.0) ++lo_clip;
        }
    }
    const double n = std::max(1, IMG_W * IMG_H);
    const double luma_mean = sum_luma / n;
    const double gray_mean = sum_gray / n;
    const double luma_var = std::max(0.0, (sum_luma_sq / n) - (luma_mean * luma_mean));
    const double luma_std = std::sqrt(luma_var);
    const double hi_clip_ratio = static_cast<double>(hi_clip) / n;
    const double lo_clip_ratio = static_cast<double>(lo_clip) / n;
    const bool washed_out = hi_clip_ratio >= 0.020
        || (luma_mean >= 176.0 && gray_mean <= 34.0)
        || (luma_mean >= 188.0 && luma_std <= 34.0);
    const bool low_contrast = luma_std <= 26.0
        || gray_mean <= 18.0
        || (lo_clip_ratio >= 0.18 && luma_mean <= 84.0);
    return {
        luma_mean,
        luma_std,
        gray_mean,
        hi_clip_ratio,
        lo_clip_ratio,
        washed_out,
        low_contrast,
    };
}

static void sharpen_gray(std::vector<uint8_t>& gray_frame,
                         std::vector<uint8_t>& gray_temp,
                         int width,
                         int height,
                         double amount) {
    gray_temp = gray_frame;
    const double gain = std::isfinite(amount) ? amount : 0.6;
    for (int y = 1; y < height - 1; ++y) {
        const int row = y * width;
        for (int x = 1; x < width - 1; ++x) {
            const int idx = row + x;
            const int center = gray_frame[idx];
            const int lap = center * 4
                - gray_frame[idx - 1]
                - gray_frame[idx + 1]
                - gray_frame[idx - width]
                - gray_frame[idx + width];
            const double next = static_cast<double>(center) + gain * static_cast<double>(lap);
            gray_temp[idx] = static_cast<uint8_t>(clamp_int(static_cast<int>(std::lround(next)), 0, 255));
        }
    }
    gray_frame.swap(gray_temp);
}

static void build_integral_gray(const std::vector<uint8_t>& gray_frame,
                                std::vector<uint32_t>& sat_frame,
                                int width,
                                int height) {
    sat_frame.assign((width + 1) * (height + 1), 0);
    const int stride = width + 1;
    int src_idx = 0;
    for (int y = 0; y < height; ++y) {
        uint32_t row_sum = 0;
        const int sat_row = (y + 1) * stride;
        const int sat_prev = y * stride;
        for (int x = 0; x < width; ++x) {
            row_sum += gray_frame[src_idx++];
            sat_frame[sat_row + x + 1] = sat_frame[sat_prev + x + 1] + row_sum;
        }
    }
}

static void adaptive_threshold_gray(const std::vector<uint8_t>& gray_frame,
                                    std::vector<uint8_t>& bin_frame,
                                    const std::vector<uint32_t>& sat_frame,
                                    int width,
                                    int height,
                                    int block_size,
                                    int threshold_bias) {
    bin_frame.resize(width * height);
    const int stride = width + 1;
    const int radius = block_size >> 1;
    const int bias = threshold_bias;
    for (int y = 0; y < height; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(height - 1, y + radius);
        const int top = y0 * stride;
        const int bottom = (y1 + 1) * stride;
        const int row = y * width;
        for (int x = 0; x < width; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(width - 1, x + radius);
            const int area = (x1 - x0 + 1) * (y1 - y0 + 1);
            const uint32_t sum = sat_frame[bottom + x1 + 1]
                - sat_frame[top + x1 + 1]
                - sat_frame[bottom + x0]
                + sat_frame[top + x0];
            const int mean = static_cast<int>(sum / static_cast<uint32_t>(area));
            bin_frame[row + x] = gray_frame[row + x] > mean + bias ? 255U : 0U;
        }
    }
}

static PreprocessedFrames preprocess_symbol_frame(const cv::Mat& img,
                                                  const ColorCalibration& calib,
                                                  DecodeBuffers& buffers,
                                                  bool sharpen_hint,
                                                  double sharpen_strength) {
    const SignalFrameStats stats = build_signal_frames(
        img,
        calib,
        buffers.gray_frame,
        buffers.luma_frame
    );
    if (sharpen_hint && sharpen_strength > 0.0) {
        sharpen_gray(buffers.luma_frame, buffers.gray_temp, IMG_W, IMG_H, sharpen_strength);
    }
    build_integral_gray(buffers.luma_frame, buffers.sat_frame, IMG_W, IMG_H);
    const bool binary_hint = sharpen_hint || stats.washed_out || stats.low_contrast;
    adaptive_threshold_gray(
        buffers.luma_frame,
        buffers.bin_frame,
        buffers.sat_frame,
        IMG_W,
        IMG_H,
        binary_hint ? BINARY_SHARP_BLOCK_SIZE : BINARY_BLOCK_SIZE,
        BINARY_THRESHOLD_BIAS
    );
    return {
        &buffers.gray_frame,
        &buffers.luma_frame,
        &buffers.bin_frame,
        stats.washed_out || stats.low_contrast,
        binary_hint,
        stats,
    };
}

static ColorMatch nearest_color(const cv::Vec3b& px, const ColorCalibration& calib) {
    const Rgb rgb = apply_color_transform(
        static_cast<double>(px[2]),
        static_cast<double>(px[1]),
        static_cast<double>(px[0]),
        calib
    );
    const double span = std::max({rgb.r, rgb.g, rgb.b}) - std::min({rgb.r, rgb.g, rgb.b});
    const Rgb vote_sample = stretch_normalized_color_sample(rgb);
    int best = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    double second_dist = std::numeric_limits<double>::infinity();
    for (int i = 0; i < NUM_COLORS; ++i) {
        const Rgb& ref = calib.refs[i];
        const double dr = rgb.r - ref.r;
        const double dg = rgb.g - ref.g;
        const double db = rgb.b - ref.b;
        const double abs_dist = dr * dr + dg * dg + db * db;
        const double rel_dist = relative_color_dist(vote_sample, calib.vote_refs[i]);
        const double dist = abs_dist * COLOR_VOTE_ABS_WEIGHT + rel_dist * COLOR_VOTE_REL_WEIGHT;
        if (dist < min_dist) {
            second_dist = min_dist;
            min_dist = dist;
            best = i;
        } else if (dist < second_dist) {
            second_dist = dist;
        }
    }
    return {best, min_dist, second_dist, span};
}

struct CellSample10 {
    int sx = 0;
    int sy = 0;
    int sum = 0;
};

static CellSample10 sample_cell10(const std::vector<uint8_t>& signal_frame,
                                  int x0,
                                  int y0,
                                  std::array<uint8_t, SAMPLE_AREA>& cell10) {
    const int sx = clamp_int(x0 - 1, 0, IMG_W - CELL_SAMPLE_SIZE);
    const int sy = clamp_int(y0 - 1, 0, IMG_H - CELL_SAMPLE_SIZE);
    int sum = 0;
    int k = 0;
    for (int r = 0; r < CELL_SAMPLE_SIZE; ++r) {
        const int row = (sy + r) * IMG_W + sx;
        for (int c = 0; c < CELL_SAMPLE_SIZE; ++c) {
            const uint8_t v = signal_frame[row + c];
            cell10[k++] = v;
            sum += static_cast<int>(v);
        }
    }
    return {sx, sy, sum};
}

struct HashSample10 {
    uint32_t mask_lo = 0;
    uint32_t mask_hi = 0;
    uint16_t mask16 = 0;
};

static uint16_t hash_block16(const std::array<uint16_t, 16>& block16);

static HashSample10 hash_subwindow10(const std::array<uint8_t, SAMPLE_AREA>& cell10,
                                     int drift_idx,
                                     std::array<uint16_t, 16>& block16,
                                     double threshold) {
    const auto& map = SUBWINDOW_MAP[drift_idx];
    block16.fill(0);
    uint32_t mask_lo = 0;
    uint32_t mask_hi = 0;
    for (int i = 0; i < 64; ++i) {
        const uint8_t v = cell10[map[i]];
        if (static_cast<double>(v) > threshold) {
            if (i < 32) mask_lo |= (1U << i);
            else mask_hi |= (1U << (i - 32));
        }
        block16[BLOCK16_MAP[i]] = static_cast<uint16_t>(block16[BLOCK16_MAP[i]] + v);
    }
    return {mask_lo, mask_hi, hash_block16(block16)};
}

static uint16_t hash_block16(const std::array<uint16_t, 16>& block16) {
    int sum = 0;
    for (int i = 0; i < 16; ++i) sum += block16[i];
    const double threshold = static_cast<double>(sum) / 16.0;
    uint16_t mask = 0;
    for (int i = 0; i < 16; ++i) {
        if (static_cast<double>(block16[i]) > threshold) mask = static_cast<uint16_t>(mask | (1U << i));
    }
    return mask;
}

static inline std::tuple<int, int, int> match_pattern_combined(uint16_t mask16,
                                                               uint32_t mask_lo,
                                                               uint32_t mask_hi,
                                                               const PatternDict& dict) {
    int best_pat = 0;
    int best_dist64 = 65;
    int best_dist16 = 17;
    for (int i = 0; i < static_cast<int>(dict.masks64.size()); ++i) {
        const int d64 = popcount32(mask_lo ^ dict.lo[i]) + popcount32(mask_hi ^ dict.hi[i]);
        const int d16 = popcount32(static_cast<uint32_t>(mask16 ^ dict.masks16[i]));
        if (d64 < best_dist64 || (d64 == best_dist64 && d16 < best_dist16)) {
            best_pat = i;
            best_dist64 = d64;
            best_dist16 = d16;
        }
    }
    return {best_pat, best_dist64, best_dist16};
}

static inline uint8_t drift_index_from_offset(int dx, int dy) {
    const int sx = dx < 0 ? -1 : (dx > 0 ? 1 : 0);
    const int sy = dy < 0 ? -1 : (dy > 0 ? 1 : 0);
    return static_cast<uint8_t>((sy + 1) * 3 + (sx + 1));
}

static uint8_t calc_cooldown(uint8_t previous, uint8_t idx) {
    if (idx == 4) return 4;
    if ((idx & 1U) == 0U) return COOL_NONE;
    if ((previous ^ idx) == 6U) return COOL_NONE;
    return idx;
}

static std::vector<int> choose_search_drift_indices(uint8_t cooldown) {
    const int count = (cooldown == COOL_INIT) ? static_cast<int>(HASH_ORDER.size()) : HASH_FAST_N;
    std::vector<int> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        const int drift_idx = HASH_ORDER[i];
        out.push_back(drift_idx);
    }
    return out;
}

static bool should_prefer_bitgrid_candidate(const CandidateHit& primary,
                                            const CandidateHit& candidate,
                                            bool force_hint) {
    const int gain64 = primary.best_dist64 - candidate.best_dist64;
    if (candidate.best_dist64 == 0 && primary.best_dist64 >= 3) {
        return true;
    }
    if (force_hint) {
        if (gain64 >= BITGRID_ACCEPT_GAIN_HINT) {
            return true;
        }
        return candidate.best_dist64 == primary.best_dist64
            && candidate.best_dist16 + 1 < primary.best_dist16;
    }
    if (candidate.best_dist64 > 2) {
        return false;
    }
    if (gain64 >= BITGRID_ACCEPT_GAIN) {
        return true;
    }
    return primary.best_dist64 >= 10
        && gain64 > 0
        && candidate.best_dist16 <= primary.best_dist16;
}

static bool should_prefer_luma_candidate(const CandidateHit& primary,
                                         const CandidateHit& candidate,
                                         bool force_hint) {
    const int gain64 = primary.best_dist64 - candidate.best_dist64;
    if (candidate.best_dist64 == 0 && primary.best_dist64 >= 2) {
        return true;
    }
    if (candidate.best_dist64 > primary.best_dist64) {
        return false;
    }
    if (force_hint) {
        if (gain64 >= 1 && candidate.best_dist16 <= primary.best_dist16 + 1) {
            return true;
        }
        return gain64 == 0 && candidate.best_dist16 + 1 < primary.best_dist16;
    }
    if (gain64 >= 2) {
        return true;
    }
    if (primary.best_dist64 >= 8
        && gain64 >= 1
        && candidate.best_dist16 <= primary.best_dist16) {
        return true;
    }
    return primary.best_dist64 >= 10
        && gain64 == 0
        && candidate.best_dist16 + 1 < primary.best_dist16;
}

static std::pair<int, double> decode_color_from_mask(const cv::Mat& img,
                                                     int x0,
                                                     int y0,
                                                     int pat_idx,
                                                     const PatternDict& dict,
                                                     const ColorCalibration& calib) {
    const uint32_t best_mask_lo = dict.lo[pat_idx];
    const uint32_t best_mask_hi = dict.hi[pat_idx];
    std::array<uint16_t, 4> cnt_all {{0, 0, 0, 0}};
    std::array<uint16_t, 4> cnt_strong {{0, 0, 0, 0}};
    std::array<double, 4> dist_all {{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> dist_strong {{0.0, 0.0, 0.0, 0.0}};
    int valid_all = 0;
    int valid_strong = 0;
    for (int pr = 0; pr < TILE_SIZE; ++pr) {
        for (int pc = 0; pc < TILE_SIZE; ++pc) {
            const int bit = pr * TILE_SIZE + pc;
            if (!mask_is_on(best_mask_lo, best_mask_hi, bit)) continue;
            const cv::Vec3b px = img.at<cv::Vec3b>(y0 + pr, x0 + pc);
            const ColorMatch m = nearest_color(px, calib);
            const double gap = std::max(0.0, std::sqrt(m.second_dist) - std::sqrt(m.dist));
            if (m.span < COLOR_VOTE_MIN_SPAN && gap < COLOR_VOTE_MIN_GAP) continue;
            cnt_all[m.idx] = static_cast<uint16_t>(cnt_all[m.idx] + 1);
            dist_all[m.idx] += m.dist;
            ++valid_all;
            if (m.span >= COLOR_VOTE_STRONG_SPAN || gap >= COLOR_VOTE_STRONG_GAP) {
                cnt_strong[m.idx] = static_cast<uint16_t>(cnt_strong[m.idx] + 1);
                dist_strong[m.idx] += m.dist;
                ++valid_strong;
            }
        }
    }

    auto pick_best = [](const std::array<uint16_t, 4>& cnt_buf,
                        const std::array<double, 4>& dist_buf,
                        int valid) -> std::pair<int, double> {
        if (valid <= 0) return {0, std::numeric_limits<double>::infinity()};
        int best_color = 0;
        int best_cnt = -1;
        double best_avg = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            const int cnt = cnt_buf[i];
            if (cnt <= 0) continue;
            const double avg = dist_buf[i] / static_cast<double>(cnt);
            if (cnt > best_cnt || (cnt == best_cnt && avg < best_avg)) {
                best_color = i;
                best_cnt = cnt;
                best_avg = avg;
            }
        }
        return {best_color, best_avg};
    };

    if (valid_strong >= std::max(3, valid_all >> 2)) return pick_best(cnt_strong, dist_strong, valid_strong);
    return pick_best(cnt_all, dist_all, valid_all);
}

static DecodedCell decode_cell_adaptive(const cv::Mat& img,
                                        const PreprocessedFrames& frames,
                                        int x0,
                                        int y0,
                                        uint8_t cooldown,
                                        std::array<uint8_t, SAMPLE_AREA>& cell10,
                                        std::array<uint16_t, 16>& block16,
                                        const PatternDict& dict,
                                        const ColorCalibration& calib,
                                        const RecognizerOptions& options) {
    const std::vector<int> search_drift_indices = choose_search_drift_indices(cooldown);

    auto decode_candidate = [&](const std::vector<uint8_t>& signal_frame) {
        CandidateHit best;
        best.best_sample_x = clamp_int(x0, 0, IMG_W - TILE_SIZE);
        best.best_sample_y = clamp_int(y0, 0, IMG_H - TILE_SIZE);

        auto consider = [&](const CellSample10& sample, int drift_idx, const std::tuple<int, int, int>& hit) {
            const int ox = drift_idx % 3;
            const int oy = drift_idx / 3;
            const int sample_x = sample.sx + ox;
            const int sample_y = sample.sy + oy;
            const int dx = sample_x - x0;
            const int dy = sample_y - y0;
            const int pat = std::get<0>(hit);
            const int dist64 = std::get<1>(hit);
            const int dist16 = std::get<2>(hit);
            const int radius = std::abs(dx) + std::abs(dy);
            if (dist64 < best.best_dist64
                || (dist64 == best.best_dist64 && dist16 < best.best_dist16)
                || (dist64 == best.best_dist64 && dist16 == best.best_dist16 && radius < best.best_radius)) {
                best.best_pat = pat;
                best.best_dist64 = dist64;
                best.best_dist16 = dist16;
                best.best_dx = dx;
                best.best_dy = dy;
                best.best_sample_x = sample_x;
                best.best_sample_y = sample_y;
                best.best_radius = radius;
            }
        };

        auto evaluate_sample = [&](const CellSample10& sample, const std::vector<int>& drift_indices) -> bool {
            const double threshold = static_cast<double>(sample.sum) / static_cast<double>(SAMPLE_AREA);
            for (int drift_idx : drift_indices) {
                const HashSample10 hashes = hash_subwindow10(cell10, drift_idx, block16, threshold);
                const auto hit = match_pattern_combined(hashes.mask16, hashes.mask_lo, hashes.mask_hi, dict);
                consider(sample, drift_idx, hit);
                if (best.best_dist64 <= 2 && best.best_dist16 == 0 && drift_idx == 4) {
                    return true;
                }
            }
            return false;
        };

        CellSample10 sample = sample_cell10(signal_frame, x0, y0, cell10);
        evaluate_sample(sample, search_drift_indices);

        if (best.best_dist64 > 8 || best.best_dist16 > 1) {
            static const std::vector<int> center_only = {4};
            for (const auto& off : SEARCH_EXTENDED) {
                if (std::abs(off[0]) <= 1 && std::abs(off[1]) <= 1) continue;
                sample = sample_cell10(signal_frame, x0 + off[0], y0 + off[1], cell10);
                evaluate_sample(sample, center_only);
                if (best.best_dist64 <= 2 && best.best_dist16 == 0) {
                    break;
                }
            }
        }
        return best;
    };

    CandidateHit best = decode_candidate(*frames.primary_frame);

    if (options.enable_luma_recheck
        && frames.luma_frame
        && (frames.luma_hint
            || best.best_dist64 > LUMA_RECHECK_DIST64
            || (best.best_dist64 > 4 && best.best_dist16 > LUMA_RECHECK_DIST16))) {
        const CandidateHit luma = decode_candidate(*frames.luma_frame);
        if (should_prefer_luma_candidate(best, luma, frames.luma_hint)) {
            best = luma;
        }
    }

    if (options.enable_bitgrid_recheck
        && frames.bitgrid_frame
        && (frames.bitgrid_hint
            || best.best_dist64 > BITGRID_RECHECK_DIST64
            || (best.best_dist64 > 6 && best.best_dist16 > BITGRID_RECHECK_DIST16))) {
        const CandidateHit bitgrid = decode_candidate(*frames.bitgrid_frame);
        if (should_prefer_bitgrid_candidate(best, bitgrid, frames.bitgrid_hint)) {
            best = bitgrid;
        }
    }

    const auto color = decode_color_from_mask(img, best.best_sample_x, best.best_sample_y, best.best_pat, dict, calib);
    const int p_bits = pattern_bits_for_dict(static_cast<int>(dict.masks64.size()));
    const uint8_t symbol = static_cast<uint8_t>(((color.first << p_bits) | best.best_pat) & 0x3F);
    return {
        symbol,
        static_cast<uint16_t>((best.best_dist64 << 2) + best.best_dist16),
        drift_index_from_offset(best.best_dx, best.best_dy),
        static_cast<int8_t>(best.best_dx),
        static_cast<int8_t>(best.best_dy)
    };
}

static DecodeLayout build_decode_layout() {
    DecodeLayout layout;
    std::vector<int16_t> xs;
    std::vector<int16_t> ys;
    std::vector<int16_t> rows;
    std::vector<int16_t> cols;
    std::vector<uint8_t> kinds;
    std::vector<int32_t> rc_to_idx(GRID_ROWS * GRID_COLS, -1);
    for (int r = 0; r < GRID_ROWS; ++r) {
        for (int c = 0; c < GRID_COLS; ++c) {
            if (is_anchor_reserved(r, c)) continue;
            const int idx = static_cast<int>(xs.size());
            rows.push_back(static_cast<int16_t>(r));
            cols.push_back(static_cast<int16_t>(c));
            xs.push_back(static_cast<int16_t>(MARGIN + c * STRIDE));
            ys.push_back(static_cast<int16_t>(MARGIN + r * STRIDE));
            if (is_calibration_cell(r, c)) kinds.push_back(CELL_KIND_CAL);
            else if (is_header_cell(r, c)) kinds.push_back(CELL_KIND_HEADER);
            else kinds.push_back(CELL_KIND_PAYLOAD);
            rc_to_idx[r * GRID_COLS + c] = idx;
        }
    }

    const int n = static_cast<int>(xs.size());
    std::vector<int32_t> neighbors(n * 4, -1);
    for (int i = 0; i < n; ++i) {
        const int r = rows[i];
        const int c = cols[i];
        neighbors[i * 4] = (c + 1 < GRID_COLS) ? rc_to_idx[r * GRID_COLS + c + 1] : -1;
        neighbors[i * 4 + 1] = (c - 1 >= 0) ? rc_to_idx[r * GRID_COLS + c - 1] : -1;
        neighbors[i * 4 + 2] = (r + 1 < GRID_ROWS) ? rc_to_idx[(r + 1) * GRID_COLS + c] : -1;
        neighbors[i * 4 + 3] = (r - 1 >= 0) ? rc_to_idx[(r - 1) * GRID_COLS + c] : -1;
    }

    std::vector<uint32_t> seeds;
    std::vector<uint8_t> seen(n, 0);
    auto push_seed = [&](int r, int c, uint16_t prio) {
        const int idx = rc_to_idx[r * GRID_COLS + c];
        if (idx < 0 || seen[idx]) return;
        seen[idx] = 1;
        seeds.push_back(pack_heap_node(idx, prio));
    };
    push_seed(0, 6, 0);
    push_seed(0, Config::RIGHT_INNER_COL, 0);
    push_seed(Config::BOTTOM_INNER_ROW, 6, 0);
    push_seed(Config::BOTTOM_INNER_ROW, Config::RIGHT_INNER_COL, 0);
    push_seed(6, 0, 1);
    push_seed(6, GRID_COLS - 1, 1);
    push_seed(GRID_ROWS - 7, 0, 1);
    push_seed(GRID_ROWS - 7, GRID_COLS - 1, 1);

    layout.count = n;
    layout.x = std::move(xs);
    layout.y = std::move(ys);
    layout.row = std::move(rows);
    layout.col = std::move(cols);
    layout.kind = std::move(kinds);
    layout.neighbors = std::move(neighbors);
    layout.seeds = std::move(seeds);
    return layout;
}

static const DecodeLayout& get_decode_layout() {
    static const DecodeLayout layout = build_decode_layout();
    return layout;
}
static void init_decode_buffers(DecodeBuffers& buffers, const DecodeLayout& layout) {
    buffers.pending.assign(layout.count, 0);
    buffers.drift_x.assign(layout.count, 0);
    buffers.drift_y.assign(layout.count, 0);
    buffers.priority.assign(layout.count, 0);
    buffers.cooldown.assign(layout.count, 0);
    buffers.symbol.assign(layout.count, 0);
    buffers.gray_frame.assign(IMG_W * IMG_H, 0);
    buffers.gray_temp.assign(IMG_W * IMG_H, 0);
    buffers.luma_frame.assign(IMG_W * IMG_H, 0);
    buffers.bin_frame.assign(IMG_W * IMG_H, 0);
    buffers.sat_frame.assign((IMG_W + 1) * (IMG_H + 1), 0);
}

static void try_queue_neighbor(int next,
                               int8_t drift_x,
                               int8_t drift_y,
                               uint16_t prio,
                               uint8_t cooldown,
                               DecodeBuffers& buffers,
                               std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>& heap) {
    if (next < 0 || buffers.pending[next] == 0) return;
    if (buffers.priority[next] <= prio) return;
    buffers.drift_x[next] = drift_x;
    buffers.drift_y[next] = drift_y;
    buffers.priority[next] = prio;
    buffers.cooldown[next] = cooldown;
    heap.push(pack_heap_node(next, prio));
}

static void queue_adjacents(int idx,
                            int8_t drift_x,
                            int8_t drift_y,
                            uint16_t prio,
                            uint8_t cooldown,
                            const DecodeLayout& layout,
                            DecodeBuffers& buffers,
                            std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>& heap) {
    const int b = idx * 4;
    try_queue_neighbor(layout.neighbors[b], drift_x, drift_y, prio, cooldown, buffers, heap);
    try_queue_neighbor(layout.neighbors[b + 1], drift_x, drift_y, prio, cooldown, buffers, heap);
    try_queue_neighbor(layout.neighbors[b + 2], drift_x, drift_y, prio, cooldown, buffers, heap);
    try_queue_neighbor(layout.neighbors[b + 3], drift_x, drift_y, prio, cooldown, buffers, heap);
}

static void queue_aggressive(int idx,
                             int8_t drift_x,
                             int8_t drift_y,
                             uint16_t prio,
                             uint8_t cooldown,
                             const DecodeLayout& layout,
                             DecodeBuffers& buffers,
                             std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>& heap) {
    const int b = idx * 4;
    const int right = layout.neighbors[b];
    const int left = layout.neighbors[b + 1];
    const int down = layout.neighbors[b + 2];
    const int up = layout.neighbors[b + 3];

    if (right >= 0 && left >= 0) {
        const int rr = layout.neighbors[right * 4];
        const int rrr = rr >= 0 ? layout.neighbors[rr * 4] : -1;
        const int ll = layout.neighbors[left * 4 + 1];
        const int lll = ll >= 0 ? layout.neighbors[ll * 4 + 1] : -1;
        try_queue_neighbor(rr, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(rrr, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(ll, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(lll, drift_x, drift_y, prio, cooldown, buffers, heap);
    }

    if (up >= 0 && down >= 0) {
        const int uu = layout.neighbors[up * 4 + 3];
        const int uuu = uu >= 0 ? layout.neighbors[uu * 4 + 3] : -1;
        const int dd = layout.neighbors[down * 4 + 2];
        const int ddd = dd >= 0 ? layout.neighbors[dd * 4 + 2] : -1;
        try_queue_neighbor(uu, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(uuu, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(dd, drift_x, drift_y, prio, cooldown, buffers, heap);
        try_queue_neighbor(ddd, drift_x, drift_y, prio, cooldown, buffers, heap);
    }
}

static void decode_by_priority(const cv::Mat& img,
                               const PatternDict& dict,
                               const ColorCalibration& calib,
                               const RecognizerOptions& options,
                               DecodeBuffers& buffers) {
    const DecodeLayout& layout = get_decode_layout();
    const PreprocessedFrames frames = preprocess_symbol_frame(
        img,
        calib,
        buffers,
        options.sharpen_hint,
        options.sharpen_strength
    );
    std::fill(buffers.pending.begin(), buffers.pending.end(), 1);
    std::fill(buffers.drift_x.begin(), buffers.drift_x.end(), 0);
    std::fill(buffers.drift_y.begin(), buffers.drift_y.end(), 0);
    std::fill(buffers.priority.begin(), buffers.priority.end(), PRIO_INIT);
    std::fill(buffers.cooldown.begin(), buffers.cooldown.end(), COOL_INIT);
    std::fill(buffers.symbol.begin(), buffers.symbol.end(), 0);

    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> heap;
    for (uint32_t seed : layout.seeds) heap.push(seed);

    int decoded = 0;
    while (!heap.empty() && decoded < layout.count) {
        const uint32_t node = heap.top();
        heap.pop();
        const int idx = unpack_heap_idx(node);
        if (buffers.pending[idx] == 0) continue;
        buffers.pending[idx] = 0;
        ++decoded;

        const uint16_t prev_err = buffers.priority[idx];
        const uint8_t prev_cooldown = buffers.cooldown[idx];
        const DecodedCell cell = decode_cell_adaptive(
            img,
            frames,
            layout.x[idx] + buffers.drift_x[idx],
            layout.y[idx] + buffers.drift_y[idx],
            prev_cooldown,
            buffers.cell10,
            buffers.block16,
            dict,
            calib,
            options
        );

        const int8_t ndx = static_cast<int8_t>(clamp_int(static_cast<int>(buffers.drift_x[idx]) + cell.drift_x, -DRIFT_MAX, DRIFT_MAX));
        const int8_t ndy = static_cast<int8_t>(clamp_int(static_cast<int>(buffers.drift_y[idx]) + cell.drift_y, -DRIFT_MAX, DRIFT_MAX));
        const uint8_t next_cooldown = calc_cooldown(prev_cooldown, cell.drift_idx);
        queue_adjacents(idx, ndx, ndy, cell.best_dist, next_cooldown, layout, buffers, heap);
        if (prev_err < 3 && cell.best_dist < 3 && prev_cooldown == 4 && next_cooldown == 4) {
            queue_aggressive(idx, ndx, ndy, cell.best_dist, next_cooldown, layout, buffers, heap);
        }

        buffers.drift_x[idx] = ndx;
        buffers.drift_y[idx] = ndy;
        buffers.priority[idx] = cell.best_dist;
        buffers.cooldown[idx] = next_cooldown;
        buffers.symbol[idx] = cell.symbol;
    }

    if (decoded < layout.count) {
        for (int idx = 0; idx < layout.count; ++idx) {
            if (buffers.pending[idx] == 0) continue;
            const DecodedCell cell = decode_cell_adaptive(
                img,
                frames,
                layout.x[idx],
                layout.y[idx],
                COOL_INIT,
                buffers.cell10,
                buffers.block16,
                dict,
                calib,
                options
            );
            buffers.pending[idx] = 0;
            buffers.symbol[idx] = cell.symbol;
            buffers.priority[idx] = cell.best_dist;
        }
    }
}

static double measure_blur_score(const cv::Mat& img, double margin_ratio = 0.08, int sample_n = 48) {
    if (img.empty()) {
        return 0.0;
    }
    const int src_w = img.cols;
    const int src_h = img.rows;
    const int margin_x = std::max(0, static_cast<int>(std::lround(src_w * margin_ratio)));
    const int margin_y = std::max(0, static_cast<int>(std::lround(src_h * margin_ratio)));
    const int sample_x = std::min(std::max(0, margin_x), src_w - 1);
    const int sample_y = std::min(std::max(0, margin_y), src_h - 1);
    const int sample_w = std::max(1, src_w - 2 * margin_x);
    const int sample_h = std::max(1, src_h - 2 * margin_y);
    const int right = std::min(src_w, sample_x + sample_w);
    const int bottom = std::min(src_h, sample_y + sample_h);
    cv::Rect roi(sample_x, sample_y, std::max(1, right - sample_x), std::max(1, bottom - sample_y));
    cv::Mat patch = img(roi);
    cv::Mat sample;
    cv::resize(patch, sample, cv::Size(sample_n, sample_n), 0.0, 0.0, cv::INTER_LINEAR);
    double sum = 0.0;
    int count = 0;
    for (int y = 1; y < sample.rows - 1; ++y) {
        for (int x = 1; x < sample.cols - 1; ++x) {
            const cv::Vec3b& l = sample.at<cv::Vec3b>(y, x - 1);
            const cv::Vec3b& r = sample.at<cv::Vec3b>(y, x + 1);
            const cv::Vec3b& u = sample.at<cv::Vec3b>(y - 1, x);
            const cv::Vec3b& d = sample.at<cv::Vec3b>(y + 1, x);
            const int gray_l = (l[2] * 77 + l[1] * 150 + l[0] * 29) >> 8;
            const int gray_r = (r[2] * 77 + r[1] * 150 + r[0] * 29) >> 8;
            const int gray_u = (u[2] * 77 + u[1] * 150 + u[0] * 29) >> 8;
            const int gray_d = (d[2] * 77 + d[1] * 150 + d[0] * 29) >> 8;
            sum += std::abs(gray_r - gray_l);
            sum += std::abs(gray_d - gray_u);
            count += 2;
        }
    }
    return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
}

struct PackedBits {
    std::vector<uint8_t> bytes;
    int tail_bits = 0;
};

PackedBits pack6_bits(const std::vector<uint8_t>& symbols) {
    PackedBits packed;
    packed.bytes.resize((symbols.size() * 6 + 7) / 8);
    size_t write_idx = 0;
    uint32_t buffer = 0;
    int bits = 0;
    for (uint8_t symbol : symbols) {
        buffer = (buffer << 6) | (symbol & 0x3FU);
        bits += 6;
        while (bits >= 8) {
            packed.bytes[write_idx++] = static_cast<uint8_t>((buffer >> (bits - 8)) & 0xFFU);
            bits -= 8;
        }
    }
    if (bits > 0) {
        packed.bytes[write_idx++] = static_cast<uint8_t>((buffer << (8 - bits)) & 0xFFU);
    }
    packed.bytes.resize(write_idx);
    packed.tail_bits = bits;
    return packed;
}

cv::Mat normalize_input(const cv::Mat& input) {
    cv::Mat bgr;
    if (input.channels() == 3) {
        bgr = input;
    } else if (input.channels() == 4) {
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
    } else if (input.channels() == 1) {
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    } else {
        throw std::runtime_error("unsupported deskewed image format");
    }
    if (bgr.cols == IMG_W && bgr.rows == IMG_H) {
        return bgr;
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(IMG_W, IMG_H), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

}  // namespace

PatternRecognizer::PatternRecognizer(PatternDictionary dict, RecognizerOptions options)
    : dict_(std::move(dict)), options_(options) {}

RecognizeResult PatternRecognizer::Decode(const cv::Mat& deskewed) const {
    RecognizeResult result;
    if (dict_.empty() || deskewed.empty()) {
        return result;
    }

    const cv::Mat img = normalize_input(deskewed);
    const PatternDict dict = build_pattern_dict(dict_.masks64);
    const DecodeLayout& layout = get_decode_layout();
    DecodeBuffers buffers;
    init_decode_buffers(buffers, layout);
    const ColorCalibration calib = estimate_color_calibration(img);
    RecognizerOptions runtime_options = options_;
    const double blur_score = measure_blur_score(img);
    if (blur_score < 13.0) {
        runtime_options.sharpen_hint = true;
        if (!(runtime_options.sharpen_strength > 0.0)) {
            runtime_options.sharpen_strength = 0.6;
        }
    }
    decode_by_priority(img, dict, calib, runtime_options, buffers);

    double sum_pattern_dist = 0.0;
    int pattern_count = 0;
    for (int i = 0; i < layout.count; ++i) {
        if (layout.kind[i] == CELL_KIND_HEADER) {
            result.header_symbols.push_back(buffers.symbol[i]);
        } else if (layout.kind[i] == CELL_KIND_PAYLOAD) {
            result.payload_symbols.push_back(buffers.symbol[i]);
        }
        if (layout.kind[i] != CELL_KIND_CAL) {
            sum_pattern_dist += static_cast<double>(buffers.priority[i]);
            ++pattern_count;
        }
    }

    const PackedBits header = pack6_bits(result.header_symbols);
    const PackedBits payload = pack6_bits(result.payload_symbols);
    result.header_bytes = header.bytes;
    result.payload_bytes = payload.bytes;
    result.header_tail_bits = header.tail_bits;
    result.payload_tail_bits = payload.tail_bits;
    result.avg_pattern_dist = pattern_count > 0 ? (sum_pattern_dist / static_cast<double>(pattern_count)) : 0.0;
    result.pattern_bits = dict_.pattern_bits();
    result.ok = true;
    return result;
}

}  // namespace camdrop::vision
