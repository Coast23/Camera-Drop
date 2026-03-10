#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef _MSC_VER
#  include <intrin.h>
static inline int popcount32(uint32_t x) { return static_cast<int>(__popcnt(x)); }
static inline int popcount64(uint64_t x) { return static_cast<int>(__popcnt64(x)); }
#else
static inline int popcount32(uint32_t x) { return __builtin_popcount(x); }
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
#endif

static constexpr int GRID_SIZE = 112;
static constexpr int STRIDE = 9;
static constexpr int MARGIN = 8;
static constexpr int IMG_SIZE = 1024;
static constexpr int TILE_SIZE = 8;
static constexpr int CELL_SAMPLE_SIZE = TILE_SIZE + 2;
static constexpr int SAMPLE_AREA = CELL_SAMPLE_SIZE * CELL_SAMPLE_SIZE;
static constexpr int NUM_COLORS = 4;
static constexpr int ANCHOR_OUT_START = 2;
static constexpr int ANCHOR_L1_SIZE = 56;
static constexpr int ANCHOR_L2_INSET = 7;
static constexpr int ANCHOR_L2_SIZE = 42;
static constexpr int ANCHOR_L3_INSET = 14;
static constexpr int ANCHOR_L3_SIZE = 28;
static constexpr int ANCHOR_L4_INSET = 21;
static constexpr int ANCHOR_L4_SIZE = 14;
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
    std::array<uint8_t, SAMPLE_AREA> cell10 {};
    std::array<uint16_t, 16> block16 {};
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

static inline int pattern_bits_for_dict(int n) {
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    return bits;
}

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint32_t pack_heap_node(int idx, uint16_t prio) {
    return (static_cast<uint32_t>(prio) << HEAP_IDX_BITS) | static_cast<uint32_t>(idx & HEAP_IDX_MASK);
}

static inline int unpack_heap_idx(uint32_t node) {
    return static_cast<int>(node & HEAP_IDX_MASK);
}

static inline bool is_anchor_reserved(int r, int c) {
    if (r < 6 && c < 6) return true;
    if (r < 6 && c > 105) return true;
    if (r > 105 && c < 6) return true;
    if (r > 105 && c > 105) return true;
    return false;
}

static inline bool is_calibration_cell(int r, int c) {
    return r == 0 && c >= 6 && c < 14;
}

static inline bool is_header_cell(int r, int c) {
    return r == 0 && c >= 14 && c < 46;
}

static inline bool is_payload_cell(int r, int c) {
    return !is_anchor_reserved(r, c) && !is_calibration_cell(r, c) && !is_header_cell(r, c);
}

static inline bool mask_is_on(uint32_t mask_lo, uint32_t mask_hi, int bit) {
    if (bit < 32) return ((mask_lo >> bit) & 1U) != 0;
    return ((mask_hi >> (bit - 32)) & 1U) != 0;
}

static inline std::pair<uint32_t, uint32_t> split_mask64(uint64_t mask) {
    return {
        static_cast<uint32_t>(mask & 0xFFFFFFFFULL),
        static_cast<uint32_t>((mask >> 32) & 0xFFFFFFFFULL)
    };
}

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

static inline uint16_t compress_mask64_to_16(uint64_t mask) {
    const auto parts = split_mask64(mask);
    return compress_mask64_to_16(parts.first, parts.second);
}

static inline bool mask64_is_on(uint64_t mask, int x, int y) {
    return ((mask >> (y * 8 + x)) & 1ULL) != 0ULL;
}

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

static Rgb sample_rect_mean_rgb(const cv::Mat& img, int x0, int y0, int size) {
    const int sx = clamp_int(x0, 0, IMG_SIZE - size);
    const int sy = clamp_int(y0, 0, IMG_SIZE - size);
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

template <typename ScoreFn>
static Rgb sample_rect_selective_rgb(const cv::Mat& img,
                                     int x0,
                                     int y0,
                                     int size,
                                     ScoreFn score_pixel,
                                     double keep_ratio) {
    const int sx = clamp_int(x0, 0, IMG_SIZE - size);
    const int sy = clamp_int(y0, 0, IMG_SIZE - size);
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

static Rgb average_rgb(const Rgb& a, const Rgb& b) {
    return {(a.r + b.r) * 0.5, (a.g + b.g) * 0.5, (a.b + b.b) * 0.5};
}

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

static Rgb subtract_rgb(const Rgb& a, const Rgb& b) {
    return {
        std::max(0.0, a.r - b.r),
        std::max(0.0, a.g - b.g),
        std::max(0.0, a.b - b.b),
    };
}

static double color_rect_score(double r, double g, double b, int color_idx) {
    switch (color_idx) {
        case 0: return (r + g) - (b * 2.0);
        case 1: return (g * 2.0) - (r + b);
        case 2: return (g + b) - (r * 2.0);
        case 3: return (r + b) - (g * 2.0);
        default: return 0.0;
    }
}

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
    const int base_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int base_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int outer_inset = 2;
    const int outer_size = 8;
    const int inner_base_x = base_x + ANCHOR_L3_INSET;
    const int inner_base_y = base_y + ANCHOR_L3_INSET;
    const int inner_half = ANCHOR_L3_SIZE >> 1;
    const int inner_inset = 1;
    const int inner_size = 6;
    const int tl_base_x = ANCHOR_OUT_START;
    const int tl_base_y = ANCHOR_OUT_START;
    const int tr_base_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int tr_base_y = ANCHOR_OUT_START;
    const int bl_base_x = ANCHOR_OUT_START;
    const int bl_base_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;

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

static void build_gray_frame(const cv::Mat& img, const ColorCalibration& calib, std::vector<uint8_t>& gray_frame) {
    gray_frame.resize(IMG_SIZE * IMG_SIZE);
    for (int y = 0; y < IMG_SIZE; ++y) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < IMG_SIZE; ++x) {
            const cv::Vec3b& px = row[x];
            const Rgb rgb = apply_color_transform(
                static_cast<double>(px[2]),
                static_cast<double>(px[1]),
                static_cast<double>(px[0]),
                calib
            );
            gray_frame[y * IMG_SIZE + x] = compute_color_signal(rgb.r, rgb.g, rgb.b);
        }
    }
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
    const int sx = clamp_int(x0 - 1, 0, IMG_SIZE - CELL_SAMPLE_SIZE);
    const int sy = clamp_int(y0 - 1, 0, IMG_SIZE - CELL_SAMPLE_SIZE);
    int sum = 0;
    int k = 0;
    for (int r = 0; r < CELL_SAMPLE_SIZE; ++r) {
        const int row = (sy + r) * IMG_SIZE + sx;
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
                                        const std::vector<uint8_t>& signal_frame,
                                        int x0,
                                        int y0,
                                        uint8_t cooldown,
                                        std::array<uint8_t, SAMPLE_AREA>& cell10,
                                        std::array<uint16_t, 16>& block16,
                                        const PatternDict& dict,
                                        const ColorCalibration& calib) {
    const std::vector<int> search_drift_indices = choose_search_drift_indices(cooldown);
    int best_pat = 0;
    int best_dist16 = 17;
    int best_dist64 = 65;
    int best_dx = 0;
    int best_dy = 0;
    int best_sample_x = clamp_int(x0, 0, IMG_SIZE - TILE_SIZE);
    int best_sample_y = clamp_int(y0, 0, IMG_SIZE - TILE_SIZE);
    int best_radius = 0;

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
        if (dist64 < best_dist64
            || (dist64 == best_dist64 && dist16 < best_dist16)
            || (dist64 == best_dist64 && dist16 == best_dist16 && radius < best_radius)) {
            best_pat = pat;
            best_dist64 = dist64;
            best_dist16 = dist16;
            best_dx = dx;
            best_dy = dy;
            best_sample_x = sample_x;
            best_sample_y = sample_y;
            best_radius = radius;
        }
    };

    auto evaluate_sample = [&](const CellSample10& sample, const std::vector<int>& drift_indices) {
        const double threshold = static_cast<double>(sample.sum) / static_cast<double>(SAMPLE_AREA);
        for (int drift_idx : drift_indices) {
            const HashSample10 hashes = hash_subwindow10(cell10, drift_idx, block16, threshold);
            const auto hit = match_pattern_combined(hashes.mask16, hashes.mask_lo, hashes.mask_hi, dict);
            consider(sample, drift_idx, hit);
        }
    };

    CellSample10 sample = sample_cell10(signal_frame, x0, y0, cell10);
    evaluate_sample(sample, search_drift_indices);

    if (best_dist64 > 8 || best_dist16 > 1) {
        const std::vector<int> center_only = {4};
        for (const auto& off : SEARCH_EXTENDED) {
            if (std::abs(off[0]) <= 1 && std::abs(off[1]) <= 1) continue;
            sample = sample_cell10(signal_frame, x0 + off[0], y0 + off[1], cell10);
            evaluate_sample(sample, center_only);
            if (best_dist64 <= 2 && best_dist16 == 0) break;
        }
    }

    const auto color = decode_color_from_mask(img, best_sample_x, best_sample_y, best_pat, dict, calib);
    const int p_bits = pattern_bits_for_dict(static_cast<int>(dict.masks64.size()));
    const uint8_t symbol = static_cast<uint8_t>(((color.first << p_bits) | best_pat) & 0x3F);
    return {
        symbol,
        static_cast<uint16_t>((best_dist64 << 2) + best_dist16),
        drift_index_from_offset(best_dx, best_dy),
        static_cast<int8_t>(best_dx),
        static_cast<int8_t>(best_dy)
    };
}

static DecodeLayout build_decode_layout() {
    DecodeLayout layout;
    std::vector<int16_t> xs;
    std::vector<int16_t> ys;
    std::vector<int16_t> rows;
    std::vector<int16_t> cols;
    std::vector<uint8_t> kinds;
    std::vector<int32_t> rc_to_idx(GRID_SIZE * GRID_SIZE, -1);
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (is_anchor_reserved(r, c)) continue;
            const int idx = static_cast<int>(xs.size());
            rows.push_back(static_cast<int16_t>(r));
            cols.push_back(static_cast<int16_t>(c));
            xs.push_back(static_cast<int16_t>(MARGIN + c * STRIDE));
            ys.push_back(static_cast<int16_t>(MARGIN + r * STRIDE));
            if (is_calibration_cell(r, c)) kinds.push_back(CELL_KIND_CAL);
            else if (is_header_cell(r, c)) kinds.push_back(CELL_KIND_HEADER);
            else kinds.push_back(CELL_KIND_PAYLOAD);
            rc_to_idx[r * GRID_SIZE + c] = idx;
        }
    }

    const int n = static_cast<int>(xs.size());
    std::vector<int32_t> neighbors(n * 4, -1);
    for (int i = 0; i < n; ++i) {
        const int r = rows[i];
        const int c = cols[i];
        neighbors[i * 4] = (c + 1 < GRID_SIZE) ? rc_to_idx[r * GRID_SIZE + c + 1] : -1;
        neighbors[i * 4 + 1] = (c - 1 >= 0) ? rc_to_idx[r * GRID_SIZE + c - 1] : -1;
        neighbors[i * 4 + 2] = (r + 1 < GRID_SIZE) ? rc_to_idx[(r + 1) * GRID_SIZE + c] : -1;
        neighbors[i * 4 + 3] = (r - 1 >= 0) ? rc_to_idx[(r - 1) * GRID_SIZE + c] : -1;
    }

    std::vector<uint32_t> seeds;
    std::vector<uint8_t> seen(n, 0);
    auto push_seed = [&](int r, int c, uint16_t prio) {
        const int idx = rc_to_idx[r * GRID_SIZE + c];
        if (idx < 0 || seen[idx]) return;
        seen[idx] = 1;
        seeds.push_back(pack_heap_node(idx, prio));
    };
    push_seed(0, 6, 0);
    push_seed(0, 105, 0);
    push_seed(111, 6, 0);
    push_seed(111, 105, 0);
    push_seed(6, 0, 1);
    push_seed(6, 111, 1);
    push_seed(105, 0, 1);
    push_seed(105, 111, 1);

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
    buffers.gray_frame.assign(IMG_SIZE * IMG_SIZE, 0);
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
                               DecodeBuffers& buffers) {
    const DecodeLayout& layout = get_decode_layout();
    build_gray_frame(img, calib, buffers.gray_frame);
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
            buffers.gray_frame,
            layout.x[idx] + buffers.drift_x[idx],
            layout.y[idx] + buffers.drift_y[idx],
            prev_cooldown,
            buffers.cell10,
            buffers.block16,
            dict,
            calib
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
                buffers.gray_frame,
                layout.x[idx],
                layout.y[idx],
                COOL_INIT,
                buffers.cell10,
                buffers.block16,
                dict,
                calib
            );
            buffers.pending[idx] = 0;
            buffers.symbol[idx] = cell.symbol;
            buffers.priority[idx] = cell.best_dist;
        }
    }
}

static DecodeStats decode_frame(const cv::Mat& img,
                                const PatternDict& dict,
                                const std::vector<uint8_t>& raw) {
    const DecodeLayout& layout = get_decode_layout();
    DecodeBuffers buffers;
    init_decode_buffers(buffers, layout);
    const ColorCalibration calib = estimate_color_calibration(img);
    decode_by_priority(img, dict, calib, buffers);

    const int n = static_cast<int>(dict.masks64.size());
    const int p_bits = pattern_bits_for_dict(n);
    DecodeStats stats;
    for (int i = 0; i < layout.count; ++i) {
        if (layout.kind[i] != CELL_KIND_PAYLOAD) continue;
        const int r = layout.row[i];
        const int c = layout.col[i];
        const uint8_t dec_symbol = buffers.symbol[i];
        const uint8_t exp_symbol = raw[r * GRID_SIZE + c];
        const int dec_pat = dec_symbol & (n - 1);
        const int dec_col = dec_symbol >> p_bits;
        const int exp_pat = exp_symbol & (n - 1);
        const int exp_col = exp_symbol >> p_bits;
        ++stats.total;
        if (dec_pat == exp_pat) ++stats.pattern_correct;
        if (dec_col == exp_col) ++stats.color_correct;
        if (dec_symbol == exp_symbol) ++stats.symbol_correct;
    }
    return stats;
}

static void draw_normal_anchor(cv::Mat& img, int x0, int y0) {
    cv::rectangle(img, cv::Rect(x0, y0, ANCHOR_L1_SIZE, ANCHOR_L1_SIZE), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET, ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + ANCHOR_L3_INSET, y0 + ANCHOR_L3_INSET, ANCHOR_L3_SIZE, ANCHOR_L3_SIZE), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET, ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
}

static void draw_br_anchor(cv::Mat& img, int x0, int y0) {
    const int h1 = ANCHOR_L1_SIZE / 2;
    cv::rectangle(img, cv::Rect(x0, y0, h1, h1), cv::Scalar(0, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + h1, y0, h1, h1), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, y0 + h1, h1, h1), cv::Scalar(255, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + h1, y0 + h1, h1, h1), cv::Scalar(255, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET, ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
    const int h3 = ANCHOR_L3_SIZE / 2;
    const int ix = x0 + ANCHOR_L3_INSET;
    const int iy = y0 + ANCHOR_L3_INSET;
    cv::rectangle(img, cv::Rect(ix, iy, h3, h3), cv::Scalar(0, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix + h3, iy, h3, h3), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix, iy + h3, h3, h3), cv::Scalar(255, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(ix + h3, iy + h3, h3, h3), cv::Scalar(255, 255, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET, ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), cv::Scalar(0, 0, 0), cv::FILLED);
}

static void draw_anchors(cv::Mat& img) {
    const int tl_x = ANCHOR_OUT_START;
    const int tl_y = ANCHOR_OUT_START;
    const int tr_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int tr_y = ANCHOR_OUT_START;
    const int bl_x = ANCHOR_OUT_START;
    const int bl_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int br_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const int br_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    draw_normal_anchor(img, tl_x, tl_y);
    draw_normal_anchor(img, tr_x, tr_y);
    draw_normal_anchor(img, bl_x, bl_y);
    draw_br_anchor(img, br_x, br_y);
}

static EncodedFrame encode_frame(const PatternDict& dict, unsigned rng_seed) {
    EncodedFrame res;
    res.img = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
    res.raw.assign(GRID_SIZE * GRID_SIZE, 0);

    draw_anchors(res.img);
    for (int i = 0; i < 8; ++i) {
        const int start_x = MARGIN + (6 + i) * STRIDE;
        const int start_y = MARGIN;
        const cv::Vec3b& color = COLORS_BGR[i % NUM_COLORS];
        cv::rectangle(res.img, cv::Rect(start_x, start_y, TILE_SIZE, TILE_SIZE), cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
    }
    for (int i = 14; i < 46; ++i) {
        cv::rectangle(res.img, cv::Rect(MARGIN + i * STRIDE, MARGIN, TILE_SIZE, TILE_SIZE), cv::Scalar(128, 128, 128), cv::FILLED);
    }

    const int n = static_cast<int>(dict.masks64.size());
    const int p_bits = pattern_bits_for_dict(n);
    cv::RNG rng(rng_seed);
    const int total_vals = n * NUM_COLORS;
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (!is_payload_cell(r, c)) continue;
            const uint8_t data = static_cast<uint8_t>(rng.uniform(0, total_vals));
            res.raw[r * GRID_SIZE + c] = data;
            const int pat_idx = data & (n - 1);
            const int color_idx = data >> p_bits;
            const int sx = MARGIN + c * STRIDE;
            const int sy = MARGIN + r * STRIDE;
            const uint64_t mask = dict.masks64[pat_idx];
            for (int pr = 0; pr < TILE_SIZE; ++pr) {
                for (int pc = 0; pc < TILE_SIZE; ++pc) {
                    if ((mask >> (pr * TILE_SIZE + pc)) & 1ULL) {
                        res.img.at<cv::Vec3b>(sy + pr, sx + pc) = COLORS_BGR[color_idx];
                    }
                }
            }
        }
    }
    return res;
}

static void stimulate_moire(cv::Mat& img, cv::RNG& rng, double scale) {
    const float amp = static_cast<float>(0.02 + 0.05 * (1.0 - scale) + rng.uniform(0.0, 0.02));
    const float fx = static_cast<float>(0.18 + rng.uniform(0.0, 0.22));
    const float fy = static_cast<float>(0.16 + rng.uniform(0.0, 0.24));
    const float phase = static_cast<float>(rng.uniform(0.0, 6.283185307179586));
    for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; ++c) {
            const float wave = 0.55f * std::sinf(r * fy + c * fx + phase)
                             + 0.45f * std::sinf(r * (fy * 0.37f) - c * (fx * 0.29f) + phase * 0.7f);
            const float m = 1.0f + amp * wave;
            cv::Vec3b& px = img.at<cv::Vec3b>(r, c);
            px[0] = cv::saturate_cast<uchar>(px[0] * m);
            px[1] = cv::saturate_cast<uchar>(px[1] * m);
            px[2] = cv::saturate_cast<uchar>(px[2] * m);
        }
    }
}

static void apply_motion_blur(cv::Mat& img, int kernel_size, int mode) {
    kernel_size = std::max(3, kernel_size | 1);
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);
    const int mid = kernel_size >> 1;
    if (mode == 0) {
        for (int x = 0; x < kernel_size; ++x) kernel.at<float>(mid, x) = 1.0f;
    } else if (mode == 1) {
        for (int y = 0; y < kernel_size; ++y) kernel.at<float>(y, mid) = 1.0f;
    } else if (mode == 2) {
        for (int i = 0; i < kernel_size; ++i) kernel.at<float>(i, i) = 1.0f;
    } else {
        for (int i = 0; i < kernel_size; ++i) kernel.at<float>(i, kernel_size - 1 - i) = 1.0f;
    }
    kernel /= cv::sum(kernel)[0];
    cv::filter2D(img, img, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
}

static void stimulate_blur(cv::Mat& img, cv::RNG& rng, double scale) {
    const double sigma = 0.25 + (1.0 - scale) * 0.65 + rng.uniform(0.0, 0.35);
    const int gauss_k = sigma >= 1.15 ? 5 : 3;
    cv::GaussianBlur(img, img, cv::Size(gauss_k, gauss_k), sigma, sigma, cv::BORDER_REPLICATE);
    if (rng.uniform(0.0, 1.0) < 0.45) {
        int kernel = scale < 0.62 ? 5 : 3;
        if (rng.uniform(0.0, 1.0) < 0.12) kernel += 2;
        apply_motion_blur(img, kernel, rng.uniform(0, 4));
    }
    if (rng.uniform(0.0, 1.0) < 0.20) {
        cv::Mat ghost;
        const double gx = rng.uniform(-1.2, 1.2);
        const double gy = rng.uniform(-1.2, 1.2);
        const cv::Matx23d mat(1.0, 0.0, gx, 0.0, 1.0, gy);
        cv::warpAffine(img, ghost, mat, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::addWeighted(img, 0.93, ghost, 0.07, 0.0, img);
    }
}

static void stimulate_color_cast(cv::Mat& img, cv::RNG& rng, double scale) {
    const bool harsh_bright = rng.uniform(0.0, 1.0) < 0.18;
    const bool harsh_dark = !harsh_bright && rng.uniform(0.0, 1.0) < 0.18;
    const double exposure = harsh_bright
        ? rng.uniform(1.08, 1.32)
        : (harsh_dark ? rng.uniform(0.72, 0.92) : rng.uniform(0.92, 1.10));
    const double gamma = harsh_bright
        ? rng.uniform(0.88, 1.02)
        : (harsh_dark ? rng.uniform(1.00, 1.22) : rng.uniform(0.95, 1.10));
    const double contrast = harsh_bright
        ? rng.uniform(0.88, 1.02)
        : (harsh_dark ? rng.uniform(0.96, 1.10) : rng.uniform(0.94, 1.08));
    const double lift = harsh_dark
        ? rng.uniform(3.0, 14.0 + (1.0 - scale) * 6.0)
        : rng.uniform(0.0, 6.0 + (1.0 - scale) * 4.0);
    const double white_clip = harsh_bright ? rng.uniform(0.92, 0.98) : 1.0;
    const std::array<double, 3> gains {{
        rng.uniform(0.90, 1.10),
        rng.uniform(0.84, 1.18),
        rng.uniform(0.84, 1.18),
    }};
    for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; ++c) {
            cv::Vec3b& px = img.at<cv::Vec3b>(r, c);
            for (int ch = 0; ch < 3; ++ch) {
                double v = static_cast<double>(px[ch]) / 255.0;
                v = std::pow(std::clamp(v, 0.0, 1.0), gamma);
                v = (v - 0.5) * contrast + 0.5;
                v = v * exposure * gains[ch] + lift / 255.0;
                if (white_clip < 1.0 && v > white_clip) {
                    v = white_clip + (v - white_clip) * 0.08;
                }
                px[ch] = cv::saturate_cast<uchar>(std::clamp(v, 0.0, 1.0) * 255.0);
            }
        }
    }
}

static void stimulate_noise(cv::Mat& img, cv::RNG& rng, double scale) {
    const double sigma = 2.0 + (1.0 - scale) * 4.0 + rng.uniform(0.0, 3.5);
    cv::Mat img16;
    img.convertTo(img16, CV_16SC3);
    cv::Mat noise(img.size(), CV_16SC3);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar(0, 0, 0), cv::Scalar(sigma, sigma, sigma));
    cv::add(img16, noise, img16, cv::noArray(), CV_16SC3);
    img16.convertTo(img, CV_8UC3);
    if (rng.uniform(0.0, 1.0) < 0.08) {
        const int sprinkles = 48 + static_cast<int>((1.0 - scale) * 96.0);
        for (int i = 0; i < sprinkles; ++i) {
            const int x = rng.uniform(0, img.cols);
            const int y = rng.uniform(0, img.rows);
            cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
            const uchar v = rng.uniform(0, 2) ? 255 : 0;
            px = cv::Vec3b(v, v, v);
        }
    }
}

static cv::Mat apply_offset(const cv::Mat& img, double dx, double dy) {
    if (dx == 0.0 && dy == 0.0) return img;
    cv::Mat shifted;
    const cv::Matx23d mat(1.0, 0.0, dx, 0.0, 1.0, dy);
    cv::warpAffine(img, shifted, mat, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return shifted;
}

static void accumulate_stats(WeightedStats& total, const DecodeStats& add, double weight) {
    total.symbol_correct += weight * add.symbol_correct;
    total.pattern_correct += weight * add.pattern_correct;
    total.color_correct += weight * add.color_correct;
    total.total += weight * add.total;
}

static double fill_range_penalty(const PatternDict& dict, const ScoreConfig& cfg) {
    if (dict.masks64.empty()) return 0.0;
    double avg_penalty = 0.0;
    double max_penalty = 0.0;
    for (uint64_t mask : dict.masks64) {
        const double ink = static_cast<double>(popcount64(mask));
        double penalty = 0.0;
        if (ink < cfg.min_fill) penalty = (cfg.min_fill - ink) / std::max(1.0, cfg.min_fill);
        else if (ink > cfg.max_fill) penalty = (ink - cfg.max_fill) / std::max(1.0, 64.0 - cfg.max_fill);
        avg_penalty += penalty;
        max_penalty = std::max(max_penalty, penalty);
    }
    avg_penalty /= static_cast<double>(dict.masks64.size());
    return 0.5 * avg_penalty + 0.5 * max_penalty;
}

static double fill_balance_penalty(const PatternDict& dict) {
    if (dict.masks64.size() < 2) return 0.0;
    double mean = 0.0;
    for (uint64_t mask : dict.masks64) mean += static_cast<double>(popcount64(mask));
    mean /= static_cast<double>(dict.masks64.size());
    double variance = 0.0;
    for (uint64_t mask : dict.masks64) {
        const double diff = static_cast<double>(popcount64(mask)) - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(dict.masks64.size());
    return std::sqrt(variance) / 64.0;
}

static double fragile_ink_penalty(const PatternDict& dict) {
    if (dict.masks64.empty()) return 0.0;
    double avg_penalty = 0.0;
    double max_penalty = 0.0;
    for (uint64_t mask : dict.masks64) {
        const int ink = popcount64(mask);
        if (ink <= 0) {
            avg_penalty += 1.0;
            max_penalty = 1.0;
            continue;
        }
        double penalty = 0.0;
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                if (!mask64_is_on(mask, x, y)) continue;
                int neighbors = 0;
                for (int yy = std::max(0, y - 1); yy <= std::min(7, y + 1); ++yy) {
                    for (int xx = std::max(0, x - 1); xx <= std::min(7, x + 1); ++xx) {
                        if (xx == x && yy == y) continue;
                        neighbors += mask64_is_on(mask, xx, yy) ? 1 : 0;
                    }
                }
                if (neighbors <= 0) penalty += 1.0;
                else if (neighbors == 1) penalty += 0.65;
                else if (neighbors == 2) penalty += 0.28;
            }
        }
        penalty /= static_cast<double>(ink);
        avg_penalty += penalty;
        max_penalty = std::max(max_penalty, penalty);
    }
    avg_penalty /= static_cast<double>(dict.masks64.size());
    return 0.65 * avg_penalty + 0.35 * max_penalty;
}

static double nearest_distance_penalty64(const PatternDict& dict) {
    if (dict.masks64.size() < 2) return 0.0;
    constexpr double target = 18.0;
    double avg_penalty = 0.0;
    double max_penalty = 0.0;
    for (size_t i = 0; i < dict.masks64.size(); ++i) {
        int nearest = 64;
        for (size_t j = 0; j < dict.masks64.size(); ++j) {
            if (i == j) continue;
            nearest = std::min(nearest, popcount64(dict.masks64[i] ^ dict.masks64[j]));
        }
        const double penalty = std::max(0.0, target - static_cast<double>(nearest)) / target;
        avg_penalty += penalty;
        max_penalty = std::max(max_penalty, penalty);
    }
    avg_penalty /= static_cast<double>(dict.masks64.size());
    return 0.55 * avg_penalty + 0.45 * max_penalty;
}

static double nearest_distance_penalty16(const PatternDict& dict) {
    if (dict.masks16.size() < 2) return 0.0;
    constexpr double target = 5.0;
    double avg_penalty = 0.0;
    double max_penalty = 0.0;
    for (size_t i = 0; i < dict.masks16.size(); ++i) {
        int nearest = 16;
        for (size_t j = 0; j < dict.masks16.size(); ++j) {
            if (i == j) continue;
            nearest = std::min(nearest, popcount32(static_cast<uint32_t>(dict.masks16[i] ^ dict.masks16[j])));
        }
        const double penalty = std::max(0.0, target - static_cast<double>(nearest)) / target;
        avg_penalty += penalty;
        max_penalty = std::max(max_penalty, penalty);
    }
    avg_penalty /= static_cast<double>(dict.masks16.size());
    return 0.60 * avg_penalty + 0.40 * max_penalty;
}

static double shift_confusion_penalty(const PatternDict& dict) {
    if (dict.masks64.size() < 2) return 0.0;
    static const std::array<std::array<int, 2>, 8> shifts = {{
        {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}},
        {{-1, -1}}, {{1, -1}}, {{-1, 1}}, {{1, 1}},
    }};
    constexpr double target = 15.0;
    double avg_penalty = 0.0;
    double max_penalty = 0.0;
    for (size_t i = 0; i < dict.masks64.size(); ++i) {
        double nearest = 64.0;
        for (size_t j = 0; j < dict.masks64.size(); ++j) {
            if (i == j) continue;
            for (const auto& shift : shifts) {
                const uint64_t shifted = translate_mask64(dict.masks64[j], shift[0], shift[1]);
                const int d64 = popcount64(dict.masks64[i] ^ shifted);
                const int d16 = popcount32(static_cast<uint32_t>(dict.masks16[i] ^ compress_mask64_to_16(shifted)));
                const double combined = 0.72 * static_cast<double>(d64)
                                      + 0.28 * static_cast<double>(d16 * 4);
                nearest = std::min(nearest, combined);
            }
        }
        const double penalty = std::max(0.0, target - nearest) / target;
        avg_penalty += penalty;
        max_penalty = std::max(max_penalty, penalty);
    }
    avg_penalty /= static_cast<double>(dict.masks64.size());
    return 0.55 * avg_penalty + 0.45 * max_penalty;
}

static double blend_score(const WeightedStats& stats, const PatternDict& dict, const ScoreConfig& cfg) {
    if (stats.total <= 0.0) return 0.0;
    const double symbol_acc = stats.symbol_correct / stats.total;
    const double pattern_acc = stats.pattern_correct / stats.total;
    const double color_acc = stats.color_correct / stats.total;
    const double positive_weight = std::max(0.0, cfg.symbol_weight)
        + std::max(0.0, cfg.pattern_weight)
        + std::max(0.0, cfg.color_weight);
    const double norm = positive_weight > 0.0 ? positive_weight : 1.0;
    double score = 0.0;
    score += std::max(0.0, cfg.symbol_weight) * symbol_acc;
    score += std::max(0.0, cfg.pattern_weight) * pattern_acc;
    score += std::max(0.0, cfg.color_weight) * color_acc;
    score /= norm;
    score -= std::max(0.0, cfg.sparse_penalty_weight) * fill_range_penalty(dict, cfg);
    score -= std::max(0.0, cfg.balance_penalty_weight) * fill_balance_penalty(dict);
    score -= std::max(0.0, cfg.fragility_penalty_weight) * fragile_ink_penalty(dict);
    score -= std::max(0.0, cfg.distance64_penalty_weight) * nearest_distance_penalty64(dict);
    score -= std::max(0.0, cfg.distance16_penalty_weight) * nearest_distance_penalty16(dict);
    score -= std::max(0.0, cfg.shift_penalty_weight) * shift_confusion_penalty(dict);
    return std::clamp(score, 0.0, 1.0);
}

static double eval_one_combo(const std::vector<uint64_t>& masks64,
                             const std::vector<double>& scales,
                             const std::vector<double>& weights,
                             const std::vector<unsigned>& rng_seeds,
                             const ScoreConfig& score_cfg) {
    const PatternDict dict = build_pattern_dict(masks64);
    WeightedStats stats;
    for (unsigned seed : rng_seeds) {
        const EncodedFrame encoded = encode_frame(dict, seed);
        for (int si = 0; si < static_cast<int>(scales.size()); ++si) {
            const double scale = scales[si];
            const double scale_weight = weights[si];
            const int px = std::max(2, static_cast<int>(IMG_SIZE * scale + 0.5));
            cv::Mat scaled;
            cv::Mat restored;
            cv::resize(encoded.img, scaled, cv::Size(px, px), 0, 0, cv::INTER_AREA);
            cv::resize(scaled, restored, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_LINEAR);
            cv::RNG cv_rng(static_cast<uint64_t>(seed) * 1315423911ULL + static_cast<uint64_t>(si + 1) * 2654435761ULL);
            stimulate_moire(restored, cv_rng, scale);
            stimulate_blur(restored, cv_rng, scale);
            stimulate_color_cast(restored, cv_rng, scale);
            stimulate_noise(restored, cv_rng, scale);
            for (const OffsetCase& offset_case : OFFSET_CASES) {
                const cv::Mat shifted = apply_offset(restored, offset_case.dx, offset_case.dy);
                const DecodeStats frame_stats = decode_frame(shifted, dict, encoded.raw);
                accumulate_stats(stats, frame_stats, scale_weight * offset_case.weight);
            }
        }
    }
    return blend_score(stats, dict, score_cfg);
}
static std::string read_all_stdin() {
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    return buf.str();
}

static std::string find_array(const std::string& json, const std::string& key) {
    const std::string kstr = "\"" + key + "\"";
    size_t pos = json.find(kstr);
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos);
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t end = pos;
    for (; end < json.size(); ++end) {
        if (json[end] == '[') ++depth;
        else if (json[end] == ']') {
            --depth;
            if (depth == 0) break;
        }
    }
    return json.substr(pos, end - pos + 1);
}

static std::vector<double> parse_double_array(const std::string& arr) {
    std::vector<double> res;
    std::string s = arr;
    for (char& c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ss(s);
    double v = 0.0;
    while (ss >> v) res.push_back(v);
    return res;
}

static std::vector<unsigned> parse_uint_array(const std::string& arr) {
    std::vector<unsigned> res;
    std::string s = arr;
    for (char& c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ss(s);
    unsigned v = 0;
    while (ss >> v) res.push_back(v);
    return res;
}

static std::vector<std::vector<uint64_t>> parse_combos(const std::string& arr) {
    std::vector<std::vector<uint64_t>> res;
    size_t pos = 0;
    while (pos < arr.size()) {
        pos = arr.find('[', pos + 1);
        if (pos == std::string::npos) break;
        const size_t end = arr.find(']', pos);
        if (end == std::string::npos) break;
        const std::string inner = arr.substr(pos + 1, end - pos - 1);
        std::vector<uint64_t> combo;
        std::istringstream ss(inner);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\n' || tok.front() == '\r')) tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\n' || tok.back() == '\r')) tok.pop_back();
            if (!tok.empty()) combo.push_back(std::stoull(tok));
        }
        if (!combo.empty()) res.push_back(combo);
        pos = end + 1;
    }
    return res;
}

static ScoreConfig parse_score_config(const std::string& input) {
    ScoreConfig cfg;
    const std::vector<double> values = parse_double_array(find_array(input, "score_weights"));
    if (values.size() >= 7) {
        cfg.symbol_weight = values[0];
        cfg.pattern_weight = values[1];
        cfg.color_weight = values[2];
        cfg.sparse_penalty_weight = values[3];
        cfg.balance_penalty_weight = values[4];
        cfg.min_fill = values[5];
        cfg.max_fill = values[6];
    }
    if (values.size() >= 11) {
        cfg.fragility_penalty_weight = values[7];
        cfg.distance64_penalty_weight = values[8];
        cfg.distance16_penalty_weight = values[9];
        cfg.shift_penalty_weight = values[10];
    }
    return cfg;
}

int main() {
    const std::string input = read_all_stdin();
    std::vector<double> scales = parse_double_array(find_array(input, "scales"));
    std::vector<double> weights = parse_double_array(find_array(input, "weights"));
    std::vector<unsigned> seeds = parse_uint_array(find_array(input, "seeds"));
    std::vector<std::vector<uint64_t>> combos = parse_combos(find_array(input, "combos"));
    const ScoreConfig score_cfg = parse_score_config(input);

    if (scales.empty()) scales = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    if (weights.empty()) weights = {1, 2, 3, 4, 5, 6};
    if (seeds.empty()) seeds = {0, 1, 2};
    if (weights.size() < scales.size()) weights.resize(scales.size(), weights.empty() ? 1.0 : weights.back());

    std::printf("{\"scores\":[");
    for (int i = 0; i < static_cast<int>(combos.size()); ++i) {
        const double score = eval_one_combo(combos[i], scales, weights, seeds, score_cfg);
        std::printf("%.6f", score);
        if (i + 1 < static_cast<int>(combos.size())) std::printf(",");
    }
    std::printf("]}\n");
    return 0;
}
