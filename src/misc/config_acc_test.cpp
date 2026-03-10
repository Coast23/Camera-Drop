// Author: Coast23
// Date: 2026-02-26
/*
在 pattern_generator.cpp 的基础上，测试不同配置（图案集大小、颜色数）的抗干扰能力
*/

#include <cstdio>
#include <vector>
#include <random>
#include <string>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>
#ifdef _MSC_VER
#  include <intrin.h>
static inline int portable_popcountll(unsigned long long x) { return static_cast<int>(__popcnt64(x)); }
#else
static inline int portable_popcountll(unsigned long long x) { return __builtin_popcountll(x); }
#endif
constexpr int ct_log2(int x) { return x <= 1 ? 0 : 1 + ct_log2(x >> 1); }

// 固定参数
const int GRID_SIZE = 112;
const int STRIDE    = 9;
const int MARGIN    = 8;
const int IMG_SIZE  = 1024;
const int INF       = 1145141919;

// Anchor 层级常量（TL/TR/BL 和 BR 共用相同尺寸，仅颜色不同）
const int ANCHOR_OUT_START = 2;
const int ANCHOR_L1_SIZE   = 56;   // 外层彩色
const int ANCHOR_L2_INSET  = 7;    // 黑色层 inset
const int ANCHOR_L2_SIZE   = 42;
const int ANCHOR_L3_INSET  = 14;   // 内彩色层 inset
const int ANCHOR_L3_SIZE   = 28;
const int ANCHOR_L4_INSET  = 21;   // 内黑色层 inset
const int ANCHOR_L4_SIZE   = 14;

// 图案数和颜色数
const int NUM_PATTERNS = 16;
const int NUM_COLORS   = 4;

// 信道模拟开关
const bool STIMULATE_MOIRE      = true;
const bool STIMULATE_BLUR       = true;
const bool STIMULATE_COLOR_CAST = true;
const bool STIMULATE_NOISE      = true;

// ── 内置算法生成字典 ─────────────────────────────────────────────
std::vector<uint64_t> gen_dict() {
    auto expand = [&](uint16_t mask) -> uint64_t {
        uint64_t res = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                if ((mask >> (r * 4 + c)) & 1) {
                    uint32_t r8 = r << 1, c8 = c << 1;
                    res |= (1ULL << (r8 * 8 + c8));
                    res |= (1ULL << (r8 * 8 + c8 + 1));
                    res |= (1ULL << ((r8 + 1) * 8 + c8));
                    res |= (1ULL << ((r8 + 1) * 8 + c8 + 1));
                }
        return res;
    };
    auto popcnt = [](uint64_t x) { return portable_popcountll(x); };

    auto get_dist = [&](std::vector<uint16_t>& pick) -> int {
        int res = INF;
        for (int i = 0; i < (int)pick.size(); ++i)
            for (int j = i + 1; j < (int)pick.size(); ++j)
                res = std::min(res, popcnt(pick[i] ^ pick[j]));
        return res;
    };

    std::vector<uint16_t> cand;
    for (int i = 0; i < (1 << 16); ++i) {
        int p = popcnt(i);
        if (p >= 6 && p <= 10) cand.push_back(i);
    }

    std::vector<uint16_t> pick;
    pick.push_back(0x00FF);
    std::vector<int> dist(cand.size(), INF);
    for (int i = 0; i < (int)cand.size(); ++i)
        dist[i] = popcnt(cand[i] ^ pick[0]);

    for (int k = 1; k < NUM_PATTERNS; ++k) {
        int best = -1, maxDist = -1;
        for (int i = 0; i < (int)cand.size(); ++i)
            if (dist[i] > maxDist) { maxDist = dist[i]; best = i; }
        pick.push_back(cand[best]);
        for (int i = 0; i < (int)cand.size(); ++i)
            dist[i] = std::min(dist[i], popcnt(cand[i] ^ pick.back()));
    }

    std::vector<uint64_t> Dict;
    for (auto& m : pick) Dict.push_back(expand(m));
    return Dict;
}

// ── 从外部 patterns_dir 加载字典 ─────────────────────────────────
// 期望 00.png, 01.png, ..., 0f.png（共16个，灰度，任意尺寸）
// 每张图 resize 到 8×8 → 阈值二值化 → 转 uint64 mask
std::vector<uint64_t> load_pattern_dir(const std::string& dir) {
    std::vector<uint64_t> dict;
    for (int i = 0; i < NUM_PATTERNS; i++) {
        char fname[32];
        snprintf(fname, sizeof(fname), "%02x.png", i);
        std::string path = dir + "/" + fname;
        cv::Mat pat = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (pat.empty()) {
            printf("警告: 无法加载 %s，使用零 mask\n", path.c_str());
            dict.push_back(0ULL);
            continue;
        }
        cv::Mat pat8;
        cv::resize(pat, pat8, {8, 8}, 0, 0, cv::INTER_NEAREST);
        cv::Mat bin;
        cv::threshold(pat8, bin, 128, 1, cv::THRESH_BINARY_INV);  // 黑色像素=有效格
        uint64_t mask = 0;
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                if (bin.at<uchar>(r, c)) mask |= (1ULL << (r * 8 + c));
        dict.push_back(mask);
    }
    return dict;
}

int main(int argc, char** argv) {
    assert((NUM_PATTERNS & (NUM_PATTERNS - 1)) == 0);
    assert((NUM_COLORS   & (NUM_COLORS   - 1)) == 0);

    // ── 命令行参数 ───────────────────────────────────────────────
    uint64_t seed      = 114514;
    std::string pat_dir = "";

    if (argc >= 2) seed = std::stoull(argv[1]);
    if (argc >= 3) pat_dir = argv[2];

    printf("seed=%llu  patterns=%s\n", (unsigned long long)seed,
           pat_dir.empty() ? "(内置算法)" : pat_dir.c_str());

    using namespace cv;

    constexpr int P_BITS = ct_log2(NUM_PATTERNS);
    constexpr int C_BITS = ct_log2(NUM_COLORS);

    // ── 颜色表 ───────────────────────────────────────────────────
    auto get_color = [&](int color_idx) -> Vec3b {
        switch (color_idx) {
            case 0: return Vec3b(0, 255, 255);   // Yellow (R+G)
            case 1: return Vec3b(0, 255, 0);     // Green  (G)
            case 2: return Vec3b(255, 255, 0);   // Cyan   (G+B)
            case 3: return Vec3b(255, 0, 255);   // Magenta(R+B)
            case 4: return Vec3b(0, 0, 255);     // Red    (R)
            case 5: return Vec3b(255, 0, 0);     // Blue   (B)
            case 6: return Vec3b(255, 255, 255); // White
            case 7: return Vec3b(0, 128, 255);   // Orange
            default: return Vec3b(255, 255, 255);
        }
    };

    auto match_color = [&](Vec3b pixel) -> int {
        int best = -1, min_d = INF;
        for (int i = 0; i < NUM_COLORS; ++i) {
            Vec3b ref = get_color(i);
            int db = pixel[0] - ref[0];
            int dg = pixel[1] - ref[1];
            int dr = pixel[2] - ref[2];
            int d = db*db + dg*dg + dr*dr;
            if (d < min_d) { min_d = d; best = i; }
        }
        return best;
    };

    // ── 生成字典 ─────────────────────────────────────────────────
    auto Dict = pat_dir.empty() ? gen_dict() : load_pattern_dir(pat_dir);

    auto match_pattern = [&](uint64_t mask) -> int {
        int best = 0, min_d = 65;
        for (int i = 0; i < (int)Dict.size(); ++i) {
            int d = portable_popcountll(mask ^ Dict[i]);
            if (d < min_d) { min_d = d; best = i; }
        }
        return best;
    };

    // ── 布局保留区 ───────────────────────────────────────────────
    auto is_reserved = [&](int r, int c) -> bool {
        if (r < 6 && c < 6)     return true;  // TL
        if (r < 6 && c > 105)   return true;  // TR
        if (r > 105 && c < 6)   return true;  // BL
        if (r > 105 && c > 105) return true;  // BR
        if (r == 0 && c >= 6 && c < 14) return true;   // 颜色校准块
        if (r == 0 && c >= 14 && c < 46) return true;  // 帧头预留
        return false;
    };

// ─────────────────────────────────────────────────────────────────
// Anchor 结构（L1/L2/L3/L4 同心方块）
//   TL/TR/BL（普通）: 白-黑-白-黑（经典同心方块）
//   BR（anchor_br）:  彩-黑-彩-黑（四象限彩色，用于方向识别）
// ─────────────────────────────────────────────────────────────────

    // 普通锚点（白-黑-白-黑）
    auto draw_normal_anchor = [&](Mat& img, int x0, int y0) -> void {
        rectangle(img, Rect(x0, y0, ANCHOR_L1_SIZE, ANCHOR_L1_SIZE),
                  Scalar(255,255,255), FILLED);
        rectangle(img, Rect(x0+ANCHOR_L2_INSET, y0+ANCHOR_L2_INSET,
                            ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), Scalar(0,0,0), FILLED);
        rectangle(img, Rect(x0+ANCHOR_L3_INSET, y0+ANCHOR_L3_INSET,
                            ANCHOR_L3_SIZE, ANCHOR_L3_SIZE), Scalar(255,255,255), FILLED);
        rectangle(img, Rect(x0+ANCHOR_L4_INSET, y0+ANCHOR_L4_INSET,
                            ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), Scalar(0,0,0), FILLED);
    };

    // BR 锚点（彩-黑-彩-黑）—— 四象限彩色，唯一且用于方向识别
    auto draw_br_anchor = [&](Mat& img, int x0, int y0) -> void {
        // L1: 56×56 四象限彩色
        int h1 = ANCHOR_L1_SIZE / 2;
        rectangle(img, Rect(x0,      y0,      h1, h1), Vec3b(0,255,255),  FILLED); // Yellow
        rectangle(img, Rect(x0+h1,   y0,      h1, h1), Vec3b(0,255,0),    FILLED); // Green
        rectangle(img, Rect(x0,      y0+h1,   h1, h1), Vec3b(255,0,255),  FILLED); // Magenta
        rectangle(img, Rect(x0+h1,   y0+h1,   h1, h1), Vec3b(255,255,0), FILLED); // Cyan
        // L2: 42×42 黑色
        rectangle(img, Rect(x0+ANCHOR_L2_INSET, y0+ANCHOR_L2_INSET,
                            ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), Scalar(0,0,0), FILLED);
        // L3: 28×28 四象限彩色
        int h3 = ANCHOR_L3_SIZE / 2;
        int ix = x0 + ANCHOR_L3_INSET, iy = y0 + ANCHOR_L3_INSET;
        rectangle(img, Rect(ix,    iy,    h3, h3), Vec3b(0,255,255),  FILLED); // Yellow
        rectangle(img, Rect(ix+h3, iy,    h3, h3), Vec3b(0,255,0),    FILLED); // Green
        rectangle(img, Rect(ix,    iy+h3, h3, h3), Vec3b(255,0,255),  FILLED); // Magenta
        rectangle(img, Rect(ix+h3, iy+h3, h3, h3), Vec3b(255,255,0), FILLED); // Cyan
        // L4: 14×14 黑色
        rectangle(img, Rect(x0+ANCHOR_L4_INSET, y0+ANCHOR_L4_INSET,
                            ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), Scalar(0,0,0), FILLED);
    };

    auto draw_anchors = [&](Mat& img) -> void {
        constexpr int tl_x = ANCHOR_OUT_START;
        constexpr int tl_y = ANCHOR_OUT_START;
        constexpr int tr_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int tr_y = ANCHOR_OUT_START;
        constexpr int bl_x = ANCHOR_OUT_START;
        constexpr int bl_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int br_x = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int br_y = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;

        draw_normal_anchor(img, tl_x, tl_y);  // TL: 白黑白黑
        draw_normal_anchor(img, tr_x, tr_y);  // TR: 白黑白黑
        draw_normal_anchor(img, bl_x, bl_y);  // BL: 白黑白黑
        draw_br_anchor    (img, br_x, br_y);  // BR: 彩黑彩黑（方向标记）
    };

    // ── 编码 ─────────────────────────────────────────────────────
    Mat encoder_img(IMG_SIZE, IMG_SIZE, CV_8UC3, Scalar(0, 0, 0));

    std::vector<uint8_t> raw_data(GRID_SIZE * GRID_SIZE + 1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_data(0, NUM_PATTERNS * NUM_COLORS - 1);

    draw_anchors(encoder_img);

    // 标准颜色校准块
    for (int i = 0; i < 8; ++i) {
        int startX = MARGIN + (6 + i) * STRIDE;
        int startY = MARGIN + 0 * STRIDE;
        rectangle(encoder_img, Rect(startX, startY, 8, 8), get_color(i % NUM_COLORS), FILLED);
    }
    // 帧头预留（灰色）
    for (int i = 14; i < 46; ++i)
        rectangle(encoder_img, Rect(MARGIN + i * STRIDE, MARGIN, 8, 8), Scalar(128, 128, 128), FILLED);

    int valid_data_tiles = 0;
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (is_reserved(r, c)) continue;
            ++valid_data_tiles;

            uint8_t data = dist_data(rng);
            raw_data[r * GRID_SIZE + c] = data;

            int pattern_idx = data & (NUM_PATTERNS - 1);
            int color_idx   = data >> P_BITS;

            Vec3b draw_color = get_color(color_idx);
            uint64_t mask    = Dict[pattern_idx];

            int startX = MARGIN + c * STRIDE;
            int startY = MARGIN + r * STRIDE;

            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if ((mask >> (pr * 8 + pc)) & 1)
                        encoder_img.at<Vec3b>(startY + pr, startX + pc) = draw_color;
        }
    }

    // 输出文件名带种子
    std::string seed_str = std::to_string(seed);
    std::string enc_file    = "encoded_" + seed_str + ".png";
    std::string camera_file = "camera_"  + seed_str + ".png";
    imwrite(enc_file, encoder_img);

    // ── 信道模拟 ─────────────────────────────────────────────────
    Mat camera_img = encoder_img.clone();

    auto stimulate_moire = [](Mat& img) {
        for (int r = 0; r < IMG_SIZE; ++r)
            for (int c = 0; c < IMG_SIZE; ++c) {
                float m = 0.85f + 0.20f * sin(r * 0.45f + c * 0.35f);
                Vec3b& px = img.at<Vec3b>(r, c);
                px[0] = saturate_cast<uchar>(px[0] * m);
                px[1] = saturate_cast<uchar>(px[1] * m);
                px[2] = saturate_cast<uchar>(px[2] * m);
            }
    };
    auto stimulate_blur = [](Mat& img) {
        GaussianBlur(img, img, Size(5, 5), 1.2);
    };
    auto stimulate_color_cast = [](Mat& img) {
        for (int r = 0; r < IMG_SIZE; ++r)
            for (int c = 0; c < IMG_SIZE; ++c) {
                Vec3b& px = img.at<Vec3b>(r, c);
                px[0] = saturate_cast<uchar>(px[0] * 0.8 + 50);
                px[1] = saturate_cast<uchar>(px[1] * 0.9 + 50);
                px[2] = saturate_cast<uchar>(px[2] * 1.1 + 40);
            }
    };
    auto stimulate_noise = [](Mat& img) {
        Mat noise(IMG_SIZE, IMG_SIZE, CV_8UC3);
        randn(noise, Scalar(0,0,0), Scalar(15,15,15));
        add(img, noise, img);
    };

    if (STIMULATE_MOIRE)      stimulate_moire(camera_img);
    if (STIMULATE_BLUR)       stimulate_blur(camera_img);
    if (STIMULATE_COLOR_CAST) stimulate_color_cast(camera_img);
    if (STIMULATE_NOISE)      stimulate_noise(camera_img);
    imwrite(camera_file, camera_img);

    // ── 评测 ─────────────────────────────────────────────────────
    Mat gray_img;
    cvtColor(camera_img, gray_img, COLOR_BGR2GRAY);

    int correct = 0, error_patterns = 0, error_colors = 0;

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (is_reserved(r, c)) continue;

            int startX = MARGIN + c * STRIDE;
            int startY = MARGIN + r * STRIDE;
            Rect roi(startX, startY, 8, 8);

            Mat cell_gray = gray_img(roi);
            Mat cell_bgr  = camera_img(roi);

            Mat bin_cell;
            threshold(cell_gray, bin_cell, 0, 255, THRESH_BINARY | THRESH_OTSU);

            uint64_t tile_mask = 0;
            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if (bin_cell.at<uchar>(pr, pc) > 128)
                        tile_mask |= (1ULL << (pr * 8 + pc));

            int best_pat = match_pattern(tile_mask);

            int sumB = 0, sumG = 0, sumR = 0, valid_px = 0;
            uint64_t best_mask = Dict[best_pat];
            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if ((best_mask >> (pr * 8 + pc)) & 1) {
                        Vec3b& px = cell_bgr.at<Vec3b>(pr, pc);
                        sumB += px[0]; sumG += px[1]; sumR += px[2];
                        ++valid_px;
                    }

            int best_color = 0;
            if (valid_px > 0) {
                Vec3b avg(sumB/valid_px, sumG/valid_px, sumR/valid_px);
                best_color = match_color(avg);
            }

            uint8_t decoded  = (best_color << P_BITS) | best_pat;
            uint8_t expected = raw_data[r * GRID_SIZE + c];

            if (decoded == expected) ++correct;
            else {
                if (best_pat   != (expected & (NUM_PATTERNS - 1))) ++error_patterns;
                if (best_color != (expected >> P_BITS))             ++error_colors;
            }
        }
    }

    int payload_bytes = valid_data_tiles * (P_BITS + C_BITS) / 8;

    puts("========================================");
    printf("seed=%llu  patterns=%s\n", (unsigned long long)seed,
           pat_dir.empty() ? "(内置)" : pat_dir.c_str());
    printf("Configuration: %d patterns, %d colors\n", NUM_PATTERNS, NUM_COLORS);
    printf("Payload per Frame: %d Bytes\n", payload_bytes);
    printf("Correctly Decoded: %d / %d\n", correct, valid_data_tiles);
    printf("Error Patterns: %d  Error Colors: %d\n", error_patterns, error_colors);
    printf("Accuracy: %.2f%%\n", 100.0 * correct / valid_data_tiles);
    puts("========================================");

    return 0;
}
