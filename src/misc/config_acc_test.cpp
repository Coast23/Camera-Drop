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

// 固定参数
const int GRID_SIZE = 112;
const int STRIDE = 9;
const int MARGIN = 8;
const int IMG_SIZE = 1024;
const int INF = 1145141919;

// 图案数和颜色数可调，但必须是 2 的次幂，且颜色数不能超过 8
const int NUM_PATTERNS = 16;  // 图案数
const int NUM_COLORS = 4;     // 颜色数

// 信道模拟开关
const bool STIMULATE_MOIRE = true;      // 摩尔纹
const bool STIMULATE_BLUR = true;       // 失焦模糊
const bool STIMULATE_COLOR_CAST = true; // 相机偏色
const bool STIMULATE_NOISE = true;      // 高斯白噪声

std::vector<uint64_t> gen_dict(){
    // 把 4x4 的 mask 拓展为 8x8 的 mask
    auto expand = [&](uint16_t mask) -> uint64_t {
        uint64_t res = 0;
        for(int r = 0; r < 4; ++r){
            for(int c = 0; c < 4; ++c){
                if((mask >> (r * 4 + c)) & 1){
                    uint32_t r8 = r << 1;
                    uint32_t c8 = c << 1;
                    res |= (1ULL << (r8 * 8 + c8));            // 左上
                    res |= (1ULL << (r8 * 8 + c8 + 1));        // 右上
                    res |= (1ULL << ((r8 + 1) * 8 + c8));      // 左下
                    res |= (1ULL << ((r8 + 1) * 8 + c8 + 1));  // 右下
                }
            }
        }
        return res;
    };

    auto popcnt = [&](uint64_t x) -> int {
        return __builtin_popcountll(x);
    };

    // 获取图案集的最小 Hamming Distance
    auto get_dist = [&](std::vector<uint16_t>& pick) -> int {
        int res = INF;
        int n = pick.size();
        for(int i = 0; i < n; ++i){
            for(int j = i + 1; j < n; ++j){
                res = std::min(res, popcnt(pick[i] ^ pick[j]));
            }
        }
        return res;
    };

    // 4x4 图案中，只选白色个数在 6~10 之间的图案
    std::vector<uint16_t> cand;
    for(int i = 0; i < (1 << 16); ++i){
        int p = popcnt(i);
        if(p >= 6 and p <= 10) cand.push_back(i);
    }

    // 钦定第一个为黑白上下对半的图案。
    std::vector<uint16_t> pick; pick.push_back(0x00FF);
    std::vector<int> dist(cand.size(), INF);

    for(int i = 0; i < cand.size(); ++i) dist[i] = popcnt(cand[i] ^ pick[0]);

    for(int k = 1; k < NUM_PATTERNS; ++k){
        int best = -1, maxDist = -1;
        for(int i = 0; i < cand.size(); ++i){
            if(dist[i] > maxDist){
                maxDist = dist[i];
                best = i;
            }
        }
        uint16_t picked = cand[best];
        pick.push_back(picked);
        for(int i = 0; i < cand.size(); ++i) dist[i] = std::min(dist[i], popcnt(cand[i] ^ picked));
    }
    
    std::vector<uint64_t> Dict;
    for(auto& mask4x4 : pick) Dict.push_back(expand(mask4x4));
    return Dict;
}

int main(){
    // 必须是 2 的次幂
    assert((NUM_PATTERNS & (NUM_PATTERNS - 1)) == 0);
    assert((NUM_COLORS & (NUM_COLORS - 1)) == 0);

    using namespace cv;
    constexpr int P_BITS = std::__lg(NUM_PATTERNS);
    constexpr int C_BITS = std::__lg(NUM_COLORS);

    auto get_color = [&](int color_idx) -> Vec3b {
        switch(color_idx){
            case 0: return Vec3b(0, 255, 255);   // Yellow (R+G)
            case 1: return Vec3b(0, 255, 0);     // Green  (G)
            case 2: return Vec3b(255, 255, 0);   // Cyan   (G+B)
            case 3: return Vec3b(255, 0, 255);   // Magenta(R+B)
            case 4: return Vec3b(0, 0, 255);     // Red    (R)
            case 5: return Vec3b(255, 0, 0);     // Blue   (B)
            case 6: return Vec3b(255, 255, 255); // White
            case 7: return Vec3b(0, 128, 255);   // Black 的代替
            default: return Vec3b(255, 255, 255);
        }
    };

    auto match_color = [&](Vec3b pixel) -> int {
        int best = -1, min_d = INF;
        for(int i = 0; i < NUM_COLORS; ++i){
            Vec3b ref = get_color(i);
            int db = pixel[0] - ref[0];
            int dg = pixel[1] - ref[1];
            int dr = pixel[2] - ref[2];
            int d = db * db + dg * dg + dr * dr;
            if(d < min_d){
                min_d = d;
                best = i;
            }
        }
        return best;
    };

    auto Dict = gen_dict();

    auto match_pattern = [&Dict](uint64_t mask) -> int {
        int best = 0, min_d = 65;
        for(int i = 0; i < Dict.size(); ++i){
            int dist =  __builtin_popcountll(mask ^ Dict[i]);
            if(dist < min_d){
                min_d = dist;
                best = i;
            }
        }
        return best;
    };

    // 布局映射器：判断是否是非编码区
    auto is_reserved = [&](int r, int c) -> bool {
        // 四个角的 7x7 定位块
        if(r < 7 and c < 7) return true;         // TL
        if(r < 7 and c >= 105) return true;      // TR
        if(r >= 105 and c < 7) return true;      // BL
        if(r >= 105 and c >= 105) return true;   // BR
        // 颜色校准块
        if(r == 0 and c >= 7 and c < 15) return true;
        // 帧头预留区
        if(r == 0 and c >= 15 and c < 47) return true;
        return false; // 可用数据块
    };

    auto draw_finder_pattern = [&](Mat& img, int grid_r, int grid_c) -> void {
        int x = MARGIN + grid_c * STRIDE;
        int y = MARGIN + grid_r * STRIDE;
        // 外圈白框背景(由于默认全黑，直接填白色和内部黑色即可)
        rectangle(img, Rect(x, y, 63, 63), Scalar(255, 255, 255), FILLED);
        rectangle(img, Rect(x + 9, y + 9, 45, 45), Scalar(0, 0, 0), FILLED);
        rectangle(img, Rect(x + 18, y + 18, 27, 27), Scalar(255, 255, 255), FILLED);
    };

    Mat encoder_img(IMG_SIZE, IMG_SIZE, CV_8UC3, Scalar(0, 0, 0));
    
    std::vector<uint8_t> raw_data(GRID_SIZE * GRID_SIZE + 1);
    std::mt19937 rng(114514);
    std::uniform_int_distribution<int> dist_data(0, NUM_PATTERNS * NUM_COLORS - 1);
    
    // ===========================
    //      绘制布局宏观元素
    // ===========================
    draw_finder_pattern(encoder_img, 0, 0);       // Top-Left
    draw_finder_pattern(encoder_img, 0, 105);     // Top-Right
    draw_finder_pattern(encoder_img, 105, 0);     // Bottom-Left
    draw_finder_pattern(encoder_img, 105, 105);   // Bottom-Right

    // 标准颜色块
    for(int i = 0; i < 8; ++i){
        int startX = MARGIN + (7 + i) * STRIDE;
        int startY = MARGIN + 0 * STRIDE;
        rectangle(encoder_img, Rect(startX, startY, 8, 8), get_color(i % NUM_COLORS), FILLED);
    }
    // Frame Header 预留
    for(int i = 15; i < 47; ++i){
        rectangle(encoder_img, Rect(MARGIN + i * STRIDE, MARGIN, 8, 8), Scalar(128, 128, 128), FILLED);
    }

    // ===========================
    //      编码实际数据
    // ===========================

    int valid_data_tiles = 0;
    for(int r = 0; r < GRID_SIZE; ++r){
        for(int c = 0; c < GRID_SIZE; ++c){
            if(is_reserved(r, c)) continue;
            ++valid_data_tiles;

            uint8_t data = dist_data(rng);
          //  raw_data.push_back(data);
            raw_data[r * GRID_SIZE + c] = data;

            // 高 C_BITS 位为颜色，低 P_BITS 位为图案
            int pattern_idx = data & (NUM_PATTERNS - 1);
            int color_idx = data >> P_BITS;
        
            Vec3b draw_color = get_color(color_idx);
            uint64_t mask = Dict[pattern_idx];
        
            int startX = MARGIN + c * STRIDE;
            int startY = MARGIN + r * STRIDE;
        
            for(int pr = 0; pr < 8; ++pr){
                for(int pc = 0; pc < 8; ++pc){
                    if((mask >> (pr * 8 + pc)) & 1){
                        encoder_img.at<Vec3b>(startY + pr, startX + pc) = draw_color;
                    }
                }
            }
        }
    }
    
    std::string enc_file = format("encoded_%dp%dc.png", NUM_PATTERNS, NUM_COLORS);
    imwrite(enc_file, encoder_img);

    // ===========================
    //      模拟恶劣信道
    // ===========================

    Mat camera_img = encoder_img.clone();
    
    // 模拟摩尔纹
    auto stimulate_moire = [](Mat& camera_img) -> void {
        for(int r = 0; r < IMG_SIZE; ++r) {
            for(int c = 0; c < IMG_SIZE; ++c) {
                float moire = 0.85f + 0.20f * sin(r * 0.45f + c * 0.35f);
                Vec3b& px = camera_img.at<Vec3b>(r, c);
                px[0] = saturate_cast<uchar>(px[0] * moire);
                px[1] = saturate_cast<uchar>(px[1] * moire);
                px[2] = saturate_cast<uchar>(px[2] * moire);
            }
        }
    };

    // 模拟失焦模糊
    auto stimulate_blur = [](Mat& camera_img) -> void {
        GaussianBlur(camera_img, camera_img, Size(5, 5), 1.2); 
    };

    // 模拟相机偏色和曝光底噪
    auto stimulate_color_cast = [](Mat& camera_img) -> void {
        for(int r = 0; r < IMG_SIZE; ++r) {
            for(int c = 0; c < IMG_SIZE; ++c) {
                Vec3b& px = camera_img.at<Vec3b>(r, c);
                px[0] = saturate_cast<uchar>(px[0] * 0.8 + 50);  // Blue 衰减
                px[1] = saturate_cast<uchar>(px[1] * 0.9 + 50);  // Green 衰减
                px[2] = saturate_cast<uchar>(px[2] * 1.1 + 40);  // Red 增强
            }
        }
    };
    
    // 高斯白噪声
    auto stimulate_noise = [](Mat& camera_img) -> void {
        Mat noise(IMG_SIZE, IMG_SIZE, CV_8UC3);
        randn(noise, Scalar(0,0,0), Scalar(15,15,15)); 
        add(camera_img, noise, camera_img);
    };

    if(STIMULATE_MOIRE) stimulate_moire(camera_img);
    if(STIMULATE_BLUR) stimulate_blur(camera_img);
    if(STIMULATE_COLOR_CAST) stimulate_color_cast(camera_img);
    if(STIMULATE_NOISE) stimulate_noise(camera_img);

    std::string camera_file = format("camera_%dp%dc.png", NUM_PATTERNS, NUM_COLORS);
    imwrite(camera_file, camera_img);

    // ===========================
    //      解码与评测
    // ===========================

    Mat gray_img;
    cvtColor(camera_img, gray_img, COLOR_BGR2GRAY);
    
    int correct = 0;
    int error_patterns = 0;
    int error_colors = 0;

    for(int r = 0; r < GRID_SIZE; ++r){
        for(int c = 0; c < GRID_SIZE; ++c){
            if(is_reserved(r, c)) continue;

            int startX = MARGIN + c * STRIDE;
            int startY = MARGIN + r * STRIDE;
            Rect roi(startX, startY, 8, 8);
            
            Mat cell_gray = gray_img(roi);
            Mat cell_bgr  = camera_img(roi);

            Mat binary_cell;
            threshold(cell_gray, binary_cell, 0, 255, THRESH_BINARY | THRESH_OTSU);
        
            uint64_t tile_mask = 0;
            for(int pr = 0; pr < 8; ++pr){
                for(int pc = 0; pc < 8; ++pc){
                    if(binary_cell.at<uchar>(pr, pc) > 128){
                        tile_mask |= (1ULL << (pr * 8 + pc));
                    }
                }
            }
            int best_pat = match_pattern(tile_mask);
            int sumB = 0, sumG = 0, sumR = 0, valid_pixels = 0;
            uint64_t best_mask = Dict[best_pat];
            for(int pr = 0; pr < 8; ++pr){
                for(int pc = 0; pc < 8; ++pc){
                    if((best_mask >> (pr * 8 + pc)) & 1){
                        Vec3b& px = cell_bgr.at<Vec3b>(pr, pc);
                        sumB += px[0];
                        sumG += px[1];
                        sumR += px[2];
                        ++valid_pixels;
                    }
                }
            }
            int best_color = 0;
            if(valid_pixels > 0){
                Vec3b avg_color = Vec3b(sumB / valid_pixels, sumG / valid_pixels, sumR / valid_pixels);
                best_color = match_color(avg_color);
            }
            
            uint8_t decoded_byte = (best_color << P_BITS) | best_pat;
            uint8_t expected_byte = raw_data[r * GRID_SIZE + c];
            
            if(decoded_byte == expected_byte) ++correct;
            else{
                if(best_pat != (expected_byte & (NUM_PATTERNS - 1))) ++error_patterns;
                if(best_color != (expected_byte >> P_BITS)) ++error_colors;
            }
        }
    }

    int payload_bytes = valid_data_tiles * (NUM_PATTERNS + NUM_COLORS) >> 3;
    
    puts("========================================");
    printf("Configuration: %d patterns, %d colors\n", NUM_PATTERNS, NUM_COLORS);
    printf("Payload per Frame: %d Bytes\n", payload_bytes);
    printf("Correctly Decoded: %d / %d\n", correct, valid_data_tiles);
    printf("Error Patterns: %d\n", error_patterns);
    printf("Error Colors: %d\n", error_colors);
    printf("Accuracy: %.2f%%\n", 100.0 * correct / valid_data_tiles);
    puts("========================================");
}