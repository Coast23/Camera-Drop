#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util/config.hpp"
#include "vision/pattern_dict.hpp"
#include "vision/recognizer.hpp"

namespace fs = std::filesystem;

namespace {

cv::Vec3b get_color(int color_idx) {
    switch (color_idx) {
        case 0: return cv::Vec3b(0, 255, 255);
        case 1: return cv::Vec3b(0, 255, 0);
        case 2: return cv::Vec3b(255, 255, 0);
        case 3: return cv::Vec3b(255, 0, 255);
        default: return cv::Vec3b(255, 255, 255);
    }
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

bool is_header_cell(int r, int c) {
    return r == Config::HEADER_ROW && c >= Config::HEADER_COL_BEGIN && c < Config::HEADER_COL_END;
}

void draw_symbol_tile(cv::Mat& img, int grid_r, int grid_c, uint8_t symbol, const camdrop::vision::PatternDictionary& dict) {
    const int pat_idx = symbol & (dict.size() - 1);
    const int color_idx = symbol >> dict.pattern_bits();
    const uint64_t mask = dict.masks64[pat_idx];
    const cv::Vec3b color = get_color(color_idx);
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

struct ExpectedData {
    cv::Mat image;
    std::vector<uint8_t> header_symbols;
    std::vector<uint8_t> payload_symbols;
};

ExpectedData generate_test_image(const camdrop::vision::PatternDictionary& dict, uint32_t seed) {
    ExpectedData out;
    out.image = cv::Mat::zeros(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);

    draw_normal_anchor(out.image, Config::ANCHOR_OUT_START, Config::ANCHOR_OUT_START);
    draw_normal_anchor(out.image, Config::IMG_WIDTH - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE, Config::ANCHOR_OUT_START);
    draw_normal_anchor(out.image, Config::ANCHOR_OUT_START, Config::IMG_HEIGHT - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE);
    draw_br_anchor(out.image, Config::IMG_WIDTH - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE, Config::IMG_HEIGHT - Config::ANCHOR_OUT_START - Config::ANCHOR_L1_SIZE);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, (1 << Config::BITS_PER_UNIT) - 1);

    for (int r = 0; r < Config::GRID_R; ++r) {
        for (int c = 0; c < Config::GRID_C; ++c) {
            if (is_anchor_reserved(r, c)) {
                continue;
            }
            uint8_t symbol = 0;
            if (is_calibration_cell(r, c)) {
                symbol = static_cast<uint8_t>((((c - Config::CALIB_COL_BEGIN) & 3) << dict.pattern_bits()) | 0);
            } else {
                symbol = static_cast<uint8_t>(dist(rng));
            }
            draw_symbol_tile(out.image, r, c, symbol, dict);
            if (is_header_cell(r, c)) {
                out.header_symbols.push_back(symbol);
            } else if (!is_calibration_cell(r, c)) {
                out.payload_symbols.push_back(symbol);
            }
        }
    }

    return out;
}

double calc_symbol_acc(const std::vector<uint8_t>& got, const std::vector<uint8_t>& exp) {
    if (got.empty() || exp.empty() || got.size() != exp.size()) {
        return 0.0;
    }
    size_t ok = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        ok += (got[i] == exp[i]) ? 1 : 0;
    }
    return 100.0 * static_cast<double>(ok) / static_cast<double>(got.size());
}

double calc_component_acc(const std::vector<uint8_t>& got, const std::vector<uint8_t>& exp, int mask, int shift) {
    if (got.empty() || exp.empty() || got.size() != exp.size()) {
        return 0.0;
    }
    size_t ok = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const int a = (got[i] >> shift) & mask;
        const int b = (exp[i] >> shift) & mask;
        ok += (a == b) ? 1 : 0;
    }
    return 100.0 * static_cast<double>(ok) / static_cast<double>(got.size());
}

void print_usage() {
    std::cout << "Usage: recognizer_debug [--patterns <dir>] [--seed <n>] [--out <png>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string pattern_dir = "pattern_finder/best_v2";
    std::string out_path = "recognizer_debug.png";
    uint32_t seed = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--patterns" && i + 1 < argc) {
            pattern_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            print_usage();
            return 1;
        }
    }

    try {
        const camdrop::vision::PatternDictionary dict = camdrop::vision::PatternDictionary::LoadFromDirectory(pattern_dir);
        const ExpectedData expected = generate_test_image(dict, seed);
        cv::imwrite(out_path, expected.image);

        camdrop::vision::PatternRecognizer recognizer(dict);
        const camdrop::vision::RecognizeResult decoded = recognizer.Decode(expected.image);
        if (!decoded.ok) {
            std::cerr << "recognizer returned not-ok\n";
            return 2;
        }

        const double header_sym = calc_symbol_acc(decoded.header_symbols, expected.header_symbols);
        const double payload_sym = calc_symbol_acc(decoded.payload_symbols, expected.payload_symbols);
        const double payload_pat = calc_component_acc(decoded.payload_symbols, expected.payload_symbols, dict.size() - 1, 0);
        const double payload_col = calc_component_acc(decoded.payload_symbols, expected.payload_symbols, 0x3, dict.pattern_bits());

        std::cout << "generated: " << fs::absolute(out_path).string() << '\n';
        std::cout << "header symbols: " << decoded.header_symbols.size() << " acc=" << std::fixed << std::setprecision(3) << header_sym << "%\n";
        std::cout << "payload symbols: " << decoded.payload_symbols.size() << " acc=" << payload_sym << "%\n";
        std::cout << "payload pattern acc=" << payload_pat << "% color acc=" << payload_col << "% avgPatternDist=" << decoded.avg_pattern_dist << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 10;
    }
}
