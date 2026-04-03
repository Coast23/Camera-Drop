#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util/config.hpp"
#include "vision/frame_pipeline.hpp"
#include "vision/frame_renderer.hpp"
#include "vision/pattern_dict.hpp"
#include "vision/recognizer.hpp"
#include "vision/visual_frame_codec.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string source_path;
    std::string capture_path;
    std::string out_dir = "recognizer_diff_out";
    std::string dump_patches_dir;
    std::string model_path = "web/model/best_dynamic.onnx";
    std::string pattern_dir = "pattern_finder/best_v2";
    std::string pattern_cnn_model_path;
    bool source_deskewed = false;
    bool capture_deskewed = false;
};

struct DecodedFrame {
    cv::Mat raw;
    cv::Mat deskewed;
    camdrop::vision::RecognizeResult result;
    bool ok = false;
};

void print_usage() {
    std::cout
        << "Usage: recognizer_diff --source <image> --capture <image> [--out <dir>]\n"
        << "                       [--dump-patches <dir>] [--model <onnx>] [--patterns <dir>]\n"
        << "                       [--pattern-cnn-model <onnx>]\n"
        << "                       [--source-deskewed] [--capture-deskewed]\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            opts.source_path = argv[++i];
        } else if (arg == "--capture" && i + 1 < argc) {
            opts.capture_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            opts.out_dir = argv[++i];
        } else if (arg == "--dump-patches" && i + 1 < argc) {
            opts.dump_patches_dir = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            opts.pattern_dir = argv[++i];
        } else if (arg == "--pattern-cnn-model" && i + 1 < argc) {
            opts.pattern_cnn_model_path = argv[++i];
        } else if (arg == "--source-deskewed") {
            opts.source_deskewed = true;
        } else if (arg == "--capture-deskewed") {
            opts.capture_deskewed = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.source_path.empty() || opts.capture_path.empty()) {
        throw std::runtime_error("missing --source or --capture");
    }
    return opts;
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

DecodedFrame decode_frame(const fs::path& path,
                          camdrop::vision::FramePipeline* pipeline,
                          camdrop::vision::PatternRecognizer* recognizer,
                          bool input_deskewed) {
    DecodedFrame out;
    out.raw = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (out.raw.empty()) {
        throw std::runtime_error("failed to load image: " + path.string());
    }

    if (input_deskewed) {
        out.deskewed = out.raw;
        out.result = recognizer->Decode(out.deskewed);
        out.ok = out.result.ok;
        return out;
    }

    if (!pipeline) {
        throw std::runtime_error("pipeline is required for non-deskewed input");
    }
    const auto res = pipeline->Process(out.raw);
    out.ok = res.recognized;
    if (out.ok) {
        out.deskewed = res.deskewed_image;
        out.result = res.recognize;
    }
    return out;
}

void blend_rect(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, double alpha) {
    cv::Rect clamped = rect & cv::Rect(0, 0, img.cols, img.rows);
    if (clamped.width <= 0 || clamped.height <= 0) return;
    cv::Mat roi = img(clamped);
    cv::Mat overlay(roi.size(), roi.type(), color);
    cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
}

void ensure_dir(const fs::path& dir) {
    if (dir.empty()) return;
    fs::create_directories(dir);
}

cv::Mat extract_gray_patch12(const cv::Mat& img, int x, int y) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    const int patch_size = Config::TILE_SIZE + 4;
    const int sx = std::clamp(x - 2, 0, img.cols - patch_size);
    const int sy = std::clamp(y - 2, 0, img.rows - patch_size);
    return gray(cv::Rect(sx, sy, patch_size, patch_size)).clone();
}

}  // namespace

fs::path find_resource(const char* argv0, const char* relative_path) {
    fs::path exe_dir = fs::absolute(fs::path(argv0)).parent_path();
    fs::path local = exe_dir / relative_path;
    if (fs::exists(local)) return local;
    const char* source_dir = std::getenv("CAMDROP_SOURCE_DIR");
    if (source_dir) {
        fs::path dev = fs::path(source_dir) / relative_path;
        if (fs::exists(dev)) return dev;
    }
    return {};
}

fs::path default_model_path(const char* argv0) {
    auto p = find_resource(argv0, "web/model/best_dynamic.onnx");
    if (!p.empty()) return p;
    return "web/model/best_dynamic.onnx";
}

fs::path default_pattern_dir(const char* argv0) {
    auto p = find_resource(argv0, "pattern_finder/best_v2");
    if (!p.empty()) return p;
    return "pattern_finder/best_v2";
}

fs::path default_pattern_cnn_model_path(const char* argv0) {
    return find_resource(argv0, "cnn/models/pattern_cnn_se110.onnx");
}

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.model_path == "web/model/best_dynamic.onnx")
            opts.model_path = default_model_path(argv[0]).string();
        if (opts.pattern_dir == "pattern_finder/best_v2")
            opts.pattern_dir = default_pattern_dir(argv[0]).string();
        if (opts.pattern_cnn_model_path.empty())
            opts.pattern_cnn_model_path = default_pattern_cnn_model_path(argv[0]).string();
        ensure_dir(opts.out_dir);

        const camdrop::vision::PatternDictionary dict =
            camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir);
        camdrop::vision::PatternRecognizer recognizer(dict);

        camdrop::vision::FramePipelineConfig cfg;
        cfg.model_path = opts.model_path;
        cfg.pattern_dir = opts.pattern_dir;
        cfg.pattern_cnn_model_path = opts.pattern_cnn_model_path;
        camdrop::vision::FramePipeline pipeline(cfg);

        camdrop::vision::FramePipeline* source_pipeline = opts.source_deskewed ? nullptr : &pipeline;
        camdrop::vision::FramePipeline* capture_pipeline = opts.capture_deskewed ? nullptr : &pipeline;

        const DecodedFrame source = decode_frame(opts.source_path, source_pipeline, &recognizer, opts.source_deskewed);
        const DecodedFrame capture = decode_frame(opts.capture_path, capture_pipeline, &recognizer, opts.capture_deskewed);

        if (!source.ok) {
            std::cerr << "source decode failed\n";
            return 2;
        }
        if (!capture.ok) {
            std::cerr << "capture decode failed\n";
            return 3;
        }

        const std::vector<uint8_t> src_symbols = camdrop::vision::RecognizeResultToInterleavedSymbols(source.result);
        const std::vector<uint8_t> cap_symbols = camdrop::vision::RecognizeResultToInterleavedSymbols(capture.result);
        if (src_symbols.size() != cap_symbols.size()) {
            std::cerr << "symbol count mismatch: src=" << src_symbols.size()
                      << " cap=" << cap_symbols.size() << "\n";
            return 4;
        }

        const int pat_mask = dict.size() - 1;
        const int pat_bits = dict.pattern_bits();
        const int edge_margin = 8;

        size_t idx = 0;
        size_t symbol_ok = 0, pattern_ok = 0, color_ok = 0;
        size_t header_ok = 0, header_total = 0;
        size_t edge_total = 0, edge_symbol_err = 0;
        size_t center_total = 0, center_symbol_err = 0;
        std::vector<int> pattern_total(dict.size(), 0);
        std::vector<int> pattern_err(dict.size(), 0);
        std::vector<int> pattern_confusion(dict.size() * dict.size(), 0);

        std::vector<int> row_err(Config::GRID_R, 0);
        std::vector<int> col_err(Config::GRID_C, 0);

        cv::Mat diff_symbol = capture.deskewed.clone();
        cv::Mat diff_pattern = capture.deskewed.clone();
        cv::Mat diff_color = capture.deskewed.clone();

        std::ofstream patch_csv;
        fs::path patch_root;
        if (!opts.dump_patches_dir.empty()) {
            patch_root = fs::path(opts.dump_patches_dir);
            ensure_dir(patch_root);
            ensure_dir(patch_root / "patches");
            patch_csv.open((patch_root / "labels.csv").string(), std::ios::out | std::ios::trunc);
            patch_csv << "patch_path,row,col,src_symbol,src_pattern,src_color,cap_symbol,cap_pattern,cap_color,symbol_ok,pattern_ok,color_ok\n";
        }

        for (int r = 0; r < Config::GRID_R; ++r) {
            for (int c = 0; c < Config::GRID_C; ++c) {
                if (is_anchor_reserved(r, c)) continue;
                if (is_calibration_cell(r, c)) continue;

                const uint8_t src_sym = src_symbols[idx];
                const uint8_t cap_sym = cap_symbols[idx];
                const int src_pat = src_sym & pat_mask;
                const int cap_pat = cap_sym & pat_mask;
                const bool sym_ok = (src_sym == cap_sym);
                const bool pat_ok_cell = ((src_sym & pat_mask) == (cap_sym & pat_mask));
                const bool col_ok_cell = ((src_sym >> pat_bits) == (cap_sym >> pat_bits));

                if (src_pat >= 0 && src_pat < dict.size()) {
                    pattern_total[src_pat]++;
                    if (!pat_ok_cell) {
                        pattern_err[src_pat]++;
                        if (cap_pat >= 0 && cap_pat < dict.size()) {
                            pattern_confusion[src_pat * dict.size() + cap_pat]++;
                        }
                    }
                }

                symbol_ok += sym_ok ? 1 : 0;
                pattern_ok += pat_ok_cell ? 1 : 0;
                color_ok += col_ok_cell ? 1 : 0;

                const bool is_header = is_header_cell(r, c);
                if (is_header) {
                    header_total++;
                    header_ok += sym_ok ? 1 : 0;
                }

                const bool is_edge = (r < edge_margin || c < edge_margin
                                   || r >= Config::GRID_R - edge_margin
                                   || c >= Config::GRID_C - edge_margin);
                if (is_edge) {
                    edge_total++;
                    if (!sym_ok) edge_symbol_err++;
                } else {
                    center_total++;
                    if (!sym_ok) center_symbol_err++;
                }

                if (!sym_ok) {
                    row_err[r]++;
                    col_err[c]++;
                    const int x = Config::MARGIN + c * Config::STRIDE;
                    const int y = Config::MARGIN + r * Config::STRIDE;
                    const cv::Rect rect(x, y, Config::TILE_SIZE, Config::TILE_SIZE);

                    blend_rect(diff_symbol, rect, cv::Scalar(0, 0, 255), 0.45);
                    if (!pat_ok_cell) {
                        blend_rect(diff_pattern, rect, cv::Scalar(0, 128, 255), 0.45);
                    }
                    if (!col_ok_cell) {
                        blend_rect(diff_color, rect, cv::Scalar(255, 255, 0), 0.45);
                    }
                }

                if (patch_csv.is_open()) {
                    const int x = Config::MARGIN + c * Config::STRIDE;
                    const int y = Config::MARGIN + r * Config::STRIDE;
                    const std::string patch_name =
                        "r" + std::to_string(r) +
                        "_c" + std::to_string(c) +
                        "_sp" + std::to_string(src_pat) +
                        "_cp" + std::to_string(cap_pat) + ".png";
                    const fs::path patch_rel = fs::path("patches") / patch_name;
                    const cv::Mat patch = extract_gray_patch12(capture.deskewed, x, y);
                    cv::imwrite((patch_root / patch_rel).string(), patch);
                    patch_csv << patch_rel.generic_string() << ','
                              << r << ','
                              << c << ','
                              << static_cast<int>(src_sym) << ','
                              << src_pat << ','
                              << (static_cast<int>(src_sym) >> pat_bits) << ','
                              << static_cast<int>(cap_sym) << ','
                              << cap_pat << ','
                              << (static_cast<int>(cap_sym) >> pat_bits) << ','
                              << (sym_ok ? 1 : 0) << ','
                              << (pat_ok_cell ? 1 : 0) << ','
                              << (col_ok_cell ? 1 : 0) << '\n';
                }

                idx++;
            }
        }

        const double total = static_cast<double>(idx);
        const double sym_acc = total > 0 ? 100.0 * static_cast<double>(symbol_ok) / total : 0.0;
        const double pat_acc = total > 0 ? 100.0 * static_cast<double>(pattern_ok) / total : 0.0;
        const double col_acc = total > 0 ? 100.0 * static_cast<double>(color_ok) / total : 0.0;
        const double header_acc = header_total > 0 ? 100.0 * static_cast<double>(header_ok) / header_total : 0.0;
        const double edge_err = edge_total > 0 ? 100.0 * static_cast<double>(edge_symbol_err) / edge_total : 0.0;
        const double center_err = center_total > 0 ? 100.0 * static_cast<double>(center_symbol_err) / center_total : 0.0;

        camdrop::vision::PatternFrameRenderer renderer(dict);
        const cv::Mat src_render = renderer.RenderInterleavedSymbols(src_symbols);
        const cv::Mat cap_render = renderer.RenderInterleavedSymbols(cap_symbols);

        const fs::path out_dir = fs::path(opts.out_dir);
        cv::imwrite((out_dir / "source_deskewed.png").string(), source.deskewed);
        cv::imwrite((out_dir / "capture_deskewed.png").string(), capture.deskewed);
        cv::imwrite((out_dir / "decoded_source.png").string(), src_render);
        cv::imwrite((out_dir / "decoded_capture.png").string(), cap_render);
        cv::imwrite((out_dir / "diff_symbol.png").string(), diff_symbol);
        cv::imwrite((out_dir / "diff_pattern.png").string(), diff_pattern);
        cv::imwrite((out_dir / "diff_color.png").string(), diff_color);

        std::vector<std::pair<int, int>> row_rank;
        std::vector<std::pair<int, int>> col_rank;
        std::vector<std::pair<int, int>> pat_rank;
        struct ConfusionPair {
            int count = 0;
            int src = 0;
            int cap = 0;
        };
        std::vector<ConfusionPair> confusion_rank;
        row_rank.reserve(row_err.size());
        col_rank.reserve(col_err.size());
        pat_rank.reserve(pattern_total.size());
        confusion_rank.reserve(pattern_confusion.size());
        for (int r = 0; r < static_cast<int>(row_err.size()); ++r) row_rank.push_back({row_err[r], r});
        for (int c = 0; c < static_cast<int>(col_err.size()); ++c) col_rank.push_back({col_err[c], c});
        for (int p = 0; p < dict.size(); ++p) pat_rank.push_back({pattern_err[p], p});
        for (int src_pat = 0; src_pat < dict.size(); ++src_pat) {
            for (int cap_pat = 0; cap_pat < dict.size(); ++cap_pat) {
                const int count = pattern_confusion[src_pat * dict.size() + cap_pat];
                if (count <= 0) continue;
                confusion_rank.push_back({count, src_pat, cap_pat});
            }
        }
        std::sort(row_rank.begin(), row_rank.end(), std::greater<>());
        std::sort(col_rank.begin(), col_rank.end(), std::greater<>());
        std::sort(pat_rank.begin(), pat_rank.end(), std::greater<>());
        std::sort(confusion_rank.begin(), confusion_rank.end(), [](const ConfusionPair& a, const ConfusionPair& b) {
            if (a.count != b.count) return a.count > b.count;
            if (a.src != b.src) return a.src < b.src;
            return a.cap < b.cap;
        });

        std::cout << "symbol_acc=" << std::fixed << std::setprecision(3) << sym_acc
                  << "% pattern_acc=" << pat_acc
                  << "% color_acc=" << col_acc << "%\n";
        std::cout << "header_acc=" << header_acc << "% header_total=" << header_total << "\n";
        std::cout << "edge_symbol_err=" << edge_err << "% center_symbol_err=" << center_err << "%\n";

        std::cout << "top_rows:";
        for (int i = 0; i < 5 && i < static_cast<int>(row_rank.size()); ++i) {
            std::cout << " r" << row_rank[i].second << "=" << row_rank[i].first;
        }
        std::cout << "\n";
        std::cout << "top_cols:";
        for (int i = 0; i < 5 && i < static_cast<int>(col_rank.size()); ++i) {
            std::cout << " c" << col_rank[i].second << "=" << col_rank[i].first;
        }
        std::cout << "\n";
        std::cout << "top_pattern_error:";
        for (int i = 0; i < 8 && i < static_cast<int>(pat_rank.size()); ++i) {
            const int pat = pat_rank[i].second;
            if (pattern_err[pat] <= 0 || pattern_total[pat] <= 0) continue;
            const double err_rate = 100.0 * static_cast<double>(pattern_err[pat]) / static_cast<double>(pattern_total[pat]);
            std::ostringstream label;
            label << std::hex << std::setw(2) << std::setfill('0') << pat;
            std::cout << " p" << label.str() << "=" << pattern_err[pat] << "/" << pattern_total[pat]
                      << "(" << std::fixed << std::setprecision(1) << err_rate << "%)";
        }
        std::cout << "\n";
        std::cout << "top_pattern_confusion:";
        for (int i = 0; i < 10 && i < static_cast<int>(confusion_rank.size()); ++i) {
            std::ostringstream src_label;
            std::ostringstream cap_label;
            src_label << std::hex << std::setw(2) << std::setfill('0') << confusion_rank[i].src;
            cap_label << std::hex << std::setw(2) << std::setfill('0') << confusion_rank[i].cap;
            std::cout << " " << src_label.str() << "->" << cap_label.str() << "=" << confusion_rank[i].count;
        }
        std::cout << "\n";

        std::cout << "output_dir=" << fs::absolute(out_dir).string() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        print_usage();
        return 10;
    }
}
