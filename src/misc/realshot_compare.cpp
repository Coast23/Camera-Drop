#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <string.h>   // _stricmp
#else
#  include <strings.h>  // strcasecmp
#  define _stricmp strcasecmp
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util/parallel.hpp"
#include "vision/color_cnn.hpp"
#include "vision/frame_pipeline.hpp"
#include "vision/pattern_cnn.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string source_dir;
    std::string capture_dir;
    std::string model_path = "web/model/best_dynamic.onnx";
    std::string pattern_dir = "pattern_finder/best_v2";
    std::string pattern_cnn_model_path;
    std::string color_cnn_model_path;
    std::string dump_source_deskew_dir;
    std::string dump_capture_deskew_dir;
    std::string dump_source_payload_csv;
    std::string dump_capture_payload_csv;
    bool capture_deskewed = false;
    bool source_deskewed = false;
    bool patch_track = false;
    int threads = 0;
    int ort_threads = 1;
};

struct FrameData {
    std::string name;
    std::vector<uint8_t> payload_symbols;
    double blur_score = 0.0;
    std::string localize_source;
    bool localized = false;
};

struct MatchStats {
    double symbol_acc = 0.0;
    double pattern_acc = 0.0;
    double color_acc = 0.0;
    size_t best_index = 0;
};

void apply_pattern_cnn_override(camdrop::vision::RecognizeResult& recognize,
                                const cv::Mat& deskewed_image,
                                camdrop::vision::PatternCnnClassifier* classifier) {
    if (!classifier || !recognize.ok || recognize.payload_symbols.empty()) {
        return;
    }
    const std::vector<uint8_t> payload_patterns = classifier->PredictPayloadPatterns(deskewed_image);
    if (payload_patterns.size() != recognize.payload_symbols.size()) {
        return;
    }
    for (size_t i = 0; i < recognize.payload_symbols.size(); ++i) {
        recognize.payload_symbols[i] =
            static_cast<uint8_t>((recognize.payload_symbols[i] & 0x30U) | (payload_patterns[i] & 0x0FU));
    }
}

void apply_color_cnn_override(camdrop::vision::RecognizeResult& recognize,
                              const cv::Mat& deskewed_image,
                              camdrop::vision::ColorCnnClassifier* classifier) {
    if (!classifier || !recognize.ok || recognize.payload_symbols.empty()) {
        return;
    }
    const std::vector<uint8_t> payload_colors = classifier->PredictPayloadColors(deskewed_image);
    if (payload_colors.size() != recognize.payload_symbols.size()) {
        return;
    }
    const uint8_t pattern_mask = recognize.pattern_bits > 0
        ? static_cast<uint8_t>((1U << recognize.pattern_bits) - 1U)
        : 0x0FU;
    for (size_t i = 0; i < recognize.payload_symbols.size(); ++i) {
        recognize.payload_symbols[i] =
            static_cast<uint8_t>(((payload_colors[i] & 0x03U) << recognize.pattern_bits)
                                 | (recognize.payload_symbols[i] & pattern_mask));
    }
}

void print_usage() {
    std::cout
        << "Usage: realshot_compare --source-dir <dir> --capture-dir <dir>\n"
        << "                       [--model <onnx>] [--patterns <dir>]\n"
        << "                       [--pattern-cnn-model <onnx>]\n"
        << "                       [--color-cnn-model <onnx>]\n"
        << "                       [--dump-source-deskew-dir <dir>] [--dump-capture-deskew-dir <dir>]\n"
        << "                       [--dump-source-payload-csv <csv>] [--dump-capture-payload-csv <csv>]\n"
        << "                       [--capture-deskewed] [--source-deskewed] [--no-patch-track]\n"
        << "                       [--threads <n>] [--ort-threads <n>]\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--source-dir" && i + 1 < argc) {
            opts.source_dir = argv[++i];
        } else if (arg == "--capture-dir" && i + 1 < argc) {
            opts.capture_dir = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            opts.pattern_dir = argv[++i];
        } else if (arg == "--pattern-cnn-model" && i + 1 < argc) {
            opts.pattern_cnn_model_path = argv[++i];
        } else if (arg == "--color-cnn-model" && i + 1 < argc) {
            opts.color_cnn_model_path = argv[++i];
        } else if (arg == "--dump-source-deskew-dir" && i + 1 < argc) {
            opts.dump_source_deskew_dir = argv[++i];
        } else if (arg == "--dump-capture-deskew-dir" && i + 1 < argc) {
            opts.dump_capture_deskew_dir = argv[++i];
        } else if (arg == "--dump-source-payload-csv" && i + 1 < argc) {
            opts.dump_source_payload_csv = argv[++i];
        } else if (arg == "--dump-capture-payload-csv" && i + 1 < argc) {
            opts.dump_capture_payload_csv = argv[++i];
        } else if (arg == "--capture-deskewed") {
            opts.capture_deskewed = true;
        } else if (arg == "--source-deskewed") {
            opts.source_deskewed = true;
        } else if (arg == "--no-patch-track") {
            opts.patch_track = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        } else if (arg == "--ort-threads" && i + 1 < argc) {
            opts.ort_threads = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.source_dir.empty() || opts.capture_dir.empty()) {
        throw std::runtime_error("missing --source-dir or --capture-dir");
    }
    return opts;
}

std::vector<fs::path> collect_images(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (_stricmp(ext.c_str(), ".png") == 0
            || _stricmp(ext.c_str(), ".jpg") == 0
            || _stricmp(ext.c_str(), ".jpeg") == 0) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

double measure_blur_score(const cv::Mat& img, double margin_ratio = 0.08, int sample_n = 48) {
    if (img.empty()) return 0.0;
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

void ensure_dir(const fs::path& dir) {
    if (dir.empty()) return;
    fs::create_directories(dir);
}

std::string join_symbols_hex(const std::vector<uint8_t>& symbols) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (i) oss << ' ';
        oss << std::setw(2) << static_cast<int>(symbols[i]);
    }
    return oss.str();
}

FrameData process_file(const fs::path& path,
                       camdrop::vision::FramePipeline* pipeline,
                       camdrop::vision::PatternRecognizer* recognizer,
                       camdrop::vision::PatternCnnClassifier* pattern_cnn,
                       camdrop::vision::ColorCnnClassifier* color_cnn,
                       bool deskewed_input,
                       const fs::path& dump_deskew_dir) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("failed to load image: " + path.string());
    }

    FrameData out;
    out.name = path.filename().string();
    if (deskewed_input) {
        if (!recognizer) {
            throw std::runtime_error("recognizer is required for deskewed input");
        }
        auto decoded = recognizer->Decode(image);
        apply_pattern_cnn_override(decoded, image, pattern_cnn);
        apply_color_cnn_override(decoded, image, color_cnn);
        if (!decoded.ok) {
            return out;
        }
        out.payload_symbols = decoded.payload_symbols;
        out.blur_score = measure_blur_score(image);
        if (!dump_deskew_dir.empty()) {
            ensure_dir(dump_deskew_dir);
            cv::imwrite((dump_deskew_dir / path.filename()).string(), image);
        }
        return out;
    }

    if (!pipeline) {
        throw std::runtime_error("pipeline is required for non-deskewed input");
    }
    const auto result = pipeline->Process(image);
    if (!result.localized) {
        return out;
    }
    out.localized = true;
    out.localize_source = result.localize.source;
    if (!result.recognized) {
        return out;
    }
    out.payload_symbols = result.recognize.payload_symbols;
    out.blur_score = measure_blur_score(result.deskewed_image);
    if (!dump_deskew_dir.empty()) {
        ensure_dir(dump_deskew_dir);
        cv::imwrite((dump_deskew_dir / path.filename()).string(), result.deskewed_image);
    }
    return out;
}

std::vector<FrameData> load_frames(const std::vector<fs::path>& files,
                                   const std::string& model_path,
                                   const std::string& pattern_dir,
                                   const std::string& pattern_cnn_model_path,
                                   const std::string& color_cnn_model_path,
                                   bool apply_model_overrides,
                                   bool deskewed_input,
                                   const fs::path& dump_deskew_dir,
                                   size_t threads,
                                   int ort_threads) {
    struct Worker {
        std::optional<camdrop::vision::FramePipeline> pipeline;
        std::optional<camdrop::vision::PatternRecognizer> recognizer;
        std::unique_ptr<camdrop::vision::PatternCnnClassifier> pattern_cnn;
        std::unique_ptr<camdrop::vision::ColorCnnClassifier> color_cnn;
    };

    std::vector<FrameData> out(files.size());
    std::atomic<size_t> done{0};
    std::mutex log_mu;

    camdrop::util::parallel_for_with_state(
        files.size(),
        threads,
        [&]() {
            Worker w;
            if (deskewed_input) {
                w.recognizer.emplace(camdrop::vision::PatternDictionary::LoadFromDirectory(pattern_dir));
                if (apply_model_overrides && !pattern_cnn_model_path.empty()) {
                    camdrop::vision::PatternCnnOptions options;
                    options.ort_threads = std::max(1, ort_threads);
                    w.pattern_cnn = std::make_unique<camdrop::vision::PatternCnnClassifier>(
                        pattern_cnn_model_path,
                        options);
                }
                if (apply_model_overrides && !color_cnn_model_path.empty()) {
                    camdrop::vision::ColorCnnOptions options;
                    options.ort_threads = std::max(1, ort_threads);
                    w.color_cnn = std::make_unique<camdrop::vision::ColorCnnClassifier>(
                        color_cnn_model_path,
                        options);
                }
            } else {
                camdrop::vision::FramePipelineConfig cfg;
                cfg.model_path = model_path;
                cfg.pattern_dir = pattern_dir;
                if (apply_model_overrides) {
                    cfg.pattern_cnn_model_path = pattern_cnn_model_path;
                    cfg.color_cnn_model_path = color_cnn_model_path;
                }
                cfg.patch_track_enabled = false;
                cfg.localizer_options.ort_threads = std::max(1, ort_threads);
                w.pipeline.emplace(cfg);
            }
            return w;
        },
        [&](Worker& w, size_t i) {
            out[i] = process_file(
                files[i],
                w.pipeline ? &(*w.pipeline) : nullptr,
                w.recognizer ? &(*w.recognizer) : nullptr,
                w.pattern_cnn.get(),
                w.color_cnn.get(),
                deskewed_input,
                dump_deskew_dir);
            const size_t cur = ++done;
            if (cur == files.size() || (cur % 25 == 0)) {
                std::lock_guard<std::mutex> lock(log_mu);
                std::cout << "\rprocessed " << cur << "/" << files.size() << std::flush;
            }
        });
    std::cout << '\n';
    return out;
}

MatchStats best_match(const std::vector<uint8_t>& capture, const std::vector<FrameData>& source, int pattern_bits) {
    MatchStats best;
    double best_symbol = -1.0;
    for (size_t i = 0; i < source.size(); ++i) {
        const auto& src = source[i].payload_symbols;
        if (src.empty() || src.size() != capture.size()) {
            continue;
        }
        size_t sym_ok = 0;
        size_t pat_ok = 0;
        size_t col_ok = 0;
        for (size_t k = 0; k < capture.size(); ++k) {
            const uint8_t a = capture[k];
            const uint8_t b = src[k];
            sym_ok += (a == b) ? 1 : 0;
            pat_ok += ((a & ((1 << pattern_bits) - 1)) == (b & ((1 << pattern_bits) - 1))) ? 1 : 0;
            col_ok += ((a >> pattern_bits) == (b >> pattern_bits)) ? 1 : 0;
        }
        const double symbol_acc = 100.0 * static_cast<double>(sym_ok) / static_cast<double>(capture.size());
        if (symbol_acc > best_symbol) {
            best_symbol = symbol_acc;
            best.best_index = i;
            best.symbol_acc = symbol_acc;
            best.pattern_acc = 100.0 * static_cast<double>(pat_ok) / static_cast<double>(capture.size());
            best.color_acc = 100.0 * static_cast<double>(col_ok) / static_cast<double>(capture.size());
        }
    }
    return best;
}

double mean_of(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median_of(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) == 0U) {
        return (values[mid - 1] + values[mid]) * 0.5;
    }
    return values[mid];
}

bool starts_with(const std::string& value, const char* prefix) {
    const size_t value_size = value.size();
    const size_t prefix_size = std::char_traits<char>::length(prefix);
    if (value_size < prefix_size) {
        return false;
    }
    return value.compare(0, prefix_size, prefix) == 0;
}

struct LocalizeStats {
    uint64_t total = 0;
    uint64_t patch = 0;
    uint64_t yolo = 0;
    uint64_t other = 0;
};

void update_localize_stats(const FrameData& frame, LocalizeStats& stats) {
    if (!frame.localized) {
        return;
    }
    ++stats.total;
    if (frame.localize_source == "patch-track") {
        ++stats.patch;
    } else if (starts_with(frame.localize_source, "yolo")) {
        ++stats.yolo;
    } else {
        ++stats.other;
    }
}

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

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.model_path == "web/model/best_dynamic.onnx") {
            opts.model_path = default_model_path(argv[0]).string();
        }
        if (opts.pattern_dir == "pattern_finder/best_v2") {
            opts.pattern_dir = default_pattern_dir(argv[0]).string();
        }
        if (opts.pattern_cnn_model_path.empty()) {
            opts.pattern_cnn_model_path = default_pattern_cnn_model_path(argv[0]).string();
        }
        ensure_dir(opts.dump_source_deskew_dir);
        ensure_dir(opts.dump_capture_deskew_dir);
        const size_t threads = camdrop::util::resolve_thread_count(opts.threads);
        const int ort_threads = std::max(1, opts.ort_threads);
        camdrop::vision::PatternDictionary dict =
            camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir);
        std::cout << "patch_track: " << (opts.patch_track ? "on" : "off") << '\n';
        std::cout << "threads: " << threads << " ort_threads: " << ort_threads << '\n';

        const auto source_files = collect_images(opts.source_dir);
        const auto capture_files = collect_images(opts.capture_dir);
        if (source_files.empty() || capture_files.empty()) {
            throw std::runtime_error("no image files found");
        }

        std::cout << "loading source frames...\n";
        auto source_frames = load_frames(
            source_files,
            opts.model_path,
            opts.pattern_dir,
            opts.pattern_cnn_model_path,
            opts.color_cnn_model_path,
            false,
            opts.source_deskewed,
            opts.dump_source_deskew_dir,
            threads,
            ort_threads);
        std::cout << "loading capture frames...\n";
        auto capture_frames = load_frames(
            capture_files,
            opts.model_path,
            opts.pattern_dir,
            opts.pattern_cnn_model_path,
            opts.color_cnn_model_path,
            true,
            opts.capture_deskewed,
            opts.dump_capture_deskew_dir,
            threads,
            ort_threads);

        auto dump_payload_csv = [](const std::string& path, const std::vector<FrameData>& frames) {
            if (path.empty()) return;
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to open payload csv: " + path);
            }
            out << "frame_name,payload_symbol_count,payload_symbols_hex,blur_score,localized,localize_source\n";
            for (const auto& frame : frames) {
                out << frame.name << ','
                    << frame.payload_symbols.size() << ','
                    << '"' << join_symbols_hex(frame.payload_symbols) << '"' << ','
                    << std::fixed << std::setprecision(6) << frame.blur_score << ','
                    << (frame.localized ? 1 : 0) << ','
                    << frame.localize_source << '\n';
            }
        };
        dump_payload_csv(opts.dump_source_payload_csv, source_frames);
        dump_payload_csv(opts.dump_capture_payload_csv, capture_frames);

        std::vector<FrameData> valid_source;
        for (const auto& item : source_frames) {
            if (!item.payload_symbols.empty()) valid_source.push_back(item);
        }
        if (valid_source.empty()) {
            throw std::runtime_error("no valid source frames decoded");
        }

        std::vector<double> symbol_accs;
        std::vector<double> pattern_accs;
        std::vector<double> color_accs;
        std::vector<double> blur_scores;
        LocalizeStats capture_localize;
        size_t valid_capture = 0;

        const int pattern_bits = dict.pattern_bits();
        for (const auto& cap : capture_frames) {
            update_localize_stats(cap, capture_localize);
            if (cap.payload_symbols.empty()) continue;
            const MatchStats match = best_match(cap.payload_symbols, valid_source, pattern_bits);
            symbol_accs.push_back(match.symbol_acc);
            pattern_accs.push_back(match.pattern_acc);
            color_accs.push_back(match.color_acc);
            blur_scores.push_back(cap.blur_score);
            ++valid_capture;
            std::cout << cap.name
                      << " best=" << valid_source[match.best_index].name
                      << " sym=" << std::fixed << std::setprecision(3) << match.symbol_acc
                      << "% pat=" << match.pattern_acc
                      << "% col=" << match.color_acc
                      << "% blur=" << cap.blur_score << '\n';
        }

        std::cout << "source_total=" << source_files.size()
                  << " source_valid=" << valid_source.size() << '\n';
        std::cout << "capture_total=" << capture_files.size()
                  << " capture_valid=" << valid_capture << '\n';
        std::cout << "best_match_symbol_acc_mean=" << std::fixed << std::setprecision(3) << mean_of(symbol_accs) << "%\n";
        std::cout << "best_match_symbol_acc_median=" << median_of(symbol_accs) << "%\n";
        std::cout << "best_match_pattern_acc_mean=" << mean_of(pattern_accs) << "%\n";
        std::cout << "best_match_color_acc_mean=" << mean_of(color_accs) << "%\n";
        std::cout << "capture_blur_mean=" << mean_of(blur_scores) << '\n';
        if (!opts.capture_deskewed) {
            std::cout << "capture_localize_total=" << capture_localize.total
                      << " patch=" << capture_localize.patch
                      << " yolo=" << capture_localize.yolo
                      << " other=" << capture_localize.other
                      << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
