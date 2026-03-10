#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "codec/Decoder.hpp"
#include "util/config.hpp"
#include "vision/frame_pipeline.hpp"
#include "vision/pattern_dict.hpp"
#include "vision/recognizer.hpp"
#include "vision/visual_frame_codec.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_file = "decoded.bin";
    std::string model_path = "web/model/best_dynamic.onnx";
    std::string pattern_dir = "web/pattern_sets/best_v2";
    std::string dump_deskew_dir;
    double acc = 0.95;
    bool deskewed_input = false;
    bool quad_refine = false;
    bool patch_track = true;
};

struct Stats {
    uint64_t total_frames = 0;
    uint64_t localized = 0;
    uint64_t localized_patch = 0;
    uint64_t localized_yolo = 0;
    uint64_t localized_other = 0;
    uint64_t deskewed = 0;
    uint64_t recognized = 0;
    uint64_t accepted_frames = 0;
    uint64_t unique_frames = 0;
    uint64_t duplicate_frames = 0;
    uint64_t rs_blocks_ok = 0;
    uint64_t rs_blocks_fail = 0;
    uint64_t crc_ok = 0;
    uint64_t crc_fail = 0;
    uint64_t add_ok = 0;
    uint64_t add_complete = 0;
    uint64_t add_duplicate = 0;
    uint64_t add_file_mismatch = 0;
    uint64_t add_decode_error = 0;
};

uint64_t fnv1a64(const std::vector<uint8_t>& data) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t byte : data) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool is_image_ext(const fs::path& path) {
    const std::string ext = path.extension().string();
    std::string lower;
    lower.resize(ext.size());
    std::transform(ext.begin(), ext.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower == ".png" || lower == ".jpg" || lower == ".jpeg" || lower == ".bmp";
}

bool ends_with(const std::string& value, const char* suffix) {
    const size_t value_size = value.size();
    const size_t suffix_size = std::char_traits<char>::length(suffix);
    if (value_size < suffix_size) {
        return false;
    }
    return value.compare(value_size - suffix_size, suffix_size, suffix) == 0;
}

bool starts_with(const std::string& value, const char* prefix) {
    const size_t value_size = value.size();
    const size_t prefix_size = std::char_traits<char>::length(prefix);
    if (value_size < prefix_size) {
        return false;
    }
    return value.compare(0, prefix_size, prefix) == 0;
}

bool is_generated_debug_image(const fs::path& path) {
    const std::string stem = path.stem().string();
    return stem.size() >= 8 && (
        ends_with(stem, "_deskewed") ||
        ends_with(stem, "_detected") ||
        ends_with(stem, "_web_encoded") ||
        ends_with(stem, "_web_camera") ||
        ends_with(stem, "_deskew_grid"));
}

void print_usage() {
    std::cout
        << "Usage: file_video_decoder --input <video-or-dir> [--output <file>] [--acc <0..1>]\n"
        << "                          [--model <onnx>] [--patterns <dir>] [--deskewed-input]\n"
        << "                          [--quad-refine] [--dump-deskew <dir>] [--no-patch-track]\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            opts.pattern_dir = argv[++i];
        } else if (arg == "--dump-deskew" && i + 1 < argc) {
            opts.dump_deskew_dir = argv[++i];
        } else if (arg == "--acc" && i + 1 < argc) {
            opts.acc = std::stod(argv[++i]);
        } else if (arg == "--deskewed-input") {
            opts.deskewed_input = true;
        } else if (arg == "--quad-refine") {
            opts.quad_refine = true;
        } else if (arg == "--no-patch-track") {
            opts.patch_track = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.input_path.empty()) {
        throw std::runtime_error("missing --input");
    }
    if (!(opts.acc > 0.0 && opts.acc <= 1.0)) {
        throw std::runtime_error("acc must be in (0, 1]");
    }
    return opts;
}

std::vector<fs::path> collect_input_images(const fs::path& input, bool allow_generated_debug_images = false) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(input)) {
        if (entry.is_regular_file()
            && is_image_ext(entry.path())
            && (allow_generated_debug_images || !is_generated_debug_image(entry.path()))) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

template <typename HandleFrameFn>
void decode_image_list(const std::vector<fs::path>& files, HandleFrameFn&& handle_frame) {
    for (const auto& file : files) {
        const cv::Mat image = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "skip unreadable image: " << file.string() << '\n';
            continue;
        }
        if (!handle_frame(image, file.filename().string())) {
            break;
        }
    }
}

template <typename HandleFrameFn>
void decode_video_file(const fs::path& file, HandleFrameFn&& handle_frame) {
    cv::VideoCapture cap(file.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("failed to open video: " + file.string());
    }
    cv::Mat frame;
    uint64_t idx = 0;
    while (cap.read(frame)) {
        std::ostringstream name;
        name << "video_" << std::setfill('0') << std::setw(6) << idx++;
        if (!handle_frame(frame, name.str())) {
            break;
        }
    }
}

void maybe_dump_deskew(const cv::Mat& image, const std::string& name, const std::string& dump_dir) {
    if (dump_dir.empty() || image.empty()) {
        return;
    }
    fs::create_directories(dump_dir);
    const fs::path out = fs::path(dump_dir) / (name + "_deskewed.png");
    cv::imwrite(out.string(), image);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        Config::auto_config(opts.acc);
        {
            const fs::path out_path = fs::absolute(opts.output_file);
            if (!out_path.parent_path().empty()) {
                fs::create_directories(out_path.parent_path());
            }
        }

        std::cout << "input: " << fs::absolute(opts.input_path).string() << '\n';
        std::cout << "patterns: " << fs::absolute(opts.pattern_dir).string() << '\n';
        if (!opts.deskewed_input) {
            std::cout << "model: " << fs::absolute(opts.model_path).string() << '\n';
        }
        std::cout << "patch_track: " << (opts.patch_track ? "on" : "off") << '\n';

        Decoder decoder;
        Stats stats;
        std::unordered_set<uint64_t> seen_hashes;

        std::optional<camdrop::vision::FramePipeline> pipeline;
        std::optional<camdrop::vision::PatternRecognizer> recognizer;
        if (opts.deskewed_input) {
            recognizer.emplace(camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir));
        } else {
            camdrop::vision::FramePipelineConfig cfg;
            cfg.model_path = opts.model_path;
            cfg.pattern_dir = opts.pattern_dir;
            cfg.localizer_options.refine_anchor_quad = opts.quad_refine;
            cfg.patch_track_enabled = opts.patch_track;
            pipeline.emplace(cfg);
        }

        bool completed = false;
        uint64_t completed_at_frame = 0;

        auto handle_frame = [&](const cv::Mat& frame, const std::string& name) -> bool {
            ++stats.total_frames;
            try {
                camdrop::vision::RecognizeResult recognize;
                if (opts.deskewed_input) {
                    recognize = recognizer->Decode(frame);
                    stats.deskewed += recognize.ok ? 1 : 0;
                } else {
                    camdrop::vision::PipelineResult result = pipeline->Process(frame);
                    if (!result.localized) {
                        return true;
                    }
                    ++stats.localized;
                    if (!result.localize.source.empty()) {
                        if (result.localize.source == "patch-track") {
                            ++stats.localized_patch;
                        } else if (starts_with(result.localize.source, "yolo")) {
                            ++stats.localized_yolo;
                        } else {
                            ++stats.localized_other;
                        }
                    }
                    if (!result.deskewed) {
                        return true;
                    }
                    ++stats.deskewed;
                    maybe_dump_deskew(result.deskewed_image, name, opts.dump_deskew_dir);
                    if (!result.recognized) {
                        return true;
                    }
                    recognize = std::move(result.recognize);
                }

                if (!recognize.ok) {
                    return true;
                }
                ++stats.recognized;

                const std::vector<uint8_t> frame_bytes =
                    camdrop::vision::RecognizeResultToFrameBytes(recognize);
                const uint64_t hash = fnv1a64(frame_bytes);
                if (!seen_hashes.insert(hash).second) {
                    ++stats.duplicate_frames;
                    return true;
                }

                ++stats.unique_frames;
                Decoder::ProcessPacketStats packet_stats;
                const bool accepted = decoder.process_packet(frame_bytes, &packet_stats);
                stats.rs_blocks_ok += packet_stats.rs_blocks_ok;
                stats.rs_blocks_fail += packet_stats.rs_blocks_fail;
                stats.crc_ok += packet_stats.fountain_packets_crc_ok;
                stats.crc_fail += packet_stats.fountain_packets_crc_fail;
                stats.add_ok += packet_stats.fountain_blocks_added;
                stats.add_complete += packet_stats.fountain_blocks_completed;
                stats.add_duplicate += packet_stats.fountain_blocks_duplicate;
                stats.add_file_mismatch += packet_stats.fountain_blocks_file_mismatch;
                stats.add_decode_error += packet_stats.fountain_blocks_decode_error;
                if (accepted) {
                    ++stats.accepted_frames;
                }
                if (decoder.is_complete()) {
                    completed = decoder.save_to_file(opts.output_file);
                    completed_at_frame = stats.total_frames;
                    return false;
                }
            } catch (const std::exception& ex) {
                std::cerr << "frame " << name << " failed: " << ex.what() << '\n';
            }
            return true;
        };

        const fs::path input = fs::absolute(opts.input_path);
        if (fs::is_directory(input)) {
            const std::vector<fs::path> files = collect_input_images(input, opts.deskewed_input);
            if (files.empty()) {
                throw std::runtime_error("no images found in input dir");
            }
            decode_image_list(files, handle_frame);
        } else {
            if (opts.deskewed_input && is_image_ext(input)) {
                decode_image_list(std::vector<fs::path>{input}, handle_frame);
            } else if (is_image_ext(input)) {
                decode_image_list(std::vector<fs::path>{input}, handle_frame);
            } else {
                decode_video_file(input, handle_frame);
            }
        }

        std::cout << "total_frames=" << stats.total_frames
                  << " localized=" << stats.localized
                  << " localize_patch=" << stats.localized_patch
                  << " localize_yolo=" << stats.localized_yolo
                  << " localize_other=" << stats.localized_other
                  << " deskewed=" << stats.deskewed
                  << " recognized=" << stats.recognized
                  << " accepted_frames=" << stats.accepted_frames
                  << " unique=" << stats.unique_frames
                  << " duplicate=" << stats.duplicate_frames
                  << " rs_ok=" << stats.rs_blocks_ok
                  << " rs_fail=" << stats.rs_blocks_fail
                  << " crc_ok=" << stats.crc_ok
                  << " crc_fail=" << stats.crc_fail
                  << " add_ok=" << stats.add_ok
                  << " add_complete=" << stats.add_complete
                  << " add_dup=" << stats.add_duplicate
                  << " add_file_mismatch=" << stats.add_file_mismatch
                  << " add_decode_error=" << stats.add_decode_error
                  << '\n';

        if (!decoder.is_complete() || !completed) {
            std::cerr << "decode incomplete\n";
            return 2;
        }

        std::cout << "decode complete at frame " << completed_at_frame
                  << " -> " << fs::absolute(opts.output_file).string() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
