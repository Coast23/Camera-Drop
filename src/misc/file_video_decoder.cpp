#include <algorithm>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "codec/Decoder.hpp"
#include "util/config.hpp"
#include "util/errors.hpp"
#include "util/parallel.hpp"
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
    std::string pattern_dir = "pattern_finder/best_v2";
    std::string pattern_cnn_model_path;
    std::string dump_deskew_dir;
    std::string ffmpeg_bin;
    std::string ffmpeg_frames_dir;
    double acc = 0.95;
    bool deskewed_input = false;
    bool patch_track = true;
    int threads = 0;
    int ort_threads = 1;
};

struct Stats {
    uint64_t total_frames = 0;
    uint64_t localized = 0;
    uint64_t localized_patch = 0;
    uint64_t localized_yolo = 0;
    uint64_t localized_other = 0;
    uint64_t deskewed = 0;
    uint64_t recognize_attempts = 0;
    uint64_t recognized = 0;
    uint64_t accepted_frames = 0;
    uint64_t unique_frames = 0;
    uint64_t duplicate_frames = 0;
    uint64_t signature_duplicate_skips = 0;
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

struct RawFrame {
    uint64_t index = 0;
    std::string name;
    fs::path path;
    cv::Mat image;
    bool from_file = false;
};

struct DecodedFrame {
    uint64_t index = 0;
    std::string name;
    bool localized = false;
    std::string localize_source;
    bool deskewed = false;
    bool recognition_attempted = false;
    bool recognized = false;
    bool duplicate_skipped = false;
    bool has_deskew_signature = false;
    std::array<uint8_t, 256> deskew_signature{};
    std::vector<uint8_t> frame_bytes;
    std::string error;
};

uint64_t fnv1a64(const std::vector<uint8_t>& data) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t byte : data) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool compute_thumb_signature(const cv::Mat& image, std::array<uint8_t, 256>* out) {
    if (!out || image.empty()) {
        return false;
    }
    out->fill(0);
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat small;
    cv::resize(gray, small, cv::Size(16, 16), 0.0, 0.0, cv::INTER_AREA);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(small, mean, stddev);
    const double denom = std::max(1.0, stddev[0]);
    size_t idx = 0;
    for (int y = 0; y < small.rows; ++y) {
        for (int x = 0; x < small.cols; ++x) {
            const double normalized = (static_cast<double>(small.at<uint8_t>(y, x)) - mean[0]) / denom;
            const int quantized = std::clamp(static_cast<int>(std::lround(normalized * 4.0 + 16.0)), 0, 31);
            (*out)[idx++] = static_cast<uint8_t>(quantized);
        }
    }
    return true;
}

int thumb_signature_distance(const std::array<uint8_t, 256>& lhs,
                             const std::array<uint8_t, 256>& rhs) {
    int distance = 0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        distance += std::abs(static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]));
    }
    return distance;
}

bool contains_similar_signature(const std::vector<std::array<uint8_t, 256>>& seen_signatures,
                                const std::array<uint8_t, 256>& signature,
                                int max_distance) {
    return std::any_of(seen_signatures.begin(), seen_signatures.end(), [&](const auto& seen) {
        return thumb_signature_distance(seen, signature) <= max_distance;
    });
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

std::string quote_arg(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out += ch;
    }
    out += "\"";
    return out;
}

uint64_t current_pid() {
#ifdef _WIN32
    return static_cast<uint64_t>(GetCurrentProcessId());
#else
    return static_cast<uint64_t>(getpid());
#endif
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
        << "                          [--pattern-cnn-model <onnx>]\n"
        << "                          [--dump-deskew <dir>] [--no-patch-track]\n"
        << "                          [--threads <n>] [--ort-threads <n>]\n"
        << "                          [--ffmpeg-bin <path>] [--ffmpeg-frames <dir>]\n";
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
        } else if (arg == "--pattern-cnn-model" && i + 1 < argc) {
            opts.pattern_cnn_model_path = argv[++i];
        } else if (arg == "--dump-deskew" && i + 1 < argc) {
            opts.dump_deskew_dir = argv[++i];
        } else if (arg == "--ffmpeg-bin" && i + 1 < argc) {
            opts.ffmpeg_bin = argv[++i];
        } else if (arg == "--ffmpeg-frames" && i + 1 < argc) {
            opts.ffmpeg_frames_dir = argv[++i];
        } else if (arg == "--acc" && i + 1 < argc) {
            opts.acc = std::stod(argv[++i]);
        } else if (arg == "--deskewed-input") {
            opts.deskewed_input = true;
        } else if (arg == "--no-patch-track") {
            opts.patch_track = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        } else if (arg == "--ort-threads" && i + 1 < argc) {
            opts.ort_threads = std::stoi(argv[++i]);
        } else {
            throw ConfigInvalidError("Unknown argument: " + arg);
        }
    }
    if (opts.input_path.empty()) {
        throw ConfigInvalidError("Missing --input");
    }
    if (!(opts.acc > 0.0 && opts.acc <= 1.0)) {
        throw ConfigRangeError("Accuracy must be in (0, 1], got " + std::to_string(opts.acc));
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

void enqueue_image_list(const std::vector<fs::path>& files,
                        camdrop::util::BlockingQueue<RawFrame>& queue,
                        const std::atomic<bool>& stop_requested) {
    uint64_t idx = 0;
    for (const auto& file : files) {
        if (stop_requested.load(std::memory_order_relaxed) || queue.closed()) {
            break;
        }
        RawFrame item;
        item.index = idx++;
        item.name = file.filename().string();
        item.path = file;
        item.from_file = true;
        queue.push(std::move(item));
    }
}

void enqueue_video_file(const fs::path& file,
                        camdrop::util::BlockingQueue<RawFrame>& queue,
                        const std::atomic<bool>& stop_requested) {
    cv::VideoCapture cap(file.string());
    if (!cap.isOpened()) {
        throw FileOpenError("Failed to open video: " + file.string());
    }
    cv::Mat frame;
    uint64_t idx = 0;
    while (!stop_requested.load(std::memory_order_relaxed) && cap.read(frame)) {
        if (queue.closed()) {
            break;
        }
        std::ostringstream name;
        name << "video_" << std::setfill('0') << std::setw(6) << idx++;
        RawFrame item;
        item.index = idx - 1;
        item.name = name.str();
        item.image = frame.clone();
        item.from_file = false;
        queue.push(std::move(item));
    }
}

void maybe_dump_deskew(const cv::Mat& image, const std::string& name, const std::string& dump_dir) {
    if (dump_dir.empty() || image.empty()) {
        return;
    }
    fs::create_directories(dump_dir);
    const fs::path out = fs::path(dump_dir) / (name + "_deskewed.png");
    if (!cv::imwrite(out.string(), image)) {
        std::cerr << "Warning: Failed to write debug image: " << out.string() << '\n';
    }
}

fs::path find_embedded_ffmpeg(const fs::path& argv0) {
    fs::path dir = fs::absolute(argv0).parent_path();
    {
#ifdef _WIN32
        const fs::path local = dir / "ffmpeg.exe";
#else
        const fs::path local = dir / "ffmpeg";
#endif
        if (fs::exists(local)) return local;
    }
    for (int i = 0; i < 6; ++i) {
#ifdef _WIN32
        const fs::path bin1 = dir / "third_party" / "ffmpeg" / "bin" / "ffmpeg.exe";
        if (fs::exists(bin1)) return bin1;
        const fs::path bin2 = dir / "third_party" / "ffmpeg" / "ffmpeg.exe";
        if (fs::exists(bin2)) return bin2;
#else
        const fs::path bin1 = dir / "third_party" / "ffmpeg" / "bin" / "ffmpeg";
        if (fs::exists(bin1)) return bin1;
        const fs::path bin2 = dir / "third_party" / "ffmpeg" / "ffmpeg";
        if (fs::exists(bin2)) return bin2;
#endif
        if (!dir.has_parent_path()) break;
        dir = dir.parent_path();
    }
    return "ffmpeg";
}

int extract_video_frames(const std::string& ffmpeg_bin, const fs::path& input, const fs::path& out_dir) {
    fs::create_directories(out_dir);
    const std::string output_glob = (out_dir / "frame_%06d.png").string();
    std::ostringstream cmd;
    cmd << quote_arg(ffmpeg_bin)
        << " -y -i " << quote_arg(input.string())
        << " -fps_mode passthrough -pix_fmt rgb24 " << quote_arg(output_glob);
    std::cout << "ffmpeg: " << cmd.str() << '\n';
#ifdef _WIN32
    std::vector<std::string> args = {
        ffmpeg_bin,
        "-y",
        "-i",
        input.string(),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        output_glob
    };
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    return _spawnv(_P_WAIT, args[0].c_str(), argv.data());
#else
    return std::system(cmd.str().c_str());
#endif
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
        if (opts.ffmpeg_bin.empty()) {
            opts.ffmpeg_bin = find_embedded_ffmpeg(argv[0]).string();
        }
        if (opts.model_path == "web/model/best_dynamic.onnx") {
            opts.model_path = default_model_path(argv[0]).string();
        }
        if (opts.pattern_dir == "pattern_finder/best_v2") {
            opts.pattern_dir = default_pattern_dir(argv[0]).string();
        }
        if (opts.pattern_cnn_model_path.empty()) {
            opts.pattern_cnn_model_path = default_pattern_cnn_model_path(argv[0]).string();
        }
        Config::auto_config(opts.acc);
        {
            const fs::path out_path = fs::absolute(opts.output_file);
            if (!out_path.parent_path().empty()) {
                fs::create_directories(out_path.parent_path());
            }
        }

        const size_t thread_count = camdrop::util::resolve_thread_count(opts.threads);
        const bool patch_track_enabled = opts.patch_track && thread_count <= 1;
        const int ort_threads = std::max(1, opts.ort_threads);
        constexpr int kCodeSignatureMaxDistance = 72;

        std::cout << "input: " << fs::absolute(opts.input_path).string() << '\n';
        std::cout << "patterns: " << fs::absolute(opts.pattern_dir).string() << '\n';
        if (!opts.deskewed_input) {
            std::cout << "model: " << fs::absolute(opts.model_path).string() << '\n';
        }
        if (opts.patch_track && !patch_track_enabled) {
            std::cout << "patch_track: off (forced for parallel decode; use --threads 1 to enable)\n";
        } else {
            std::cout << "patch_track: " << (patch_track_enabled ? "on" : "off") << '\n';
        }
        std::cout << "threads: " << thread_count << " ort_threads: " << ort_threads << '\n';

        Decoder decoder;
        Stats stats;
        std::unordered_set<uint64_t> seen_hashes;
        std::vector<std::array<uint8_t, 256>> seen_code_signatures;
        std::mutex seen_code_signatures_mu;
        std::mutex log_mu;
        std::atomic<bool> stop_requested{false};

        camdrop::util::BlockingQueue<RawFrame> input_queue(2);
        camdrop::util::BlockingQueue<DecodedFrame> output_queue(thread_count * 2);
        std::atomic<int> active_workers{static_cast<int>(thread_count)};

        auto worker_fn = [&]() {
            std::optional<camdrop::vision::FramePipeline> pipeline;
            std::optional<camdrop::vision::PatternRecognizer> recognizer;
            if (opts.deskewed_input) {
                recognizer.emplace(camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir));
            } else {
                camdrop::vision::FramePipelineConfig cfg;
                cfg.model_path = opts.model_path;
                cfg.pattern_dir = opts.pattern_dir;
                cfg.pattern_cnn_model_path = opts.pattern_cnn_model_path;
                cfg.patch_track_enabled = patch_track_enabled;
                cfg.localizer_options.ort_threads = ort_threads;
                pipeline.emplace(cfg);
            }

            RawFrame item;
            while (input_queue.pop(item)) {
                if (stop_requested.load(std::memory_order_relaxed)) {
                    break;
                }
                DecodedFrame out;
                out.index = item.index;
                out.name = item.name;
                try {
                    cv::Mat image;
                    if (item.from_file) {
                        image = cv::imread(item.path.string(), cv::IMREAD_COLOR);
                        if (image.empty()) {
                            std::lock_guard<std::mutex> lock(log_mu);
                            std::cerr << "Skip unreadable image: " << item.path.string() << '\n';
                            continue;
                        }
                    } else {
                        image = std::move(item.image);
                        if (image.empty()) {
                            continue;
                        }
                    }

                    camdrop::vision::RecognizeResult recognize;
                    if (opts.deskewed_input) {
                        out.recognition_attempted = true;
                        recognize = recognizer->Decode(image);
                        out.deskewed = recognize.ok;
                    } else {
                        bool duplicate_skip = false;
                        camdrop::vision::PipelineResult result = pipeline->Process(
                            image,
                            [&](const cv::Mat& deskewed_image) {
                                std::array<uint8_t, 256> signature{};
                                if (!compute_thumb_signature(deskewed_image, &signature)) {
                                    return false;
                                }
                                std::lock_guard<std::mutex> lock(seen_code_signatures_mu);
                                if (!contains_similar_signature(seen_code_signatures,
                                                                signature,
                                                                kCodeSignatureMaxDistance)) {
                                    return false;
                                }
                                out.duplicate_skipped = true;
                                out.has_deskew_signature = true;
                                out.deskew_signature = signature;
                                duplicate_skip = true;
                                return true;
                            });
                        if (!result.localized) {
                            output_queue.push(std::move(out));
                            continue;
                        }
                        out.localized = true;
                        out.localize_source = result.localize.source;
                        if (!result.deskewed) {
                            output_queue.push(std::move(out));
                            continue;
                        }
                        out.deskewed = true;
                        maybe_dump_deskew(result.deskewed_image, out.name, opts.dump_deskew_dir);
                        out.has_deskew_signature =
                            compute_thumb_signature(result.deskewed_image, &out.deskew_signature);
                        if (duplicate_skip) {
                            output_queue.push(std::move(out));
                            continue;
                        }
                        out.recognition_attempted = true;
                        if (!result.recognized) {
                            output_queue.push(std::move(out));
                            continue;
                        }
                        recognize = std::move(result.recognize);
                    }

                    if (recognize.ok) {
                        out.recognized = true;
                        out.frame_bytes = camdrop::vision::RecognizeResultToFrameBytes(recognize);
                    }
                } catch (const std::exception& ex) {
                    out.error = ex.what();
                }
                output_queue.push(std::move(out));
            }

            if (active_workers.fetch_sub(1) == 1) {
                output_queue.close();
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i) {
            workers.emplace_back(worker_fn);
        }

        std::atomic<bool> producer_failed{false};
        std::string producer_error;
        std::optional<fs::path> temp_frames_dir;
        std::thread producer([&]() {
            try {
                const fs::path input = fs::absolute(opts.input_path);
                if (fs::is_directory(input)) {
                    const std::vector<fs::path> files = collect_input_images(input, opts.deskewed_input);
                    if (files.empty()) {
                        producer_error = "No images found in input dir: " + input.string();
                        producer_failed.store(true);
                        input_queue.close();
                        return;
                    }
                    enqueue_image_list(files, input_queue, stop_requested);
                } else {
                    if (is_image_ext(input)) {
                        enqueue_image_list(std::vector<fs::path>{input}, input_queue, stop_requested);
                    } else {
                        try {
                            enqueue_video_file(input, input_queue, stop_requested);
                        } catch (const std::exception&) {
                            fs::path frames_dir;
                            if (!opts.ffmpeg_frames_dir.empty()) {
                                frames_dir = fs::absolute(opts.ffmpeg_frames_dir);
                            } else {
                                frames_dir = fs::temp_directory_path() /
                                             ("camdrop_frames_" + std::to_string(current_pid()));
                                temp_frames_dir = frames_dir;
                            }
                            const int rc = extract_video_frames(opts.ffmpeg_bin, input, frames_dir);
                            if (rc != 0) {
                                producer_error = "ffmpeg extract failed with exit code " + std::to_string(rc);
                                producer_failed.store(true);
                                input_queue.close();
                                return;
                            }
                            const std::vector<fs::path> files = collect_input_images(frames_dir, opts.deskewed_input);
                            if (files.empty()) {
                                producer_error = "ffmpeg extract produced no frames: " + frames_dir.string();
                                producer_failed.store(true);
                                input_queue.close();
                                return;
                            }
                            enqueue_image_list(files, input_queue, stop_requested);
                        }
                    }
                }
            } catch (const std::exception& ex) {
                producer_error = ex.what();
                producer_failed.store(true);
            }
            input_queue.close();
        });

        bool completed = false;
        uint64_t completed_at_frame = 0;
        DecodedFrame out;
        while (output_queue.pop(out)) {
            ++stats.total_frames;
            if (!out.error.empty()) {
                std::lock_guard<std::mutex> lock(log_mu);
                std::cerr << "Frame " << out.name << " failed: " << out.error << '\n';
                continue;
            }
            if (out.localized) {
                ++stats.localized;
                if (!out.localize_source.empty()) {
                    if (out.localize_source == "patch-track") {
                        ++stats.localized_patch;
                    } else if (starts_with(out.localize_source, "yolo")) {
                        ++stats.localized_yolo;
                    } else {
                        ++stats.localized_other;
                    }
                }
            }
            if (out.deskewed) {
                ++stats.deskewed;
            }
            if (out.duplicate_skipped) {
                ++stats.signature_duplicate_skips;
                continue;
            }
            if (out.recognition_attempted) {
                ++stats.recognize_attempts;
            }
            if (!out.recognized) {
                continue;
            }
            ++stats.recognized;
            if (out.has_deskew_signature) {
                std::lock_guard<std::mutex> lock(seen_code_signatures_mu);
                if (!contains_similar_signature(seen_code_signatures,
                                                out.deskew_signature,
                                                kCodeSignatureMaxDistance)) {
                    seen_code_signatures.push_back(out.deskew_signature);
                }
            }

            const uint64_t hash = fnv1a64(out.frame_bytes);
            if (!seen_hashes.insert(hash).second) {
                ++stats.duplicate_frames;
                continue;
            }
            ++stats.unique_frames;

            Decoder::ProcessPacketStats packet_stats;
            const bool accepted = decoder.process_packet(out.frame_bytes, &packet_stats);
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
            if (!completed && decoder.is_complete()) {
                try {
                    decoder.save_to_file(opts.output_file);
                } catch (const DecoderRuntimeError& e) {
                    std::cerr << "Frame " << out.name << " decode failed: " << e.what() << '\n';
                    continue;
                }
                completed = true;
                completed_at_frame = stats.total_frames;
                stop_requested.store(true, std::memory_order_relaxed);
                input_queue.close();
            }
        }

        producer.join();
        for (auto& th : workers) {
            th.join();
        }

        if (producer_failed.load()) {
            throw FileNotFoundError(producer_error);
        }
        if (temp_frames_dir.has_value()) {
            std::error_code ec;
            fs::remove_all(*temp_frames_dir, ec);
        }

        std::cout << "total_frames=" << stats.total_frames
                  << " localized=" << stats.localized
                  << " localize_patch=" << stats.localized_patch
                  << " localize_yolo=" << stats.localized_yolo
                  << " localize_other=" << stats.localized_other
                  << " deskewed=" << stats.deskewed
                  << " recognize_attempts=" << stats.recognize_attempts
                  << " recognized=" << stats.recognized
                  << " accepted_frames=" << stats.accepted_frames
                  << " unique=" << stats.unique_frames
                  << " duplicate=" << stats.duplicate_frames
                  << " signature_skip=" << stats.signature_duplicate_skips
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
    } catch (const CameraDropError& ex) {
        std::cerr << "CameraDrop Error: " << ex.what() << '\n';
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
