#include <cstdlib>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "codec/Encoder.hpp"
#include "util/config.hpp"
#include "util/errors.hpp"
#include "util/parallel.hpp"
#include "vision/frame_renderer.hpp"
#include "vision/pattern_dict.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_file;
    std::string out_dir = "out_frames";
    std::string pattern_dir = "web/pattern_sets/best_v2";
    std::string video_out;
    std::string ffmpeg_bin;
    double acc = 0.95;
    int fps = 30;
    int repeat = 1;
    int ffmpeg_crf = 0;
    std::string ffmpeg_preset = "slow";
    std::string ffmpeg_pix_fmt = "yuv444p";
    int screen_width = 1920;
    int screen_height = 1080;
    int code_size = 0;
    double code_fit = 0.96;
    bool wrap_screen = true;
    int threads = 0;
};

std::string quote_arg(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out += ch;
    }
    out += "\"";
    return out;
}

void print_usage() {
    std::cout
        << "Usage: file_video_encoder --input <file> [--out-dir <dir>] [--patterns <dir>]\n"
        << "                          [--acc <0..1>] [--fps <n>] [--repeat <n>] [--threads <n>]\n"
        << "                          [--screen-width <n>] [--screen-height <n>]\n"
        << "                          [--code-size <n>] [--code-fit <0..1>] [--raw-frame-output]\n"
        << "                          [--video-out <file>] [--ffmpeg-bin <path>]\n"
        << "                          [--ffmpeg-crf <n>] [--ffmpeg-preset <name>] [--ffmpeg-pix-fmt <fmt>]\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts.input_file = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            opts.out_dir = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            opts.pattern_dir = argv[++i];
        } else if (arg == "--video-out" && i + 1 < argc) {
            opts.video_out = argv[++i];
        } else if (arg == "--ffmpeg-bin" && i + 1 < argc) {
            opts.ffmpeg_bin = argv[++i];
        } else if (arg == "--ffmpeg-crf" && i + 1 < argc) {
            opts.ffmpeg_crf = std::stoi(argv[++i]);
        } else if (arg == "--ffmpeg-preset" && i + 1 < argc) {
            opts.ffmpeg_preset = argv[++i];
        } else if (arg == "--ffmpeg-pix-fmt" && i + 1 < argc) {
            opts.ffmpeg_pix_fmt = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        } else if (arg == "--acc" && i + 1 < argc) {
            opts.acc = std::stod(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            opts.fps = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            opts.repeat = std::stoi(argv[++i]);
        } else if (arg == "--screen-width" && i + 1 < argc) {
            opts.screen_width = std::stoi(argv[++i]);
        } else if (arg == "--screen-height" && i + 1 < argc) {
            opts.screen_height = std::stoi(argv[++i]);
        } else if (arg == "--code-size" && i + 1 < argc) {
            opts.code_size = std::stoi(argv[++i]);
        } else if (arg == "--code-fit" && i + 1 < argc) {
            opts.code_fit = std::stod(argv[++i]);
        } else if (arg == "--raw-frame-output") {
            opts.wrap_screen = false;
        } else {
            throw ConfigInvalidError("Unknown argument: " + arg);
        }
    }
    if (opts.input_file.empty()) {
        throw ConfigInvalidError("Missing --input");
    }
    if (opts.fps <= 0) {
        throw ConfigRangeError("FPS must be > 0, got " + std::to_string(opts.fps));
    }
    if (opts.repeat <= 0) {
        throw ConfigRangeError("Repeat must be > 0, got " + std::to_string(opts.repeat));
    }
    if (!(opts.acc > 0.0 && opts.acc <= 1.0)) {
        throw ConfigRangeError("Accuracy must be in (0, 1], got " + std::to_string(opts.acc));
    }
    if (opts.screen_width <= 0 || opts.screen_height <= 0) {
        throw ConfigRangeError("Screen size must be > 0");
    }
    if (!(opts.code_fit > 0.0 && opts.code_fit <= 1.0)) {
        throw ConfigRangeError("Code-fit must be in (0, 1], got " + std::to_string(opts.code_fit));
    }
    return opts;
}

cv::Mat compose_screen_frame(const cv::Mat& code_image, const Options& opts) {
    if (!opts.wrap_screen) {
        return code_image;
    }
    const int screen_w = opts.screen_width;
    const int screen_h = opts.screen_height;
    const int max_size = std::min(screen_w, screen_h);
    int code_size = opts.code_size > 0
      ? opts.code_size
      : static_cast<int>(std::lround(static_cast<double>(max_size) * opts.code_fit));
    code_size = std::max(64, std::min(code_size, max_size));

    cv::Mat screen(screen_h, screen_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat scaled;
    if (code_image.cols != code_size || code_image.rows != code_size) {
        cv::resize(code_image, scaled, cv::Size(code_size, code_size), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        scaled = code_image;
    }
    const int x = (screen_w - code_size) / 2;
    const int y = (screen_h - code_size) / 2;
    scaled.copyTo(screen(cv::Rect(x, y, code_size, code_size)));
    return screen;
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

int build_video_with_ffmpeg(const Options& opts, const fs::path& out_dir) {
    const std::string input_glob = (out_dir / "frame_%06d.png").string();
    std::ostringstream cmd;
    cmd << quote_arg(opts.ffmpeg_bin)
        << " -y -framerate " << opts.fps
        << " -i " << quote_arg(input_glob)
        << " -c:v libx264"
        << " -preset " << quote_arg(opts.ffmpeg_preset)
        << " -crf " << opts.ffmpeg_crf
        << " -pix_fmt " << quote_arg(opts.ffmpeg_pix_fmt)
        << " -tune stillimage"
        << " " << quote_arg(opts.video_out);
    std::cout << "ffmpeg: " << cmd.str() << '\n';
#ifdef _WIN32
    std::vector<std::string> args = {
        opts.ffmpeg_bin,
        "-y",
        "-framerate",
        std::to_string(opts.fps),
        "-i",
        input_glob,
        "-c:v",
        "libx264",
        "-preset",
        opts.ffmpeg_preset,
        "-crf",
        std::to_string(opts.ffmpeg_crf),
        "-pix_fmt",
        opts.ffmpeg_pix_fmt,
        "-tune",
        "stillimage",
        opts.video_out
    };
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    const int rc = _spawnv(_P_WAIT, args[0].c_str(), argv.data());
    return rc;
#else
    return std::system(cmd.str().c_str());
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.ffmpeg_bin.empty()) {
            opts.ffmpeg_bin = find_embedded_ffmpeg(argv[0]).string();
        }
        Config::auto_config(opts.acc);

        const fs::path out_dir = fs::absolute(opts.out_dir);
        fs::create_directories(out_dir);

        const camdrop::vision::PatternDictionary dict =
            camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir);
        camdrop::vision::PatternFrameRenderer renderer(dict);

        Encoder encoder(opts.input_file);
        if (!encoder.is_valid()) {
            throw EncoderInitError("Encoder initialization failed");
        }

        const uint32_t packet_count = encoder.packet_count_recommended();
        const uint32_t logical_frames = (packet_count + Config::FOUNTAIN_PACKETS_PER_FRAME - 1)
                                      / Config::FOUNTAIN_PACKETS_PER_FRAME;

        std::cout << "input: " << fs::absolute(opts.input_file).string() << '\n';
        std::cout << "patterns: " << fs::absolute(opts.pattern_dir).string() << '\n';
        std::cout << "out_dir: " << out_dir.string() << '\n';
        std::cout << "packet_count_recommended: " << packet_count << '\n';
        std::cout << "logical_frames: " << logical_frames << '\n';
        std::cout << "written_frames: " << static_cast<uint64_t>(logical_frames) * static_cast<uint64_t>(opts.repeat) << '\n';
        std::cout << "frame_payload_bytes: " << Config::PACKET_CAPACITY << '\n';
        if (opts.wrap_screen) {
            const int max_size = std::min(opts.screen_width, opts.screen_height);
            const int code_size = opts.code_size > 0
              ? std::min(opts.code_size, max_size)
              : static_cast<int>(std::lround(static_cast<double>(max_size) * opts.code_fit));
            std::cout << "screen_frame: " << opts.screen_width << 'x' << opts.screen_height << '\n';
            std::cout << "screen_code_size: " << code_size << '\n';
        } else {
            std::cout << "screen_frame: raw-code-frame" << '\n';
        }

        std::vector<std::vector<uint8_t>> logical_payloads;
        logical_payloads.reserve(logical_frames);
        for (uint32_t logical = 0; logical < logical_frames; ++logical) {
            const std::vector<uint8_t> frame_bytes = encoder.get_packet();
            if (frame_bytes.size() != Config::PACKET_CAPACITY) {
                throw EncoderRuntimeError("Encoder returned unexpected frame size: " +
                                         std::to_string(frame_bytes.size()) + " != " +
                                         std::to_string(Config::PACKET_CAPACITY));
            }
            logical_payloads.push_back(frame_bytes);
        }

        struct FrameJob {
            uint32_t logical_index = 0;
            uint32_t frame_index = 0;
        };
        std::vector<FrameJob> jobs;
        jobs.reserve(static_cast<size_t>(logical_frames) * static_cast<size_t>(opts.repeat));
        uint32_t frame_index = 0;
        for (uint32_t logical = 0; logical < logical_frames; ++logical) {
            for (int rep = 0; rep < opts.repeat; ++rep) {
                jobs.push_back({logical, frame_index++});
            }
        }

        const size_t threads = camdrop::util::resolve_thread_count(opts.threads);
        std::mutex err_mu;
        std::string first_error;
        std::atomic<bool> failed{false};

        auto render_job = [&](camdrop::vision::PatternFrameRenderer& local_renderer, size_t i) {
            if (failed.load(std::memory_order_relaxed)) return;
            const auto& job = jobs[i];
            try {
                const cv::Mat image = compose_screen_frame(
                    local_renderer.Render(logical_payloads[job.logical_index]), opts);
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << job.frame_index << ".png";
                const fs::path out_path = out_dir / name.str();
                if (!cv::imwrite(out_path.string(), image)) {
                    throw ImageWriteError("Failed to write image: " + out_path.string());
                }
            } catch (const std::exception& ex) {
                {
                    std::lock_guard<std::mutex> lock(err_mu);
                    if (first_error.empty()) first_error = ex.what();
                }
                failed.store(true, std::memory_order_relaxed);
            }
        };

        if (threads <= 1) {
            for (size_t i = 0; i < jobs.size(); ++i) {
                render_job(renderer, i);
                if ((i + 1) % static_cast<size_t>(opts.repeat) == 0) {
                    const size_t logical_done = (i + 1) / static_cast<size_t>(opts.repeat);
                    std::cout << "\rgenerated logical frame " << logical_done << "/" << logical_frames << std::flush;
                }
            }
            std::cout << '\n';
        } else {
            std::cout << "rendering " << jobs.size() << " frames with " << threads << " threads...\n";
            camdrop::util::parallel_for_with_state(
                jobs.size(),
                threads,
                [&]() { return camdrop::vision::PatternFrameRenderer(dict); },
                render_job);
        }

        if (failed.load()) {
            throw ImageWriteError(first_error.empty() ? "render failed" : first_error);
        }

        if (!opts.video_out.empty()) {
            const int rc = build_video_with_ffmpeg(opts, out_dir);
            if (rc != 0) {
                std::cerr << "ffmpeg failed with exit code " << rc << '\n';
                return 3;
            }
            std::cout << "video: " << fs::absolute(opts.video_out).string() << '\n';
        }

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
