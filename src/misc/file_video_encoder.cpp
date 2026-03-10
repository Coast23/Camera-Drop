#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "codec/Encoder.hpp"
#include "util/config.hpp"
#include "vision/frame_renderer.hpp"
#include "vision/pattern_dict.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_file;
    std::string out_dir = "out_frames";
    std::string pattern_dir = "web/pattern_sets/best_v2";
    std::string video_out;
    std::string ffmpeg_bin = "ffmpeg";
    double acc = 0.95;
    int fps = 30;
    int repeat = 1;
    int screen_width = 1920;
    int screen_height = 1080;
    int code_size = 0;
    double code_fit = 0.96;
    bool wrap_screen = true;
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
        << "                          [--acc <0..1>] [--fps <n>] [--repeat <n>]\n"
        << "                          [--screen-width <n>] [--screen-height <n>]\n"
        << "                          [--code-size <n>] [--code-fit <0..1>] [--raw-frame-output]\n"
        << "                          [--video-out <file>] [--ffmpeg-bin <path>]\n";
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
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.input_file.empty()) {
        throw std::runtime_error("missing --input");
    }
    if (opts.fps <= 0) {
        throw std::runtime_error("fps must be > 0");
    }
    if (opts.repeat <= 0) {
        throw std::runtime_error("repeat must be > 0");
    }
    if (!(opts.acc > 0.0 && opts.acc <= 1.0)) {
        throw std::runtime_error("acc must be in (0, 1]");
    }
    if (opts.screen_width <= 0 || opts.screen_height <= 0) {
        throw std::runtime_error("screen size must be > 0");
    }
    if (!(opts.code_fit > 0.0 && opts.code_fit <= 1.0)) {
        throw std::runtime_error("code-fit must be in (0, 1]");
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

int build_video_with_ffmpeg(const Options& opts, const fs::path& out_dir) {
    const std::string input_glob = (out_dir / "frame_%06d.png").string();
    std::ostringstream cmd;
    cmd << quote_arg(opts.ffmpeg_bin)
        << " -y -framerate " << opts.fps
        << " -i " << quote_arg(input_glob)
        << " -c:v libx264 -pix_fmt yuv420p "
        << quote_arg(opts.video_out);
    std::cout << "ffmpeg: " << cmd.str() << '\n';
    return std::system(cmd.str().c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        Config::auto_config(opts.acc);

        const fs::path out_dir = fs::absolute(opts.out_dir);
        fs::create_directories(out_dir);

        const camdrop::vision::PatternDictionary dict =
            camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir);
        camdrop::vision::PatternFrameRenderer renderer(dict);

        Encoder encoder(opts.input_file);
        if (!encoder.is_valid()) {
            throw std::runtime_error("encoder init failed");
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

        uint32_t frame_index = 0;
        for (uint32_t logical = 0; logical < logical_frames; ++logical) {
            const std::vector<uint8_t> frame_bytes = encoder.get_packet();
            if (frame_bytes.size() != Config::PACKET_CAPACITY) {
                throw std::runtime_error("encoder returned unexpected frame size");
            }
            const cv::Mat image = compose_screen_frame(renderer.Render(frame_bytes), opts);
            for (int rep = 0; rep < opts.repeat; ++rep) {
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << frame_index++ << ".png";
                const fs::path out_path = out_dir / name.str();
                if (!cv::imwrite(out_path.string(), image)) {
                    throw std::runtime_error("failed to write image: " + out_path.string());
                }
            }
            std::cout << "\rgenerated logical frame " << (logical + 1) << "/" << logical_frames << std::flush;
        }
        std::cout << '\n';

        if (!opts.video_out.empty()) {
            const int rc = build_video_with_ffmpeg(opts, out_dir);
            if (rc != 0) {
                std::cerr << "ffmpeg failed with exit code " << rc << '\n';
                return 3;
            }
            std::cout << "video: " << fs::absolute(opts.video_out).string() << '\n';
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
