#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util/config.hpp"
#include "vision/frame_pipeline.hpp"
#include "vision/frame_renderer.hpp"
#include "vision/pattern_dict.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string model_path = "web/model/best_dynamic.onnx";
    std::string pattern_dir = "pattern_finder/best_v2";
    std::string pattern_cnn_model_path;
    std::string out_dir = "rect_layout_probe_out";
    uint32_t seed = 1;
    int screen_width = 1920;
    int screen_height = 1080;
    float mild_perspective = 0.45f;
    float strong_perspective = 1.0f;
};

struct ExpectedData {
    std::vector<uint8_t> interleaved_symbols;
    std::vector<uint8_t> header_symbols;
    std::vector<uint8_t> payload_symbols;
};

struct CaseSpec {
    std::string name;
    float perspective_strength = 0.0f;
};

struct CaseResult {
    std::string name;
    bool localized = false;
    bool deskewed = false;
    bool recognized = false;
    double header_acc = 0.0;
    double payload_acc = 0.0;
    double pattern_acc = 0.0;
    double color_acc = 0.0;
    std::string localize_source;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            opts.pattern_dir = argv[++i];
        } else if (arg == "--pattern-cnn-model" && i + 1 < argc) {
            opts.pattern_cnn_model_path = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            opts.out_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--screen-width" && i + 1 < argc) {
            opts.screen_width = std::stoi(argv[++i]);
        } else if (arg == "--screen-height" && i + 1 < argc) {
            opts.screen_height = std::stoi(argv[++i]);
        } else if (arg == "--mild-perspective" && i + 1 < argc) {
            opts.mild_perspective = std::stof(argv[++i]);
        } else if (arg == "--strong-perspective" && i + 1 < argc) {
            opts.strong_perspective = std::stof(argv[++i]);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.screen_width <= 0 || opts.screen_height <= 0) {
        throw std::runtime_error("screen size must be > 0");
    }
    return opts;
}

ExpectedData make_expected_data(uint32_t seed) {
    ExpectedData out;
    out.interleaved_symbols.resize(Config::UINTS_COUNT);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, (1 << Config::BITS_PER_UNIT) - 1);
    for (uint8_t& symbol : out.interleaved_symbols) {
        symbol = static_cast<uint8_t>(dist(rng));
    }

    out.header_symbols.assign(
        out.interleaved_symbols.begin(),
        out.interleaved_symbols.begin() + Config::HEADER_SYMBOL_COUNT);
    out.payload_symbols.assign(
        out.interleaved_symbols.begin() + Config::HEADER_SYMBOL_COUNT,
        out.interleaved_symbols.end());
    return out;
}

double calc_symbol_acc(const std::vector<uint8_t>& got, const std::vector<uint8_t>& exp) {
    if (got.empty() || exp.empty() || got.size() != exp.size()) {
        return 0.0;
    }
    size_t ok = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        ok += (got[i] == exp[i]) ? 1U : 0U;
    }
    return 100.0 * static_cast<double>(ok) / static_cast<double>(got.size());
}

double calc_component_acc(const std::vector<uint8_t>& got,
                          const std::vector<uint8_t>& exp,
                          int mask,
                          int shift) {
    if (got.empty() || exp.empty() || got.size() != exp.size()) {
        return 0.0;
    }
    size_t ok = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const int a = (got[i] >> shift) & mask;
        const int b = (exp[i] >> shift) & mask;
        ok += (a == b) ? 1U : 0U;
    }
    return 100.0 * static_cast<double>(ok) / static_cast<double>(got.size());
}

cv::Mat compose_screen_frame(const cv::Mat& code_image, int screen_width, int screen_height) {
    if (code_image.empty()) {
        throw std::runtime_error("compose_screen_frame got empty image");
    }
    if (code_image.cols > screen_width || code_image.rows > screen_height) {
        throw std::runtime_error("code image exceeds screen size");
    }

    cv::Mat screen(screen_height, screen_width, CV_8UC3, cv::Scalar(0, 0, 0));
    const int x = (screen_width - code_image.cols) / 2;
    const int y = (screen_height - code_image.rows) / 2;
    code_image.copyTo(screen(cv::Rect(x, y, code_image.cols, code_image.rows)));
    return screen;
}

cv::Mat apply_screen_perspective(const cv::Mat& image, float strength) {
    if (strength <= 0.0f) {
        return image.clone();
    }

    const float w = static_cast<float>(image.cols);
    const float h = static_cast<float>(image.rows);
    const std::array<cv::Point2f, 4> src = {{
        {0.0f, 0.0f},
        {w - 1.0f, 0.0f},
        {w - 1.0f, h - 1.0f},
        {0.0f, h - 1.0f},
    }};
    const std::array<cv::Point2f, 4> target = {{
        {0.08f * w, 0.13f * h},
        {0.93f * w, 0.04f * h},
        {0.90f * w, 0.93f * h},
        {0.06f * w, 0.87f * h},
    }};

    std::array<cv::Point2f, 4> dst = src;
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i].x = src[i].x + (target[i].x - src[i].x) * strength;
        dst[i].y = src[i].y + (target[i].y - src[i].y) * strength;
    }

    const cv::Mat H = cv::getPerspectiveTransform(src.data(), dst.data());
    cv::Mat warped;
    cv::warpPerspective(
        image,
        warped,
        H,
        image.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
    return warped;
}

CaseResult run_case(const CaseSpec& spec,
                    const cv::Mat& native_code,
                    const ExpectedData& expected,
                    camdrop::vision::FramePipeline& pipeline,
                    const fs::path& out_dir,
                    int screen_width,
                    int screen_height,
                    int pattern_bits) {
    cv::Mat screen = compose_screen_frame(native_code, screen_width, screen_height);
    screen = apply_screen_perspective(screen, spec.perspective_strength);
    cv::imwrite((out_dir / (spec.name + "_input.png")).string(), screen);

    CaseResult out;
    out.name = spec.name;

    const camdrop::vision::PipelineResult result = pipeline.Process(screen);
    out.localized = result.localized;
    out.deskewed = result.deskewed;
    out.recognized = result.recognized;
    out.localize_source = result.localize.source;

    if (!result.deskewed_image.empty()) {
        cv::imwrite((out_dir / (spec.name + "_deskew.png")).string(), result.deskewed_image);
    }

    if (result.recognized) {
        out.header_acc = calc_symbol_acc(result.recognize.header_symbols, expected.header_symbols);
        out.payload_acc = calc_symbol_acc(result.recognize.payload_symbols, expected.payload_symbols);
        out.pattern_acc = calc_component_acc(
            result.recognize.payload_symbols,
            expected.payload_symbols,
            (1 << pattern_bits) - 1,
            0);
        out.color_acc = calc_component_acc(
            result.recognize.payload_symbols,
            expected.payload_symbols,
            0x3,
            pattern_bits);
    }

    return out;
}

void print_result(const CaseResult& result) {
    std::cout
        << result.name
        << " localized=" << (result.localized ? "yes" : "no")
        << " deskewed=" << (result.deskewed ? "yes" : "no")
        << " recognized=" << (result.recognized ? "yes" : "no");
    if (!result.localize_source.empty()) {
        std::cout << " source=" << result.localize_source;
    }
    if (result.recognized) {
        std::cout
            << " header_acc=" << std::fixed << std::setprecision(3) << result.header_acc << "%"
            << " payload_acc=" << result.payload_acc << "%"
            << " pattern_acc=" << result.pattern_acc << "%"
            << " color_acc=" << result.color_acc << "%";
    }
    std::cout << '\n';
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
        const fs::path out_dir = fs::absolute(opts.out_dir);
        fs::create_directories(out_dir);

        const camdrop::vision::PatternDictionary dict =
            camdrop::vision::PatternDictionary::LoadFromDirectory(opts.pattern_dir);
        camdrop::vision::PatternFrameRenderer renderer(dict);
        const ExpectedData expected = make_expected_data(opts.seed);
        const cv::Mat native_code = renderer.RenderInterleavedSymbols(expected.interleaved_symbols);
        cv::imwrite((out_dir / "native_reference.png").string(), native_code);

        camdrop::vision::FramePipelineConfig cfg;
        cfg.model_path = opts.model_path;
        cfg.pattern_dir = opts.pattern_dir;
        cfg.pattern_cnn_model_path = opts.pattern_cnn_model_path;
        cfg.patch_track_enabled = false;
        camdrop::vision::FramePipeline pipeline(cfg);

        const std::vector<CaseSpec> cases = {
            {"native_rect", 0.0f},
            {"native_rect_perspective_mild", opts.mild_perspective},
            {"native_rect_perspective_strong", opts.strong_perspective},
        };

        std::cout << "model: " << fs::absolute(opts.model_path).string() << '\n';
        std::cout << "patterns: " << fs::absolute(opts.pattern_dir).string() << '\n';
        std::cout << "out_dir: " << out_dir.string() << '\n';
        std::cout << "layout: " << Config::IMG_WIDTH << "x" << Config::IMG_HEIGHT
                  << " grid=" << Config::GRID_C << "x" << Config::GRID_R
                  << " bits_per_unit=" << Config::BITS_PER_UNIT << '\n';

        for (const auto& spec : cases) {
            print_result(run_case(
                spec,
                native_code,
                expected,
                pipeline,
                out_dir,
                opts.screen_width,
                opts.screen_height,
                dict.pattern_bits()));
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
