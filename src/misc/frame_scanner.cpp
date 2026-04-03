#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "vision/frame_pipeline.hpp"

namespace fs = std::filesystem;

namespace {

std::string hex_preview(const std::vector<uint8_t>& bytes, size_t limit) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    const size_t n = std::min(limit, bytes.size());
    for (size_t i = 0; i < n; ++i) {
        if (i) {
            oss << ' ';
        }
        oss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    if (bytes.size() > n) {
        oss << " ...";
    }
    return oss.str();
}

void print_usage() {
    std::cout
        << "Usage: frame_scanner <image>"
        << " [--model <onnx>]"
        << " [--patterns <dir>]"
        << " [--pattern-cnn-model <onnx>]"
        << " [--deskew-out <png>]\n";
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
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string image_path;
    std::string model_path = default_model_path(argv[0]).string();
    std::string pattern_dir = default_pattern_dir(argv[0]).string();
    std::string pattern_cnn_model_path = default_pattern_cnn_model_path(argv[0]).string();
    std::string deskew_out;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            pattern_dir = argv[++i];
        } else if (arg == "--pattern-cnn-model" && i + 1 < argc) {
            pattern_cnn_model_path = argv[++i];
        } else if (arg == "--deskew-out" && i + 1 < argc) {
            deskew_out = argv[++i];
        } else if (!arg.empty() && arg[0] != '-' && image_path.empty()) {
            image_path = arg;
        } else {
            print_usage();
            return 1;
        }
    }

    if (image_path.empty()) {
        print_usage();
        return 1;
    }

    if (model_path == "web/model/best_dynamic.onnx") {
        model_path = default_model_path(argv[0]).string();
    }
    if (pattern_dir == "pattern_finder/best_v2") {
        pattern_dir = default_pattern_dir(argv[0]).string();
    }

    if (deskew_out.empty()) {
        fs::path p(image_path);
        deskew_out = (p.parent_path() / (p.stem().string() + "_deskewed.png")).string();
    }

    try {
        const cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
        if (frame.empty()) {
            throw std::runtime_error("failed to load image: " + image_path);
        }

        camdrop::vision::FramePipelineConfig cfg;
        cfg.model_path = model_path;
        cfg.pattern_dir = pattern_dir;
        cfg.pattern_cnn_model_path = pattern_cnn_model_path;

        camdrop::vision::FramePipeline pipeline(cfg);
        const camdrop::vision::PipelineResult result = pipeline.Process(frame);

        if (!result.localized) {
            std::cerr << "localize failed\n";
            return 2;
        }

        std::cout << "localize ok: " << result.localize.source
                  << " infer_ms=" << std::fixed << std::setprecision(2) << result.localize.inference_ms << '\n';
        std::cout << "corners"
                  << " TL(" << result.localize.corners.tl.x << ", " << result.localize.corners.tl.y << ")"
                  << " TR(" << result.localize.corners.tr.x << ", " << result.localize.corners.tr.y << ")"
                  << " BL(" << result.localize.corners.bl.x << ", " << result.localize.corners.bl.y << ")"
                  << " BR(" << result.localize.corners.br.x << ", " << result.localize.corners.br.y << ")\n";

        if (!result.deskewed) {
            std::cerr << "deskew failed\n";
            return 3;
        }

        cv::imwrite(deskew_out, result.deskewed_image);
        std::cout << "deskew saved: " << deskew_out << '\n';

        if (!result.recognized) {
            std::cerr << "recognize failed\n";
            return 4;
        }

        std::cout << "recognize ok"
                  << " avgPatternDist=" << std::fixed << std::setprecision(2) << result.recognize.avg_pattern_dist
                  << " header=" << result.recognize.header_bytes.size() << "B"
                  << " payload=" << result.recognize.payload_bytes.size() << "B\n";
        std::cout << "header preview: " << hex_preview(result.recognize.header_bytes, 16) << '\n';
        std::cout << "payload preview: " << hex_preview(result.recognize.payload_bytes, 16) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 10;
    }
}
