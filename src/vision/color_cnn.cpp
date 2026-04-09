#include "vision/color_cnn.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "util/config.hpp"

namespace camdrop::vision {
namespace {

constexpr int kImageWidth = Config::IMG_WIDTH;
constexpr int kImageHeight = Config::IMG_HEIGHT;
constexpr int kGridRows = Config::GRID_R;
constexpr int kGridCols = Config::GRID_C;
constexpr int kStride = Config::STRIDE;
constexpr int kMargin = Config::MARGIN;
constexpr int kTileSize = Config::TILE_SIZE;
constexpr int kSamplePad = 2;
constexpr int kPatchSize = kTileSize + kSamplePad * 2;
constexpr int kPatchArea = kPatchSize * kPatchSize;
constexpr int kNumClasses = 4;

Ort::Session create_session(Ort::Env& env, const std::string& model_path, int threads) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(std::max(1, threads));
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    std::wstring wide_path(model_path.begin(), model_path.end());
    return Ort::Session(env, wide_path.c_str(), options);
#else
    return Ort::Session(env, model_path.c_str(), options);
#endif
}

cv::Mat normalize_color_input(const cv::Mat& input) {
    cv::Mat color;
    if (input.channels() == 3) {
        color = input;
    } else if (input.channels() == 1) {
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, color, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("unsupported deskewed image format for color cnn");
    }
    if (color.cols == kImageWidth && color.rows == kImageHeight) {
        return color;
    }
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(kImageWidth, kImageHeight), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

std::pair<std::vector<int16_t>, std::vector<int16_t>> build_payload_positions() {
    std::vector<int16_t> xs;
    std::vector<int16_t> ys;
    xs.reserve(Config::PAYLOAD_SYMBOL_COUNT);
    ys.reserve(Config::PAYLOAD_SYMBOL_COUNT);
    for (int r = 0; r < kGridRows; ++r) {
        for (int c = 0; c < kGridCols; ++c) {
            const bool in_left = c < Config::ANCHOR_RESERVED_CELLS;
            const bool in_right = c >= (kGridCols - Config::ANCHOR_RESERVED_CELLS);
            const bool in_top = r < Config::ANCHOR_RESERVED_CELLS;
            const bool in_bottom = r >= (kGridRows - Config::ANCHOR_RESERVED_CELLS);
            if ((in_top && in_left) || (in_top && in_right) || (in_bottom && in_left) || (in_bottom && in_right)) {
                continue;
            }
            if (r == Config::CALIB_ROW && c >= Config::CALIB_COL_BEGIN && c < Config::CALIB_COL_END) {
                continue;
            }
            if (r == Config::HEADER_ROW && c >= Config::HEADER_COL_BEGIN && c < Config::HEADER_COL_END) {
                continue;
            }
            xs.push_back(static_cast<int16_t>(kMargin + c * kStride - kSamplePad));
            ys.push_back(static_cast<int16_t>(kMargin + r * kStride - kSamplePad));
        }
    }
    return {std::move(xs), std::move(ys)};
}

}  // namespace

ColorCnnClassifier::ColorCnnClassifier(const std::string& model_path, ColorCnnOptions options)
    : options_(std::move(options)),
      env_(ORT_LOGGING_LEVEL_WARNING, "color_cnn"),
      session_(create_session(env_, model_path, options_.ort_threads)) {
    auto payload_positions = build_payload_positions();
    payload_x_ = std::move(payload_positions.first);
    payload_y_ = std::move(payload_positions.second);
    if (payload_x_.size() != Config::PAYLOAD_SYMBOL_COUNT) {
        throw std::runtime_error("color cnn payload position count mismatch");
    }
}

std::vector<uint8_t> ColorCnnClassifier::PredictPayloadColors(const cv::Mat& deskewed) {
    if (deskewed.empty()) {
        return {};
    }

    const cv::Mat color = normalize_color_input(deskewed);
    const size_t sample_count = payload_x_.size();
    std::vector<uint8_t> predictions(sample_count, 0);
    std::vector<float> input_buffer(std::min(sample_count, options_.batch_size) * 3 * kPatchArea, 0.0f);
    const size_t batch_limit = std::max<size_t>(1, options_.batch_size);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<const char*, 1> input_names = {options_.input_name.c_str()};
    const std::array<const char*, 1> output_names = {options_.output_name.c_str()};

    for (size_t begin = 0; begin < sample_count; begin += batch_limit) {
        const size_t batch = std::min(batch_limit, sample_count - begin);
        input_buffer.resize(batch * 3 * kPatchArea);
        float* dst = input_buffer.data();
        for (size_t i = 0; i < batch; ++i) {
            const int x0 = payload_x_[begin + i];
            const int y0 = payload_y_[begin + i];
            for (int ch = 0; ch < 3; ++ch) {
                for (int r = 0; r < kPatchSize; ++r) {
                    const cv::Vec3b* row = color.ptr<cv::Vec3b>(y0 + r) + x0;
                    for (int c = 0; c < kPatchSize; ++c) {
                        *dst++ = static_cast<float>(row[c][ch]);
                    }
                }
            }
        }

        const std::array<int64_t, 4> input_shape = {
            static_cast<int64_t>(batch), 3, kPatchSize, kPatchSize
        };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_buffer.data(),
            input_buffer.size(),
            input_shape.data(),
            input_shape.size());
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw std::runtime_error("color cnn returned empty output");
        }
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const size_t total_logits = info.GetElementCount();
        if (total_logits != batch * kNumClasses) {
            throw std::runtime_error("color cnn output shape mismatch");
        }

        const float* logits = outputs[0].GetTensorData<float>();
        for (size_t i = 0; i < batch; ++i) {
            const float* row = logits + i * kNumClasses;
            int best_cls = 0;
            float best_logit = row[0];
            for (int cls = 1; cls < kNumClasses; ++cls) {
                if (row[cls] > best_logit) {
                    best_logit = row[cls];
                    best_cls = cls;
                }
            }
            predictions[begin + i] = static_cast<uint8_t>(best_cls);
        }
    }

    return predictions;
}

}  // namespace camdrop::vision
