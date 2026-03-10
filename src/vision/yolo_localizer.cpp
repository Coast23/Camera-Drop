#include "vision/yolo_localizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace camdrop::vision {
namespace {

constexpr int kClsFrame = 0;
constexpr int kClsAnchor = 2;
constexpr int kClsAnchorBr = 3;
constexpr int kFeatureDimMin = 6;

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int orig_w = 0;
    int orig_h = 0;
};

struct SnapInput {
    cv::Mat image;
    float scale = 1.0f;
    float inv = 1.0f;
};

struct AnchorDetection {
    float cx = 0.0f;
    float cy = 0.0f;
    Detection det;
};

struct AnchorBuckets {
    std::vector<AnchorDetection> normals;
    std::vector<AnchorDetection> brs;
};

template <typename T>
T clamp_value(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

LetterboxResult letterbox(const cv::Mat& img, int size) {
    const int ow = img.cols;
    const int oh = img.rows;
    const float scale = std::min(static_cast<float>(size) / static_cast<float>(ow),
                                 static_cast<float>(size) / static_cast<float>(oh));
    const int nw = static_cast<int>(std::lround(static_cast<double>(ow) * scale));
    const int nh = static_cast<int>(std::lround(static_cast<double>(oh) * scale));
    const int px = (size - nw) / 2;
    const int py = (size - nh) / 2;

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));
    return {canvas, scale, px, py, ow, oh};
}

SnapInput make_snap_input(const cv::Mat& frame, int size) {
    const int vw = frame.cols;
    const int vh = frame.rows;
    const int long_side = std::max(vw, vh);
    if (long_side <= 0) {
        return {};
    }

    const float snap_scale = std::min(1.0f, static_cast<float>(size) / static_cast<float>(long_side));
    if (snap_scale >= 0.9999f) {
        return {frame, 1.0f, 1.0f};
    }

    const int sw = std::max(1, static_cast<int>(std::lround(static_cast<double>(vw) * snap_scale)));
    const int sh = std::max(1, static_cast<int>(std::lround(static_cast<double>(vh) * snap_scale)));
    cv::Mat snapped;
    cv::resize(frame, snapped, cv::Size(sw, sh), 0.0, 0.0, cv::INTER_AREA);
    return {snapped, snap_scale, 1.0f / snap_scale};
}

cv::Point2f lb_to_orig(float lx, float ly, const LetterboxResult& lb) {
    return {
        clamp_value((lx - static_cast<float>(lb.pad_x)) / lb.scale, 0.0f, static_cast<float>(lb.orig_w)),
        clamp_value((ly - static_cast<float>(lb.pad_y)) / lb.scale, 0.0f, static_cast<float>(lb.orig_h)),
    };
}

std::vector<float> to_chw_tensor(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> planes(3);
    cv::split(rgb, planes);

    const int plane_size = bgr.rows * bgr.cols;
    std::vector<float> tensor(3 * plane_size);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(tensor.data() + c * plane_size, planes[c].ptr<float>(), plane_size * sizeof(float));
    }
    return tensor;
}

std::vector<Detection> parse_output(const Ort::Value& output,
                                    const LetterboxResult& lb,
                                    float conf_threshold) {
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 3) {
        throw std::runtime_error("unexpected output0 shape");
    }

    const int n_dets = static_cast<int>(shape[1]);
    const int feat_dim = static_cast<int>(shape[2]);
    if (feat_dim < kFeatureDimMin) {
        throw std::runtime_error("unexpected output0 feature dimension");
    }

    const float* data = output.GetTensorData<float>();
    std::vector<Detection> detections;
    detections.reserve(n_dets);

    for (int i = 0; i < n_dets; ++i) {
        const float* row = data + i * feat_dim;
        const float score = row[4];
        if (score < conf_threshold) {
            continue;
        }
        const int cls = static_cast<int>(row[5]);
        if (cls != kClsFrame && cls != kClsAnchor && cls != kClsAnchorBr) {
            continue;
        }

        const cv::Point2f p1 = lb_to_orig(row[0], row[1], lb);
        const cv::Point2f p2 = lb_to_orig(row[2], row[3], lb);
        if (p2.x <= p1.x || p2.y <= p1.y) {
            continue;
        }
        detections.push_back({
            cv::Rect2f(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y),
            score,
            cls,
        });
    }

    return detections;
}

void blur_gray3(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int w, int h) {
    dst.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const int y0 = y > 0 ? (y - 1) : y;
        const int y1 = y + 1 < h ? (y + 1) : y;
        for (int x = 0; x < w; ++x) {
            const int x0 = x > 0 ? (x - 1) : x;
            const int x1 = x + 1 < w ? (x + 1) : x;
            int sum = 0;
            int count = 0;
            for (int yy = y0; yy <= y1; ++yy) {
                for (int xx = x0; xx <= x1; ++xx) {
                    sum += src[yy * w + xx];
                    ++count;
                }
            }
            dst[y * w + x] = static_cast<uint8_t>(sum / count);
        }
    }
}

std::vector<cv::Point2f> detect_anchor_quad(const cv::Mat& frame, const cv::Rect2f& box) {
    const int margin = 4;
    const int x1 = std::max(0, static_cast<int>(std::floor(box.x)) - margin);
    const int y1 = std::max(0, static_cast<int>(std::floor(box.y)) - margin);
    const int x2 = std::min(frame.cols, static_cast<int>(std::ceil(box.x + box.width)) + margin);
    const int y2 = std::min(frame.rows, static_cast<int>(std::ceil(box.y + box.height)) + margin);
    const int w = x2 - x1;
    const int h = y2 - y1;
    if (w < 6 || h < 6) {
        return {};
    }

    cv::Mat crop = frame(cv::Rect(x1, y1, w, h)).clone();
    std::vector<uint8_t> gray(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            const cv::Vec3b& px = row[x];
            gray[y * w + x] = static_cast<uint8_t>((px[2] * 77 + px[1] * 150 + px[0] * 29) >> 8);
        }
    }

    std::vector<uint8_t> blur;
    blur_gray3(gray, blur, w, h);
    std::vector<cv::Point2f> pts;
    pts.reserve(gray.size() / 4);

    const int thr = clamp_value(static_cast<int>(std::lround((w + h) * 1.2)), 48, 160);
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const int idx = y * w + x;
            const int gx = (blur[idx - w + 1] + (blur[idx + 1] << 1) + blur[idx + w + 1])
                         - (blur[idx - w - 1] + (blur[idx - 1] << 1) + blur[idx + w - 1]);
            const int gy = (blur[idx + w - 1] + (blur[idx + w] << 1) + blur[idx + w + 1])
                         - (blur[idx - w - 1] + (blur[idx - w] << 1) + blur[idx - w + 1]);
            if (std::abs(gx) + std::abs(gy) >= thr) {
                pts.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }
    }

    if (pts.size() < 10) {
        return {};
    }

    double cx = 0.0;
    double cy = 0.0;
    for (const auto& p : pts) {
        cx += p.x;
        cy += p.y;
    }
    cx /= static_cast<double>(pts.size());
    cy /= static_cast<double>(pts.size());

    double cxx = 0.0;
    double cxy = 0.0;
    double cyy = 0.0;
    for (const auto& p : pts) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }

    const double angle = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
    const double ux = std::cos(angle);
    const double uy = std::sin(angle);
    const double vx = -uy;
    const double vy = ux;

    double min_u = std::numeric_limits<double>::infinity();
    double max_u = -std::numeric_limits<double>::infinity();
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        const double u = dx * ux + dy * uy;
        const double v = dx * vx + dy * vy;
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    auto make_point = [&](double u, double v) -> cv::Point2f {
        return {
            static_cast<float>(x1 + cx + u * ux + v * vx),
            static_cast<float>(y1 + cy + u * uy + v * vy),
        };
    };

    return {
        make_point(min_u, min_v),
        make_point(max_u, min_v),
        make_point(min_u, max_v),
        make_point(max_u, max_v),
    };
}

CornerQuad build_corners_from_centers(const std::array<cv::Point2f, 4>& anchors) {
    constexpr float canon_w = static_cast<float>(kImageWidth);
    constexpr float canon_h = static_cast<float>(kImageHeight);
    constexpr float anchor_out = static_cast<float>(kAnchorOutStart);
    constexpr float anchor_center = static_cast<float>(kAnchorOutStart + (kAnchorL1Size / 2));
    const std::array<cv::Point2f, 4> src = {
        cv::Point2f(anchor_center, anchor_center),
        cv::Point2f(canon_w - anchor_center, anchor_center),
        cv::Point2f(anchor_center, canon_h - anchor_center),
        cv::Point2f(canon_w - anchor_center, canon_h - anchor_center),
    };
    const cv::Mat H = cv::getPerspectiveTransform(src.data(), anchors.data());

    std::vector<cv::Point2f> canonical = {
        cv::Point2f(anchor_out, anchor_out),
        cv::Point2f(canon_w - anchor_out, anchor_out),
        cv::Point2f(anchor_out, canon_h - anchor_out),
        cv::Point2f(canon_w - anchor_out, canon_h - anchor_out),
    };
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(canonical, projected, H);

    CornerQuad corners;
    corners.tl = projected[0];
    corners.tr = projected[1];
    corners.bl = projected[2];
    corners.br = projected[3];
    corners.out_size = static_cast<int>(std::lround(std::max({
        cv::norm(corners.tr - corners.tl),
        cv::norm(corners.br - corners.bl),
        cv::norm(corners.bl - corners.tl),
        cv::norm(corners.br - corners.tr),
    })));
    return corners;
}

CornerQuad scale_corners(const CornerQuad& in, float inv) {
    CornerQuad out = in;
    out.tl *= inv;
    out.tr *= inv;
    out.bl *= inv;
    out.br *= inv;
    out.out_size = static_cast<int>(std::lround(static_cast<double>(in.out_size) * inv));
    return out;
}

CornerQuad build_corners_from_anchor_detections(const std::vector<AnchorDetection>& normals_in,
                                                const AnchorDetection& br_anchor,
                                                const std::optional<cv::Rect2f>& frame_box,
                                                const cv::Mat& frame,
                                                float input_inv,
                                                bool refine_anchor_quad,
                                                bool* used_center_fallback) {
    std::vector<AnchorDetection> normals = normals_in;
    std::sort(normals.begin(), normals.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
        return a.det.score > b.det.score;
    });
    if (normals.size() > 3) {
        normals.resize(3);
    }

    std::sort(normals.begin(), normals.end(), [&](const AnchorDetection& a, const AnchorDetection& b) {
        const double da = (a.cx - br_anchor.cx) * (a.cx - br_anchor.cx) + (a.cy - br_anchor.cy) * (a.cy - br_anchor.cy);
        const double db = (b.cx - br_anchor.cx) * (b.cx - br_anchor.cx) + (b.cy - br_anchor.cy) * (b.cy - br_anchor.cy);
        return da > db;
    });

    const AnchorDetection tl_anchor = normals[0];
    const AnchorDetection a0 = normals[1];
    const AnchorDetection a1 = normals[2];
    const float vx = br_anchor.cx - tl_anchor.cx;
    const float vy = br_anchor.cy - tl_anchor.cy;
    const float cross = vx * (a0.cy - tl_anchor.cy) - vy * (a0.cx - tl_anchor.cx);
    const AnchorDetection tr_anchor = cross < 0.0f ? a0 : a1;
    const AnchorDetection bl_anchor = cross < 0.0f ? a1 : a0;

    if (!frame_box.has_value()) {
        if (used_center_fallback) {
            *used_center_fallback = true;
        }
        return scale_corners(build_corners_from_centers({
            cv::Point2f(tl_anchor.cx, tl_anchor.cy),
            cv::Point2f(tr_anchor.cx, tr_anchor.cy),
            cv::Point2f(bl_anchor.cx, bl_anchor.cy),
            cv::Point2f(br_anchor.cx, br_anchor.cy),
        }), input_inv);
    }

    const cv::Rect2f box = *frame_box;
    const float fcx = (box.x + box.width * 0.5f) * input_inv;
    const float fcy = (box.y + box.height * 0.5f) * input_inv;
    auto choose_outer = [&](const AnchorDetection& anchor) -> cv::Point2f {
        const cv::Rect2f det_box = anchor.det.box;
        const cv::Rect2f orig_box(
            det_box.x * input_inv,
            det_box.y * input_inv,
            det_box.width * input_inv,
            det_box.height * input_inv);
        if (refine_anchor_quad) {
            const auto quad = detect_anchor_quad(frame, orig_box);
            if (quad.size() == 4) {
                cv::Point2f best = quad[0];
                float best_d = -1.0f;
                for (const auto& p : quad) {
                    const float dd = (p.x - fcx) * (p.x - fcx) + (p.y - fcy) * (p.y - fcy);
                    if (dd > best_d) {
                        best_d = dd;
                        best = p;
                    }
                }
                return best;
            }
        }

        std::array<cv::Point2f, 4> candidates = {
            cv::Point2f(orig_box.x, orig_box.y),
            cv::Point2f(orig_box.x + orig_box.width, orig_box.y),
            cv::Point2f(orig_box.x, orig_box.y + orig_box.height),
            cv::Point2f(orig_box.x + orig_box.width, orig_box.y + orig_box.height),
        };
        cv::Point2f best = candidates[0];
        float best_d = -1.0f;
        for (const auto& p : candidates) {
            const float dd = (p.x - fcx) * (p.x - fcx) + (p.y - fcy) * (p.y - fcy);
            if (dd > best_d) {
                best_d = dd;
                best = p;
            }
        }
        return best;
    };

    CornerQuad corners;
    corners.tl = choose_outer(tl_anchor);
    corners.tr = choose_outer(tr_anchor);
    corners.bl = choose_outer(bl_anchor);
    corners.br = choose_outer(br_anchor);
    corners.out_size = static_cast<int>(std::lround(std::max(box.width, box.height) * input_inv));
    return corners;
}

AnchorBuckets collect_anchor_detections(const std::vector<Detection>& all,
                                        const std::optional<cv::Rect2f>& frame_box,
                                        float anchor_expand) {
    AnchorBuckets out;
    float bx1 = 0.0f;
    float by1 = 0.0f;
    float bx2 = 0.0f;
    float by2 = 0.0f;
    float ex = 0.0f;
    float ey = 0.0f;
    if (frame_box.has_value()) {
        const cv::Rect2f box = *frame_box;
        bx1 = box.x;
        by1 = box.y;
        bx2 = box.x + box.width;
        by2 = box.y + box.height;
        ex = box.width * anchor_expand;
        ey = box.height * anchor_expand;
    }

    for (const auto& det : all) {
        if (det.cls != kClsAnchor && det.cls != kClsAnchorBr) {
            continue;
        }
        const float ax = det.box.x + det.box.width * 0.5f;
        const float ay = det.box.y + det.box.height * 0.5f;
        if (frame_box.has_value()) {
            if (ax < bx1 - ex || ax > bx2 + ex || ay < by1 - ey || ay > by2 + ey) {
                continue;
            }
        }
        AnchorDetection anchor{ax, ay, det};
        if (det.cls == kClsAnchorBr) {
            out.brs.push_back(anchor);
        } else {
            out.normals.push_back(anchor);
        }
    }
    return out;
}

std::vector<Detection> scale_detections(const std::vector<Detection>& detections, float inv) {
    if (std::abs(inv - 1.0f) < 1e-6f) {
        return detections;
    }
    std::vector<Detection> out;
    out.reserve(detections.size());
    for (const auto& det : detections) {
        out.push_back({
            cv::Rect2f(det.box.x * inv, det.box.y * inv, det.box.width * inv, det.box.height * inv),
            det.score,
            det.cls,
        });
    }
    return out;
}

std::optional<LocalizeResult> assign_corners(const std::vector<Detection>& detections,
                                             const cv::Mat& frame,
                                             float input_inv,
                                             bool refine_anchor_quad,
                                             float anchor_expand) {
    std::optional<cv::Rect2f> best_frame;
    float best_frame_score = -1.0f;
    for (const auto& det : detections) {
        if (det.cls == kClsFrame && det.score > best_frame_score) {
            best_frame = det.box;
            best_frame_score = det.score;
        }
    }

    bool used_center_fallback = false;
    if (best_frame.has_value()) {
        AnchorBuckets scoped = collect_anchor_detections(detections, best_frame, anchor_expand);
        if (scoped.normals.size() >= 3 && !scoped.brs.empty()) {
            std::sort(scoped.brs.begin(), scoped.brs.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
                return a.det.score > b.det.score;
            });
            LocalizeResult result;
            result.ok = true;
            result.corners = build_corners_from_anchor_detections(
                scoped.normals,
                scoped.brs[0],
                best_frame,
                frame,
                input_inv,
                refine_anchor_quad,
                &used_center_fallback);
            result.detections = scale_detections(detections, input_inv);
            result.frame_box = cv::Rect2f(
                best_frame->x * input_inv,
                best_frame->y * input_inv,
                best_frame->width * input_inv,
                best_frame->height * input_inv);
            result.has_frame = true;
            result.used_center_fallback = used_center_fallback;
            return result;
        }
    }

    AnchorBuckets global = collect_anchor_detections(detections, std::nullopt, anchor_expand);
    if (global.normals.size() < 3 || global.brs.empty()) {
        return std::nullopt;
    }

    std::sort(global.brs.begin(), global.brs.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
        return a.det.score > b.det.score;
    });

    LocalizeResult result;
    result.ok = true;
    result.corners = build_corners_from_anchor_detections(
        global.normals,
        global.brs[0],
        std::nullopt,
        frame,
        input_inv,
        refine_anchor_quad,
        &used_center_fallback);
    result.detections = scale_detections(detections, input_inv);
    if (best_frame.has_value()) {
        result.frame_box = cv::Rect2f(
            best_frame->x * input_inv,
            best_frame->y * input_inv,
            best_frame->width * input_inv,
            best_frame->height * input_inv);
        result.has_frame = true;
    }
    result.used_center_fallback = used_center_fallback;
    return result;
}

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

}  // namespace

YoloLocalizer::YoloLocalizer(const std::string& model_path, YoloLocalizerOptions options)
    : options_(std::move(options)),
      env_(ORT_LOGGING_LEVEL_WARNING, "camera_drop_yolo"),
      session_(create_session(env_, model_path, options_.ort_threads)) {}

std::optional<LocalizeResult> YoloLocalizer::Locate(const cv::Mat& frame) {
    if (frame.empty()) {
        return std::nullopt;
    }

    const SnapInput snapped = make_snap_input(frame, options_.input_size);
    const LetterboxResult lb = letterbox(snapped.image, options_.input_size);
    std::vector<float> input = to_chw_tensor(lb.image);
    const std::array<int64_t, 4> input_shape = {1, 3, options_.input_size, options_.input_size};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {options_.input_name.c_str()};
    const char* output_names[] = {options_.output_name.c_str()};

    const auto t0 = std::chrono::steady_clock::now();
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<Detection> detections = parse_output(outputs[0], lb, options_.conf_threshold);
    std::optional<LocalizeResult> result = assign_corners(
        detections,
        frame,
        snapped.inv,
        options_.refine_anchor_quad,
        options_.anchor_expand);
    if (!result.has_value()) {
        return std::nullopt;
    }
    result->inference_ms = elapsed_ms;
    result->source = options_.refine_anchor_quad ? "yolo+quad" : "yolo";
    return result;
}

}  // namespace camdrop::vision
