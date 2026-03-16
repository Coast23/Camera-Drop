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

// ── Contour-based anchor center refinement (ported from web app-localizer-cv.js) ──

struct ContourInfo {
    cv::RotatedRect rect;
    cv::Point2f center;
    float size;      // (w + h) / 2
    float side_min;
    float side_max;
    float aspect;    // side_min / side_max
};

struct AnchorChainFit {
    float scale       = 0.0f;
    float size_err    = 0.0f;
    float drift_err   = 0.0f;
    float aspect_err  = 0.0f;
    float score       = 0.0f;
    float outer_scale = 1.0f;
};

static ContourInfo make_contour_info(const std::vector<cv::Point>& pts) {
    const cv::RotatedRect r = cv::minAreaRect(pts);
    const float w = r.size.width;
    const float h = r.size.height;
    const float smin = std::min(w, h);
    const float smax = std::max(w, h);
    return {r, r.center, (w + h) * 0.5f, smin, smax, smin / std::max(1e-6f, smax)};
}

static int find_nested_child(
    int parent,
    const std::vector<ContourInfo>& infos,
    const std::vector<cv::Vec4i>& hierarchy)
{
    const int first = hierarchy[parent][2];
    if (first < 0) return -1;

    const float p_size = infos[parent].size;
    const cv::Point2f p_ctr = infos[parent].center;

    int best = -1;
    float best_score = std::numeric_limits<float>::infinity();

    for (int idx = first; idx >= 0; idx = hierarchy[idx][0]) {
        if (idx >= static_cast<int>(infos.size())) break;
        const auto& c = infos[idx];
        if (c.aspect < 0.72f || c.size >= p_size * 0.96f) continue;
        const float drift = std::hypot(c.center.x - p_ctr.x, c.center.y - p_ctr.y)
                          / std::max(1.0f, p_size);
        if (drift > 0.16f) continue;
        const float ratio = c.size / std::max(1.0f, p_size);
        const float sc = drift * 4.0f + std::abs(ratio - 0.68f);
        if (sc < best_score) { best_score = sc; best = idx; }
    }
    return best;
}

// Build chain of nested contours starting from root (up to depth 4)
static std::vector<int> build_nested_chain(
    int root,
    const std::vector<ContourInfo>& infos,
    const std::vector<cv::Vec4i>& hierarchy)
{
    std::vector<int> chain;
    int cur = root;
    while (cur >= 0 && static_cast<int>(chain.size()) < 4) {
        if (cur >= static_cast<int>(infos.size())) break;
        chain.push_back(cur);
        cur = find_nested_child(cur, infos, hierarchy);
    }
    return chain;
}

// Validate nested ring size ratios against known anchor layer sequences
// Layers [56, 42, 28, 14] match Config::ANCHOR_L*_SIZE (web: NORMAL_SEQUENCES)
static std::optional<AnchorChainFit> fit_normal_chain(
    const std::vector<int>& chain,
    const std::vector<ContourInfo>& infos)
{
    if (static_cast<int>(chain.size()) < 3) return std::nullopt;

    struct Spec { float layers[4]; int n; float outer_scale; };
    static const Spec kSpecs[] = {
        {{56.f, 42.f, 28.f, 14.f}, 4, 1.0f      },
        {{42.f, 28.f, 14.f,  0.f}, 3, 56.f/42.f },
    };

    std::optional<AnchorChainFit> best;
    for (const auto& spec : kSpecs) {
        if (static_cast<int>(chain.size()) < spec.n) continue;

        float scale = 0.0f;
        for (int i = 0; i < spec.n; ++i)
            scale += infos[chain[i]].size / spec.layers[i];
        scale /= spec.n;

        float size_err = 0.0f, drift_err = 0.0f, aspect_err = 0.0f;
        const cv::Point2f root_ctr = infos[chain[0]].center;
        for (int i = 0; i < spec.n; ++i) {
            const auto& ci = infos[chain[i]];
            const float expected = spec.layers[i] * scale;
            size_err += std::abs(ci.size - expected) / std::max(1.0f, expected);
            const float d = std::hypot(ci.center.x - root_ctr.x,
                                       ci.center.y - root_ctr.y);
            drift_err = std::max(drift_err, d / std::max(1.0f, scale * 56.0f));
            aspect_err = std::max(aspect_err, std::abs(1.0f - ci.aspect));
        }
        size_err /= spec.n;
        const float score = size_err + drift_err * 1.6f + aspect_err * 0.5f;
        if (!best || score < best->score)
            best = AnchorChainFit{scale, size_err, drift_err, aspect_err, score, spec.outer_scale};
    }
    if (!best) return std::nullopt;
    if (best->size_err > 0.095f || best->drift_err > 0.12f || best->aspect_err > 0.2f)
        return std::nullopt;
    return best;
}

// Warp a RotatedRect region of src into a 64×64 image
static cv::Mat warp_anchor_roi(const cv::Mat& src, const cv::RotatedRect& rect) {
    cv::Point2f corners[4];
    rect.points(corners);

    // Sort corners: TL(min x+y), TR(min y-x), BR(max x+y), BL(max y-x)
    int tl = 0, tr = 0, br = 0, bl = 0;
    float ms = corners[0].x + corners[0].y, xs = ms;
    float md = corners[0].y - corners[0].x, xd = md;
    for (int i = 1; i < 4; ++i) {
        const float s = corners[i].x + corners[i].y;
        const float d = corners[i].y - corners[i].x;
        if (s < ms) { ms = s; tl = i; }
        if (s > xs) { xs = s; br = i; }
        if (d < md) { md = d; tr = i; }
        if (d > xd) { xd = d; bl = i; }
    }
    std::vector<cv::Point2f> src_pts = {corners[tl], corners[tr], corners[br], corners[bl]};
    std::vector<cv::Point2f> dst_pts = {{0,0},{64,0},{64,64},{0,64}};
    const cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(src, warped, M, {64, 64}, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return warped;
}

// Score 64×64 binary anchor ROI: expects W-B-W-B ring pattern from outside in
// (matches web scoreNormalAnchorRoi)
struct AnchorRoiScore {
    float outer_white, ring_black, inner_white, center_black, total;
};

/** @brief 评分锚点 ROI
    @param roi 64x64 ROI 图像
    @return 锚点 ROI 评分
*/
static AnchorRoiScore score_anchor_roi(const cv::Mat& roi) {
    constexpr float kCenter = (64 - 1) * 0.5f;
    float sums[4] = {};
    int   cnts[4] = {};
    for (int y = 0; y < 64; ++y) {
        const uint8_t* row = roi.ptr<uint8_t>(y);
        for (int x = 0; x < 64; ++x) {
            const float w   = row[x] / 255.0f;
            const float d   = std::max(std::abs((x + 0.5f) - kCenter),
                                       std::abs((y + 0.5f) - kCenter));
            const int band  = (d >= 24) ? 0 : (d >= 16) ? 1 : (d >= 8) ? 2 : 3;
            sums[band] += w;
            cnts[band] += 1;
        }
    }
    const float ow = sums[0] / std::max(1, cnts[0]);
    const float rb = sums[1] / std::max(1, cnts[1]);
    const float iw = sums[2] / std::max(1, cnts[2]);
    const float cb = sums[3] / std::max(1, cnts[3]);
    return {ow, rb, iw, cb, ow + iw + (1.0f - rb) + (1.0f - cb)};
}

// Refine anchor center within a YOLO bbox using contour-based ring detection.
// Returns the detected ring center in `frame` coordinates, or bbox center as fallback.
/** @brief 使用轮廓检测精炼锚点中心
    @param frame 输入帧
    @param bbox YOLO 边界框
    @return 精炼后的锚点中心或回退到边界框中心
*/
static cv::Point2f refine_anchor_center_contour(const cv::Mat& frame, const cv::Rect2f& bbox) {
    const cv::Point2f fallback(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);

    const int margin = std::max(4, static_cast<int>(
        std::round(std::max(bbox.width, bbox.height) * 0.15f)));
    const int x1 = std::max(0, static_cast<int>(std::floor(bbox.x)) - margin);
    const int y1 = std::max(0, static_cast<int>(std::floor(bbox.y)) - margin);
    const int x2 = std::min(frame.cols, static_cast<int>(std::ceil(bbox.x + bbox.width))  + margin);
    const int y2 = std::min(frame.rows, static_cast<int>(std::ceil(bbox.y + bbox.height)) + margin);
    const int w = x2 - x1, h = y2 - y1;
    if (w < 12 || h < 12) return fallback;

    cv::Mat gray;
    cv::cvtColor(frame(cv::Rect(x1, y1, w, h)), gray, cv::COLOR_BGR2GRAY);

    int block_size = std::max(3, static_cast<int>(std::round(std::min(w, h) / 5.0)));
    if (block_size % 2 == 0) ++block_size;
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          block_size, 5);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return fallback;

    const int n = static_cast<int>(contours.size());
    std::vector<ContourInfo> infos(n);
    for (int i = 0; i < n; ++i) infos[i] = make_contour_info(contours[i]);

    const float min_side = std::max(3.0f, std::min((float)w, (float)h) * 0.04f);
    const float max_side = std::min((float)w, (float)h) * 0.90f;

    float best_rank = -std::numeric_limits<float>::infinity();
    cv::Point2f best_center = {-1.0f, -1.0f};

    for (int i = 0; i < n; ++i) {
        if (hierarchy[i][2] < 0) continue; // must have children (nested rings)

        const auto& info = infos[i];
        if (info.aspect < 0.72f || info.side_min < min_side || info.side_max > max_side)
            continue;

        const auto chain = build_nested_chain(i, infos, hierarchy);
        const auto fit   = fit_normal_chain(chain, infos);
        if (!fit) continue;

        cv::RotatedRect outer_rect = info.rect;
        outer_rect.size.width  *= fit->outer_scale;
        outer_rect.size.height *= fit->outer_scale;

        const cv::Mat roi    = warp_anchor_roi(binary, outer_rect);
        const AnchorRoiScore ts = score_anchor_roi(roi);

        if (ts.outer_white < 0.68f || ts.inner_white < 0.62f ||
            ts.ring_black  > 0.36f || ts.center_black > 0.24f || ts.total < 3.25f)
            continue;

        const float rank = ts.total - fit->score * 3.0f
                         + (static_cast<int>(chain.size()) >= 4 ? 0.18f : 0.0f);
        if (rank > best_rank) {
            best_rank   = rank;
            best_center = {info.center.x + x1, info.center.y + y1};
        }
    }

    return best_center.x >= 0.0f ? best_center : fallback;
}

// Refine BR anchor center: same contour chain detection but NO W-B-W-B score filter
// (BR has colored quadrant rings, not black/white, so score_anchor_roi is inappropriate)
/** @brief 精炼 BR 锚点中心
    @param frame 输入帧
    @param bbox YOLO 边界框
    @return 精炼后的 BR 锚点中心或回退到边界框中心
*/
static cv::Point2f refine_br_center_contour(const cv::Mat& frame, const cv::Rect2f& bbox) {
    const cv::Point2f fallback(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);

    const int margin = std::max(4, static_cast<int>(
        std::round(std::max(bbox.width, bbox.height) * 0.15f)));
    const int x1 = std::max(0, static_cast<int>(std::floor(bbox.x)) - margin);
    const int y1 = std::max(0, static_cast<int>(std::floor(bbox.y)) - margin);
    const int x2 = std::min(frame.cols, static_cast<int>(std::ceil(bbox.x + bbox.width))  + margin);
    const int y2 = std::min(frame.rows, static_cast<int>(std::ceil(bbox.y + bbox.height)) + margin);
    const int w = x2 - x1, h = y2 - y1;
    if (w < 12 || h < 12) return fallback;

    cv::Mat gray;
    cv::cvtColor(frame(cv::Rect(x1, y1, w, h)), gray, cv::COLOR_BGR2GRAY);

    int block_size = std::max(3, static_cast<int>(std::round(std::min(w, h) / 5.0)));
    if (block_size % 2 == 0) ++block_size;
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          block_size, 5);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return fallback;

    const int n = static_cast<int>(contours.size());
    std::vector<ContourInfo> infos(n);
    for (int i = 0; i < n; ++i) infos[i] = make_contour_info(contours[i]);

    const float min_side = std::max(3.0f, std::min((float)w, (float)h) * 0.04f);
    const float max_side = std::min((float)w, (float)h) * 0.90f;

    float best_fit = std::numeric_limits<float>::infinity();
    cv::Point2f best_center = {-1.0f, -1.0f};

    for (int i = 0; i < n; ++i) {
        if (hierarchy[i][2] < 0) continue;
        const auto& info = infos[i];
        if (info.aspect < 0.72f || info.side_min < min_side || info.side_max > max_side)
            continue;
        const auto chain = build_nested_chain(i, infos, hierarchy);
        const auto fit = fit_normal_chain(chain, infos);
        if (!fit) continue;
        if (fit->score < best_fit) {
            best_fit    = fit->score;
            best_center = {info.center.x + x1, info.center.y + y1};
        }
    }

    return best_center.x >= 0.0f ? best_center : fallback;
}

// ── Corner building ──

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

// Assign TL/TR/BL roles among 3 normal anchors.
// Mirrors web's orderNormalTriple: TL = vertex opposite the longest edge (the TR-BL diagonal).
// TR/BL disambiguated by checking which remaining vertex lies on the "right" side of TL.
// Returns indices {tl, tr, bl} into the input centers array.
/** @brief 为三个正常锚点分配 TL/TR/BL 角色
    @param c 三个锚点中心
    @return 索引数组 {tl, tr, bl}
*/
static std::array<int, 3> order_normal_triple(const std::array<cv::Point2f, 3>& c) {
    // edges[i] is the edge vector opposite to vertex i
    const std::array<cv::Point2f, 3> edges = {{
        {c[1].x - c[2].x, c[1].y - c[2].y},
        {c[2].x - c[0].x, c[2].y - c[0].y},
        {c[0].x - c[1].x, c[0].y - c[1].y},
    }};
    int tl = 0;
    float max_d = -1.0f;
    for (int i = 0; i < 3; ++i) {
        const float d = edges[i].x * edges[i].x + edges[i].y * edges[i].y;
        if (d > max_d) { max_d = d; tl = i; }
    }
    auto fix = [](int i) -> int { return i < 0 ? 2 : i >= 3 ? 0 : i; };
    const cv::Point2f dep  = edges[fix(tl - 1)];
    const cv::Point2f inc  = edges[fix(tl + 1)];
    const cv::Point2f rot(-inc.y, inc.x);  // 90° CCW rotation of inc
    const cv::Point2f ovlp(dep.x - rot.x, dep.y - rot.y);
    const float dep_d = dep.x * dep.x + dep.y * dep.y;
    const float ov_d  = ovlp.x * ovlp.x + ovlp.y * ovlp.y;
    const int tr = ov_d < dep_d ? fix(tl + 1) : fix(tl - 1);
    const int bl = ov_d < dep_d ? fix(tl - 1) : fix(tl + 1);
    return {tl, tr, bl};
}

// Assign TL/TR/BL/BR roles then refine each center via contour detection,
// finally project frame corners via homography (matching web buildCornersFromCenters).
CornerQuad build_corners_from_anchor_detections(
    const std::vector<AnchorDetection>& normals_in,
    const AnchorDetection& br_anchor,
    const cv::Mat& frame,
    float input_inv)
{
    std::vector<AnchorDetection> normals = normals_in;
    std::sort(normals.begin(), normals.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
        return a.det.score > b.det.score;
    });
    if (normals.size() > 3) normals.resize(3);

    // Role assignment via web's orderNormalTriple
    const std::array<cv::Point2f, 3> centers = {{
        {normals[0].cx, normals[0].cy},
        {normals[1].cx, normals[1].cy},
        {normals[2].cx, normals[2].cy},
    }};
    const auto idx = order_normal_triple(centers);
    const AnchorDetection& tl_anchor = normals[idx[0]];
    const AnchorDetection& tr_anchor = normals[idx[1]];
    const AnchorDetection& bl_anchor = normals[idx[2]];

    auto refine_normal = [&](const AnchorDetection& anchor) -> cv::Point2f {
        const cv::Rect2f orig_box(
            anchor.det.box.x * input_inv,
            anchor.det.box.y * input_inv,
            anchor.det.box.width * input_inv,
            anchor.det.box.height * input_inv);
        return refine_anchor_center_contour(frame, orig_box);
    };
    auto refine_br = [&](const AnchorDetection& anchor) -> cv::Point2f {
        const cv::Rect2f orig_box(
            anchor.det.box.x * input_inv,
            anchor.det.box.y * input_inv,
            anchor.det.box.width * input_inv,
            anchor.det.box.height * input_inv);
        return refine_br_center_contour(frame, orig_box);
    };

    return build_corners_from_centers({
        refine_normal(tl_anchor),
        refine_normal(tr_anchor),
        refine_normal(bl_anchor),
        refine_br(br_anchor),
    });
}

AnchorBuckets collect_anchor_detections(const std::vector<Detection>& all,
                                        const std::optional<cv::Rect2f>& frame_box,
                                        float anchor_expand) {
    AnchorBuckets out;
    float bx1 = 0.0f, by1 = 0.0f, bx2 = 0.0f, by2 = 0.0f;
    float ex = 0.0f, ey = 0.0f;
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
        if (det.cls != kClsAnchor && det.cls != kClsAnchorBr) continue;
        const float ax = det.box.x + det.box.width * 0.5f;
        const float ay = det.box.y + det.box.height * 0.5f;
        if (frame_box.has_value()) {
            if (ax < bx1 - ex || ax > bx2 + ex || ay < by1 - ey || ay > by2 + ey)
                continue;
        }
        AnchorDetection anchor{ax, ay, det};
        if (det.cls == kClsAnchorBr) out.brs.push_back(anchor);
        else                          out.normals.push_back(anchor);
    }
    return out;
}

std::vector<Detection> scale_detections(const std::vector<Detection>& detections, float inv) {
    if (std::abs(inv - 1.0f) < 1e-6f) return detections;
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
                                             float anchor_expand) {
    std::optional<cv::Rect2f> best_frame;
    float best_frame_score = -1.0f;
    for (const auto& det : detections) {
        if (det.cls == kClsFrame && det.score > best_frame_score) {
            best_frame = det.box;
            best_frame_score = det.score;
        }
    }

    if (best_frame.has_value()) {
        AnchorBuckets scoped = collect_anchor_detections(detections, best_frame, anchor_expand);
        if (scoped.normals.size() >= 3 && !scoped.brs.empty()) {
            std::sort(scoped.brs.begin(), scoped.brs.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
                return a.det.score > b.det.score;
            });
            LocalizeResult result;
            result.ok = true;
            result.corners = build_corners_from_anchor_detections(
                scoped.normals, scoped.brs[0], frame, input_inv);
            result.detections = scale_detections(detections, input_inv);
            result.frame_box = cv::Rect2f(
                best_frame->x * input_inv, best_frame->y * input_inv,
                best_frame->width * input_inv, best_frame->height * input_inv);
            result.has_frame = true;
            return result;
        }
    }

    AnchorBuckets global = collect_anchor_detections(detections, std::nullopt, anchor_expand);
    if (global.normals.size() < 3 || global.brs.empty()) return std::nullopt;

    std::sort(global.brs.begin(), global.brs.end(), [](const AnchorDetection& a, const AnchorDetection& b) {
        return a.det.score > b.det.score;
    });

    LocalizeResult result;
    result.ok = true;
    result.corners = build_corners_from_anchor_detections(
        global.normals, global.brs[0], frame, input_inv);
    result.detections = scale_detections(detections, input_inv);
    if (best_frame.has_value()) {
        result.frame_box = cv::Rect2f(
            best_frame->x * input_inv, best_frame->y * input_inv,
            best_frame->width * input_inv, best_frame->height * input_inv);
        result.has_frame = true;
    }
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
    if (frame.empty()) return std::nullopt;

    const SnapInput snapped = make_snap_input(frame, options_.input_size);
    const LetterboxResult lb = letterbox(snapped.image, options_.input_size);
    std::vector<float> input = to_chw_tensor(lb.image);
    const std::array<int64_t, 4> input_shape = {1, 3, options_.input_size, options_.input_size};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input.data(), input.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[]  = {options_.input_name.c_str()};
    const char* output_names[] = {options_.output_name.c_str()};

    const auto t0 = std::chrono::steady_clock::now();
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<Detection> detections = parse_output(outputs[0], lb, options_.conf_threshold);
    std::optional<LocalizeResult> result = assign_corners(
        detections, frame, snapped.inv, options_.anchor_expand);
    if (!result.has_value()) return std::nullopt;

    result->inference_ms = elapsed_ms;
    result->source = "yolo+contour";
    return result;
}

}  // namespace camdrop::vision
