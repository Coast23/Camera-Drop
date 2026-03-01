#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────
// Anchor：一个检测到的定位块的像素范围
// ─────────────────────────────────────────────────────────────────
struct Anchor {
    int x, xmax, y, ymax;

    Anchor(int x_, int xmax_, int y_, int ymax_)
        : x(x_), xmax(xmax_), y(y_), ymax(ymax_) {}

    int xavg() const { return (x + xmax) / 2; }
    int yavg() const { return (y + ymax) / 2; }
    long long area() const { return (long long)std::abs(x - xmax) * std::abs(y - ymax); }
    cv::Point2f center() const { return {(float)xavg(), (float)yavg()}; }
};

// ─────────────────────────────────────────────────────────────────
// Corners：四个外角点（用于透视矫正）
// ─────────────────────────────────────────────────────────────────
struct Corners {
    cv::Point2f tl, tr, bl, br;
};

// ── 工具：计算 bbox 四角中离指定重心最远的角 ──
static cv::Point2f outer_corner_from_centroid(const Anchor& a, float gcx, float gcy) {
    cv::Point2f corners[4] = {
        {(float)a.x,    (float)a.y},
        {(float)a.xmax, (float)a.y},
        {(float)a.x,    (float)a.ymax},
        {(float)a.xmax, (float)a.ymax}
    };
    cv::Point2f best = corners[0];
    float bestD = -1;
    for (int i = 0; i < 4; ++i) {
        float dx = corners[i].x - gcx;
        float dy = corners[i].y - gcy;
        float d = dx*dx + dy*dy;
        if (d > bestD) { bestD = d; best = corners[i]; }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────
// 透视矫正：将识别到的 4 个角点变换到 out_size × out_size 的标准图像
// ─────────────────────────────────────────────────────────────────
cv::Mat deskew(const cv::Mat& img, const Corners& corners, int out_size = 1024) {
    std::vector<cv::Point2f> src = {corners.tl, corners.tr, corners.bl, corners.br};
    float s = (float)out_size;
    std::vector<cv::Point2f> dst = {{0,0}, {s,0}, {0,s}, {s,s}};
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat result;
    cv::warpPerspective(img, result, M, {out_size, out_size});
    return result;
}

// ─────────────────────────────────────────────────────────────────
// FrameDetector：用 YOLOv8 ONNX 模型定位目标
//
// 支持任意类别数 nc（nc = num_features - 4）
// 输出张量格式（自动检测）：
//   [1, 4+nc, 8400]  特征优先
//   [1, 8400, 4+nc]  框优先（onnxsim 后常见）
// ─────────────────────────────────────────────────────────────────
struct Detection {
    cv::Rect rect;
    int      class_id;
    float    score;
};

class FrameDetector {
public:
    FrameDetector(const std::string& model_path,
                  float conf_thresh = 0.4f,
                  float nms_thresh  = 0.45f)
        : _conf(conf_thresh), _nms(nms_thresh)
    {
        _net = cv::dnn::readNetFromONNX(model_path);
        if (_net.empty())
            throw std::runtime_error("无法加载 ONNX 模型: " + model_path);
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // 返回检测结果（原图坐标），按置信度降序
    // only_class: 若>=0，仅保留该类别（加速粗定位阶段的后处理）
    std::vector<Detection> detect(const cv::Mat& img, int only_class = -1) {
        const int inp = 640;
        int ow = img.cols, oh = img.rows;

        // 1. Letterbox resize
        float scale = std::min((float)inp / ow, (float)inp / oh);
        int nw = (int)(ow * scale), nh = (int)(oh * scale);
        int px = (inp - nw) / 2,    py = (inp - nh) / 2;

        cv::Mat canvas(inp, inp, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat resized;
        cv::resize(img, resized, {nw, nh});
        resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));

        // 2. BGR→RGB, 归一化, NCHW
        cv::Mat blob = cv::dnn::blobFromImage(
            canvas, 1.0 / 255.0, {inp, inp}, cv::Scalar(), true, false, CV_32F);
        _net.setInput(blob);

        // 3. 推理（只取 output0）
        std::vector<cv::Mat> outs;
        _net.forward(outs, std::vector<cv::String>{"output0"});

        if (outs.empty()) { printf("[YOLO] 推理无输出\n"); return {}; }

        // 4. 自动检测布局，确定 num_boxes / num_features
        const cv::Mat& raw = outs[0];
        printf("[YOLO] tensor dims=%d", raw.dims);
        for (int d = 0; d < raw.dims; ++d) printf(" [%d]=%d", d, raw.size[d]);
        printf("\n");

        int num_boxes = 0, num_feat = 0;
        bool row_major = false;

        if (raw.dims == 3) {
            int d1 = raw.size[1], d2 = raw.size[2];
            if (d1 < d2) { num_feat=d1; num_boxes=d2; row_major=false; }
            else          { num_feat=d2; num_boxes=d1; row_major=true;  }
        } else if (raw.dims == 2) {
            int d0 = raw.size[0], d1 = raw.size[1];
            if (d0 < d1) { num_feat=d0; num_boxes=d1; row_major=false; }
            else          { num_feat=d1; num_boxes=d0; row_major=true;  }
        } else {
            printf("[YOLO] 未知 tensor 形状\n");
            return {};
        }

        int nc = num_feat - 4;
        if (nc < 1) { printf("[YOLO] 特征维度异常 num_feat=%d\n", num_feat); return {}; }
        printf("[YOLO] num_boxes=%d  nc=%d  row_major=%d\n", num_boxes, nc, (int)row_major);

        const float* data = raw.ptr<float>();

        auto get_feat = [&](int box_i, int feat_i) -> float {
            return row_major ? data[box_i * num_feat + feat_i]
                             : data[feat_i * num_boxes + box_i];
        };

        // 5. 解析每个框
        float max_score = 0.f;
        std::vector<cv::Rect>  boxes;
        std::vector<float>     scores;
        std::vector<int>       class_ids;

        for (int i = 0; i < num_boxes; ++i) {
            float best_score = -1.f;
            int   best_cls   = 0;
            for (int c = 0; c < nc; ++c) {
                if (only_class >= 0 && c != only_class) continue;
                float s = get_feat(i, 4 + c);
                if (s > best_score) { best_score = s; best_cls = c; }
            }
            if (best_score > max_score) max_score = best_score;
            if (best_score < _conf) continue;

            float cx = get_feat(i, 0);
            float cy = get_feat(i, 1);
            float w  = get_feat(i, 2);
            float h  = get_feat(i, 3);

            float x1 = std::max(0.f, (cx - w/2.f - px) / scale);
            float y1 = std::max(0.f, (cy - h/2.f - py) / scale);
            float x2 = std::min((float)ow, (cx + w/2.f - px) / scale);
            float y2 = std::min((float)oh, (cy + h/2.f - py) / scale);

            boxes.push_back({(int)x1, (int)y1, (int)(x2-x1), (int)(y2-y1)});
            scores.push_back(best_score);
            class_ids.push_back(best_cls);
        }

        printf("[YOLO] max_score=%.4f  threshold=%.2f  candidates=%zu\n",
               max_score, _conf, boxes.size());

        // 6. NMS（按类别分别做）
        std::vector<Detection> result;
        for (int cls = 0; cls < nc; ++cls) {
            std::vector<cv::Rect>  cls_boxes;
            std::vector<float>     cls_scores;
            for (int i = 0; i < (int)boxes.size(); ++i) {
                if (class_ids[i] == cls) {
                    cls_boxes.push_back(boxes[i]);
                    cls_scores.push_back(scores[i]);
                }
            }
            std::vector<int> idx;
            cv::dnn::NMSBoxes(cls_boxes, cls_scores, _conf, _nms, idx);
            for (int i : idx) {
                Detection d;
                d.rect     = cls_boxes[i];
                d.class_id = cls;
                d.score    = cls_scores[i];
                result.push_back(d);
            }
        }

        std::sort(result.begin(), result.end(),
                  [](const Detection& a, const Detection& b){ return a.score > b.score; });
        return result;
    }

private:
    cv::dnn::Net _net;
    float _conf, _nms;
};

// ─────────────────────────────────────────────────────────────────
// 角色分配 + 外角选取 + 透视矫正
// ─────────────────────────────────────────────────────────────────
static void finish_deskew(cv::Mat& vis, cv::Mat& img,
                           std::vector<Anchor>& anchors,
                           std::vector<Anchor>& normal_anchors,
                           const std::vector<Detection>& br_dets,
                           const std::string& prefix)
{
    using namespace cv;
    printf("\n检测到 %zu 个定位块：\n", anchors.size());
    for (int i = 0; i < (int)anchors.size(); ++i) {
        auto& a = anchors[i];
        printf("  [%d] 中心(%d,%d)  x[%d,%d]  y[%d,%d]  面积=%lld\n",
               i, a.xavg(), a.yavg(), a.x, a.xmax, a.y, a.ymax, a.area());
    }

    for (int i = 0; i < (int)anchors.size(); ++i) {
        auto& a = anchors[i];
        Scalar col = Scalar(0, 128, 255);
        rectangle(vis, Point(a.x, a.y), Point(a.xmax, a.ymax), col, 2);
        circle(vis, Point(a.xavg(), a.yavg()), 5, col, -1);
        char lbl[8]; snprintf(lbl, sizeof(lbl), "%d", i);
        putText(vis, lbl, Point(a.xavg()+6, a.yavg()-6),
                FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }

    // ── BR-based cross-product role assignment ──────────────────
    const Detection& br_det = br_dets[0];
    Anchor brA(br_det.rect.x, br_det.rect.x + br_det.rect.width,
               br_det.rect.y, br_det.rect.y + br_det.rect.height);
    float br_cx = (float)brA.xavg(), br_cy = (float)brA.yavg();

    // TL = normal anchor 中离 BR 最远的
    int tl_idx = 0;
    float max_dist = -1;
    for (int i = 0; i < (int)normal_anchors.size(); ++i) {
        float dx = (float)normal_anchors[i].xavg() - br_cx;
        float dy = (float)normal_anchors[i].yavg() - br_cy;
        float d = dx*dx + dy*dy;
        if (d > max_dist) { max_dist = d; tl_idx = i; }
    }
    Anchor tlA = normal_anchors[tl_idx];
    float tl_cx = (float)tlA.xavg(), tl_cy = (float)tlA.yavg();

    // 剩余 normals
    std::vector<Anchor> rem;
    for (int i = 0; i < (int)normal_anchors.size(); ++i) {
        if (i != tl_idx) rem.push_back(normal_anchors[i]);
    }

    // TL→BR 向量，叉积区分 TR/BL
    float vx = br_cx - tl_cx, vy = br_cy - tl_cy;

    Anchor trA = rem[0], blA = rem[0];
    bool has_tr = false, has_bl = false;

    if (rem.size() >= 2) {
        float cross0 = vx * ((float)rem[0].yavg() - tl_cy) - vy * ((float)rem[0].xavg() - tl_cx);
        if (cross0 < 0) {
            trA = rem[0]; blA = rem[1];
        } else {
            trA = rem[1]; blA = rem[0];
        }
        has_tr = true; has_bl = true;
    } else if (rem.size() == 1) {
        float cross0 = vx * ((float)rem[0].yavg() - tl_cy) - vy * ((float)rem[0].xavg() - tl_cx);
        if (cross0 < 0) { trA = rem[0]; has_tr = true; }
        else            { blA = rem[0]; has_bl = true; }
    }

    if (!has_tr || !has_bl) {
        printf("TR 或 BL 不足，跳过 deskew。\n");
        return;
    }

    printf("角色分配 [BR-based cross-product]:\n");
    printf("  BR center=(%d,%d)\n", brA.xavg(), brA.yavg());
    printf("  TL center=(%d,%d)\n", tlA.xavg(), tlA.yavg());
    printf("  TR center=(%d,%d)\n", trA.xavg(), trA.yavg());
    printf("  BL center=(%d,%d)\n", blA.xavg(), blA.yavg());

    // ── 外角选取：离四锚点重心最远的 bbox 角 ─────────────────────
    float gcx = ((float)tlA.xavg() + (float)trA.xavg() +
                 (float)blA.xavg() + (float)brA.xavg()) / 4.0f;
    float gcy = ((float)tlA.yavg() + (float)trA.yavg() +
                 (float)blA.yavg() + (float)brA.yavg()) / 4.0f;
    printf("  centroid=(%.1f, %.1f)\n", gcx, gcy);

    Corners corners;
    corners.tl = outer_corner_from_centroid(tlA, gcx, gcy);
    corners.tr = outer_corner_from_centroid(trA, gcx, gcy);
    corners.bl = outer_corner_from_centroid(blA, gcx, gcy);
    corners.br = outer_corner_from_centroid(brA, gcx, gcy);

    printf("\n角点（centroid outer corner）：\n");
    printf("  TL = (%.1f, %.1f)\n", corners.tl.x, corners.tl.y);
    printf("  TR = (%.1f, %.1f)\n", corners.tr.x, corners.tr.y);
    printf("  BL = (%.1f, %.1f)\n", corners.bl.x, corners.bl.y);
    printf("  BR = (%.1f, %.1f)\n", corners.br.x, corners.br.y);

    auto mark = [&](cv::Point2f p, const char* label, Scalar color) {
        circle(vis, p, 10, color, 2);
        putText(vis, label, Point((int)p.x + 12, (int)p.y - 8),
                FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    };
    mark(corners.tl, "TL", Scalar(255, 100, 0));
    mark(corners.tr, "TR", Scalar(0, 255, 0));
    mark(corners.bl, "BL", Scalar(0, 128, 255));
    mark(corners.br, "BR", Scalar(0, 0, 255));

    Mat corrected = deskew(img, corners, 1024);
    imwrite(prefix + "deskewed.png", corrected);
    printf("透视矫正图像已保存为 %sdeskewed.png\n", prefix.c_str());
}

// ─────────────────────────────────────────────────────────────────
// main
// 用法：anchor_scanner image.jpg combined_fixed640.onnx
//
// combined.onnx 类别：
//   0: camera_drop_frame   1: qr_code
//   2: anchor(TL/TR/BL)    3: anchor_br
//   4: qr_finder           5: hanzi_hui
// ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    using namespace cv;

    if (argc < 3) {
        fprintf(stderr, "用法：anchor_scanner <图像> <combined_fixed640.onnx>\n");
        return 1;
    }

    std::string input_path = argv[1];
    std::string model_path = argv[2];

    // 输出文件前缀：去掉目录和扩展名，如 "../1.jpg" → "1_"
    std::string prefix;
    {
        std::string base = input_path;
        auto slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        auto dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        prefix = base + "_";
    }

    Mat img = imread(input_path);
    if (img.empty()) {
        fprintf(stderr, "错误：无法读取图像 '%s'\n", input_path.c_str());
        return 1;
    }
    printf("读取图像：%s (%dx%d)\n", input_path.c_str(), img.cols, img.rows);

    Mat vis = img.clone();

    // ── 两阶段推理─────────────────────
    // Stage 1: 640×640 全图，只解析 frame (cls=0)
    // Stage 2: 裁图 640×640 精检，所有类别
    const float FRAME_PAD = 0.35f;

    printf("合并模型：%s\n", model_path.c_str());
    FrameDetector det(model_path, 0.35f, 0.45f);

    // ── Stage 1 ──────────────────────────────────────────────────
    printf("[Stage1] 640×640, only cls=0 (frame)\n");
    auto dets1 = det.detect(img, 0);

    // 找最高置信度的 frame
    cv::Rect frame_rect;
    bool has_frame = false;
    float best_frame_score = 0.f;
    for (auto& d : dets1) {
        if (d.class_id == 0 && d.score > best_frame_score) {
            frame_rect = d.rect;
            best_frame_score = d.score;
            has_frame = true;
        }
    }

    if (!has_frame) {
        printf("未检测到 Camera-Drop 帧（class 0）。\n");
        imwrite(prefix + "detected.png", vis);
        return 1;
    }
    printf("[Stage1] 帧 bbox: x[%d,%d] y[%d,%d]  score=%.3f\n",
           frame_rect.x, frame_rect.x + frame_rect.width,
           frame_rect.y, frame_rect.y + frame_rect.height,
           best_frame_score);

    rectangle(vis, frame_rect, Scalar(0, 255, 255), 2);
    putText(vis, "frame", frame_rect.tl() + Point(4, 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    // Frame score gate（与 web 版一致：< 0.75 跳过）
    if (best_frame_score < 0.75f) {
        printf("[Stage1] frame score %.3f < 0.75, 跳过 deskew\n", best_frame_score);
        imwrite(prefix + "detected.png", vis);
        return 0;
    }

    // ── Stage 2：裁图精检 ────────────────────────────────────────
    int fw = frame_rect.width, fh = frame_rect.height;
    int pad = (int)(std::max(fw, fh) * FRAME_PAD);
    int cx1 = std::max(0, frame_rect.x - pad);
    int cy1 = std::max(0, frame_rect.y - pad);
    int cx2 = std::min(img.cols, frame_rect.x + fw + pad);
    int cy2 = std::min(img.rows, frame_rect.y + fh + pad);
    int cw = cx2 - cx1, ch = cy2 - cy1;
    printf("[Stage2] crop=[%d,%d,%d,%d] %dx%d  pad=%d\n", cx1, cy1, cx2, cy2, cw, ch, pad);

    Mat crop = img(cv::Rect(cx1, cy1, cw, ch));
    auto dets2_raw = det.detect(crop);
    printf("[Stage2] 检测到 %zu 个候选框\n", dets2_raw.size());

    // 坐标映射回原图空间
    std::vector<Detection> dets2;
    for (auto& d : dets2_raw) {
        Detection d2 = d;
        d2.rect.x += cx1;
        d2.rect.y += cy1;
        dets2.push_back(d2);
    }

    // Fallback：Stage 2 没找到 frame → 用 Stage 1 frame + Stage 2 非 frame
    bool has_frame2 = false;
    for (auto& d : dets2) {
        if (d.class_id == 0) { has_frame2 = true; break; }
    }

    std::vector<Detection> all_dets;
    if (has_frame2) {
        all_dets = dets2;
    } else {
        for (auto& d : dets1)
            if (d.class_id == 0) all_dets.push_back(d);
        for (auto& d : dets2)
            if (d.class_id != 0) all_dets.push_back(d);
    }

    // 重新找帧 rect
    frame_rect = cv::Rect();
    best_frame_score = 0.f;
    for (auto& d : all_dets) {
        if (d.class_id == 0 && d.score > best_frame_score) {
            frame_rect = d.rect;
            best_frame_score = d.score;
        }
    }

    // ── 严格锚点筛选：中心必须在 frame 内（零容差，与 web 一致）──
    std::vector<Anchor> anchors, normal_anchors;
    std::vector<Detection> br_dets;

    for (auto& d : all_dets) {
        if (d.class_id != 2 && d.class_id != 3) continue;
        int acx = d.rect.x + d.rect.width  / 2;
        int acy = d.rect.y + d.rect.height / 2;
        if (acx < frame_rect.x || acx > frame_rect.x + frame_rect.width)  continue;
        if (acy < frame_rect.y || acy > frame_rect.y + frame_rect.height) continue;

        Anchor a(d.rect.x, d.rect.x + d.rect.width,
                 d.rect.y, d.rect.y + d.rect.height);
        anchors.push_back(a);
        if (d.class_id == 3) {
            br_dets.push_back(d);
            printf("  [BR] center=(%d,%d)  score=%.3f\n", acx, acy, d.score);
        } else {
            normal_anchors.push_back(a);
        }
    }

    printf("筛选后: %zu anchors (%zu normal + %zu BR)\n",
           anchors.size(), normal_anchors.size(), br_dets.size());

    // 严格要求：>= 1 BR + >= 3 normal
    if (br_dets.size() < 1 || normal_anchors.size() < 3) {
        printf("锚点不足，跳过 deskew。\n");
        imwrite(prefix + "detected.png", vis);
        return 0;
    }

    finish_deskew(vis, img, anchors, normal_anchors, br_dets, prefix);

    imwrite(prefix + "detected.png", vis);
    printf("标注结果已保存为 %sdetected.png\n", prefix.c_str());

    return 0;
}
