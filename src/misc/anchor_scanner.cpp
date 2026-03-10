// anchor_scanner.cpp
// Camera-Drop-old 定位 + 透视矫正（ORT + Canny 版）
//
// 模型：YOLO26，ONNX（post-NMS）
// output0: [1, N, feat]  每行至少包含 [x1,y1,x2,y2, score, cls_id]（letterbox 坐标）
//
// 类别（nc=5）：
//   0: camera_drop_frame
//   1: qr_code
//   2: anchor       ← TL/TR/BL 三个普通锚点（白黑白黑同心环）
//   3: anchor_br    ← BR 锚点（四象限彩色同心环）
//   4: qr_finder
//
// 角点确定策略：
//   1. 全图推理，找置信度最高的 frame（cls=0）
//   2. 裁 frame+35% 区域精检，在 frame+5% 内筛出 3×anchor + 1×anchor_br
//   3. 对每个 anchor bbox 区域做 Canny 边缘检测 → minAreaRect → 取离 frame 中心最远的角
//   4. BR-based 角色分配 → warpPerspective
//
// 用法：anchor_scanner <图像> <best.onnx>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

// ─────────────────────────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────────────────────────
static const int   YOLO_INPUT  = 640;
static const int   FEAT_DIM_MIN = 6;
static const float CONF_THRESH = 0.35f;

static const int CLS_FRAME     = 0;
static const int CLS_ANCHOR    = 2;
static const int CLS_ANCHOR_BR = 3;

// ─────────────────────────────────────────────────────────────────
// Detection
// ─────────────────────────────────────────────────────────────────
struct Detection {
    cv::Rect2f box;   // x1,y1,w,h（原图坐标）
    float      score;
    int        cls;
};

// ─────────────────────────────────────────────────────────────────
// Letterbox
// ─────────────────────────────────────────────────────────────────
struct LetterboxInfo {
    float scale;
    int   pad_x, pad_y;
    int   orig_w, orig_h;
};

static cv::Mat letterbox(const cv::Mat& img, LetterboxInfo& info) {
    int ow = img.cols, oh = img.rows;
    float scale = std::min((float)YOLO_INPUT / ow, (float)YOLO_INPUT / oh);
    int nw = (int)(ow * scale), nh = (int)(oh * scale);
    int px = (YOLO_INPUT - nw) / 2, py = (YOLO_INPUT - nh) / 2;
    info = { scale, px, py, ow, oh };

    cv::Mat canvas(YOLO_INPUT, YOLO_INPUT, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(img, resized, {nw, nh});
    resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));
    return canvas;
}

static cv::Point2f lb_to_orig(float lx, float ly, const LetterboxInfo& lb) {
    float x = std::max(0.f, std::min((float)lb.orig_w, (lx - lb.pad_x) / lb.scale));
    float y = std::max(0.f, std::min((float)lb.orig_h, (ly - lb.pad_y) / lb.scale));
    return {x, y};
}

// ─────────────────────────────────────────────────────────────────
// 解析 output0 post-NMS
// ─────────────────────────────────────────────────────────────────
static std::vector<Detection> parse_output(
    const float* data, int n_dets, int feat_dim,
    const LetterboxInfo& lb, float conf_thresh)
{
    if (feat_dim < FEAT_DIM_MIN) return {};
    std::vector<Detection> dets;
    for (int i = 0; i < n_dets; ++i) {
        const float* row = data + i * feat_dim;
        float score = row[4];
        int   cls   = (int)row[5];
        if (score < conf_thresh) continue;
        if (cls != CLS_FRAME && cls != CLS_ANCHOR && cls != CLS_ANCHOR_BR) continue;

        cv::Point2f p1 = lb_to_orig(row[0], row[1], lb);
        cv::Point2f p2 = lb_to_orig(row[2], row[3], lb);
        dets.push_back({ {p1.x, p1.y, p2.x-p1.x, p2.y-p1.y}, score, cls });
    }
    return dets;
}

// ─────────────────────────────────────────────────────────────────
// ORT 推理（只用 output0，不请求 proto mask）
// ─────────────────────────────────────────────────────────────────
static std::vector<Detection> run_inference(
    Ort::Session& session, const cv::Mat& img, float conf_thresh)
{
    LetterboxInfo lb;
    cv::Mat canvas = letterbox(img, lb);

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<float> input_data(1 * 3 * YOLO_INPUT * YOLO_INPUT);
    std::vector<cv::Mat> planes(3);
    cv::split(rgb, planes);
    for (int c = 0; c < 3; ++c)
        memcpy(input_data.data() + c * YOLO_INPUT * YOLO_INPUT,
               planes[c].ptr<float>(), YOLO_INPUT * YOLO_INPUT * sizeof(float));

    std::vector<int64_t> input_shape = {1, 3, YOLO_INPUT, YOLO_INPUT};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[]  = {"images"};
    const char* output_names[] = {"output0"};
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names, &input_tensor, 1,
                               output_names, 1);

    auto shape0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int n_dets  = (int)shape0[1];
    int feat_dim = (int)shape0[2];
    const float* data0 = outputs[0].GetTensorData<float>();

    return parse_output(data0, n_dets, feat_dim, lb, conf_thresh);
}

// ─────────────────────────────────────────────────────────────────
// 对 anchor bbox 区域做 Canny 边缘检测，获取旋转矩形的 4 个角点
// ─────────────────────────────────────────────────────────────────
static std::vector<cv::Point2f> find_anchor_corners_image(
    const cv::Mat& img, const cv::Rect2f& bbox)
{
    const int margin = 4;
    int x1 = std::max(0,        (int)(bbox.x) - margin);
    int y1 = std::max(0,        (int)(bbox.y) - margin);
    int x2 = std::min(img.cols, (int)(bbox.x + bbox.width)  + margin);
    int y2 = std::min(img.rows, (int)(bbox.y + bbox.height) + margin);
    if (x2 <= x1 || y2 <= y1) return {};

    cv::Mat crop = img(cv::Rect(x1, y1, x2-x1, y2-y1));
    cv::Mat gray, blur, edges;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, {3, 3}, 0);
    cv::Canny(blur, edges, 30, 100);

    std::vector<cv::Point> pts;
    for (int r = 0; r < edges.rows; r++)
        for (int c = 0; c < edges.cols; c++)
            if (edges.at<uchar>(r, c) > 0)
                pts.push_back({c, r});

    if ((int)pts.size() < 10) return {};

    cv::RotatedRect rr = cv::minAreaRect(pts);
    cv::Point2f corners[4];
    rr.points(corners);

    std::vector<cv::Point2f> result;
    for (int i = 0; i < 4; i++)
        result.push_back({corners[i].x + x1, corners[i].y + y1});
    return result;
}

// ─────────────────────────────────────────────────────────────────
// 取 anchor 离 frame 中心最远的角点（Canny 分析，退化用 bbox 角）
// ─────────────────────────────────────────────────────────────────
static cv::Point2f outer_corner(const Detection& d, const cv::Mat& img,
                                float fc_x, float fc_y)
{
    auto pts = find_anchor_corners_image(img, d.box);
    if (pts.empty()) {
        pts = {
            {d.box.x,               d.box.y},
            {d.box.x + d.box.width,  d.box.y},
            {d.box.x,               d.box.y + d.box.height},
            {d.box.x + d.box.width,  d.box.y + d.box.height}
        };
    }
    cv::Point2f best = pts[0]; float bestD = -1;
    for (const auto& p : pts) {
        float dd = (p.x - fc_x)*(p.x - fc_x) + (p.y - fc_y)*(p.y - fc_y);
        if (dd > bestD) { bestD = dd; best = p; }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────
// 角色分配 + 透视矫正
// ─────────────────────────────────────────────────────────────────
static void finish_deskew(cv::Mat& vis, const cv::Mat& img,
                           const std::vector<Detection>& normal_dets,
                           const Detection& br_det,
                           const cv::Rect2f& frame_rect,
                           const std::string& prefix)
{
    using namespace cv;

    float fc_x = frame_rect.x + frame_rect.width  / 2.0f;
    float fc_y = frame_rect.y + frame_rect.height / 2.0f;

    float br_cx = br_det.box.x + br_det.box.width  / 2.0f;
    float br_cy = br_det.box.y + br_det.box.height / 2.0f;

    // TL = normal anchors 中离 BR 最远
    int tl_idx = 0; float max_dist = -1;
    for (int i = 0; i < (int)normal_dets.size(); ++i) {
        float dx = normal_dets[i].box.x + normal_dets[i].box.width/2.f  - br_cx;
        float dy = normal_dets[i].box.y + normal_dets[i].box.height/2.f - br_cy;
        float d  = dx*dx + dy*dy;
        if (d > max_dist) { max_dist = d; tl_idx = i; }
    }
    float tl_cx = normal_dets[tl_idx].box.x + normal_dets[tl_idx].box.width/2.f;
    float tl_cy = normal_dets[tl_idx].box.y + normal_dets[tl_idx].box.height/2.f;

    std::vector<int> rem;
    for (int i = 0; i < (int)normal_dets.size(); ++i) if (i != tl_idx) rem.push_back(i);
    if (rem.size() < 2) { printf("TR/BL 不足，跳过 deskew\n"); return; }

    float vx  = br_cx - tl_cx, vy = br_cy - tl_cy;
    float rx0 = normal_dets[rem[0]].box.x + normal_dets[rem[0]].box.width/2.f;
    float ry0 = normal_dets[rem[0]].box.y + normal_dets[rem[0]].box.height/2.f;
    float cross = vx*(ry0 - tl_cy) - vy*(rx0 - tl_cx);

    int tr_idx = (cross < 0) ? rem[0] : rem[1];
    int bl_idx = (cross < 0) ? rem[1] : rem[0];

    printf("角色分配 [BR-based]:\n");
    printf("  BR=(%.0f,%.0f)  TL=(%.0f,%.0f)  TR=(%.0f,%.0f)  BL=(%.0f,%.0f)\n",
           br_cx, br_cy, tl_cx, tl_cy,
           normal_dets[tr_idx].box.x + normal_dets[tr_idx].box.width/2.f,
           normal_dets[tr_idx].box.y + normal_dets[tr_idx].box.height/2.f,
           normal_dets[bl_idx].box.x + normal_dets[bl_idx].box.width/2.f,
           normal_dets[bl_idx].box.y + normal_dets[bl_idx].box.height/2.f);

    // 每个 anchor 用 Canny 分析取外角
    Point2f TL = outer_corner(normal_dets[tl_idx], img, fc_x, fc_y);
    Point2f TR = outer_corner(normal_dets[tr_idx], img, fc_x, fc_y);
    Point2f BL = outer_corner(normal_dets[bl_idx], img, fc_x, fc_y);
    Point2f BR = outer_corner(br_det,              img, fc_x, fc_y);

    printf("角点（Canny outer corner）：\n");
    printf("  TL=(%.1f,%.1f)  TR=(%.1f,%.1f)\n", TL.x, TL.y, TR.x, TR.y);
    printf("  BL=(%.1f,%.1f)  BR=(%.1f,%.1f)\n", BL.x, BL.y, BR.x, BR.y);

    auto mark = [&](Point2f p, const char* lbl, Scalar col) {
        circle(vis, p, 10, col, 2);
        putText(vis, lbl, Point((int)p.x + 12, (int)p.y - 8),
                FONT_HERSHEY_SIMPLEX, 0.7, col, 2);
    };
    mark(TL, "TL", Scalar(255,100,0));
    mark(TR, "TR", Scalar(0,255,0));
    mark(BL, "BL", Scalar(0,128,255));
    mark(BR, "BR", Scalar(0,0,255));

    const int OUT = 1024;
    std::vector<Point2f> src = {TL, TR, BL, BR};
    std::vector<Point2f> dst = {{0,0},{(float)OUT,0},{0,(float)OUT},{(float)OUT,(float)OUT}};
    Mat M = getPerspectiveTransform(src, dst);
    Mat corrected;
    warpPerspective(img, corrected, M, {OUT, OUT});
    imwrite(prefix + "deskewed.png", corrected);
    printf("[deskew] 透视矫正图像已保存\n");
}

// ─────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法：anchor_scanner <图像> <best.onnx>\n");
        return 1;
    }
    std::string input_path = argv[1];
    std::string model_path = argv[2];

    std::string prefix;
    {
        std::string base = input_path;
        auto slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        auto dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        prefix = base + "_";
    }

    cv::Mat img = cv::imread(input_path);
    if (img.empty()) { fprintf(stderr, "错误：无法读取图像 '%s'\n", input_path.c_str()); return 1; }
    printf("读取图像：%s (%dx%d)\n", input_path.c_str(), img.cols, img.rows);

    printf("加载模型：%s\n", model_path.c_str());
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "anchor_scanner");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    std::wstring wmodel(model_path.begin(), model_path.end());
    Ort::Session session(env, wmodel.c_str(), opts);
#else
    Ort::Session session(env, model_path.c_str(), opts);
#endif

    cv::Mat vis = img.clone();

    // ── Stage 1：全图推理，找 frame ──────────────────────────────
    printf("[Stage1] 全图推理...\n");
    auto dets1 = run_inference(session, img, CONF_THRESH);
    printf("[Stage1] 检测到 %zu 个\n", dets1.size());
    for (auto& d : dets1)
        printf("  cls=%d score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               d.cls, d.score, d.box.x, d.box.y,
               d.box.x + d.box.width, d.box.y + d.box.height);

    cv::Rect2f frame_rect;
    float best_frame_score = 0.f;
    bool has_frame = false;
    for (auto& d : dets1) {
        if (d.cls == CLS_FRAME && d.score > best_frame_score) {
            frame_rect = d.box; best_frame_score = d.score; has_frame = true;
        }
    }
    if (!has_frame) {
        printf("未检测到 frame（cls=0）\n");
        cv::imwrite(prefix + "detected.png", vis); return 1;
    }
    printf("[Stage1] frame bbox=(%.0f,%.0f,%.0f,%.0f) score=%.3f\n",
           frame_rect.x, frame_rect.y,
           frame_rect.x + frame_rect.width, frame_rect.y + frame_rect.height,
           best_frame_score);

    cv::Rect fr_ri((int)frame_rect.x, (int)frame_rect.y,
                   (int)frame_rect.width, (int)frame_rect.height);
    cv::rectangle(vis, fr_ri, cv::Scalar(0,255,255), 2);
    cv::putText(vis, "frame", fr_ri.tl() + cv::Point(4, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);

    // ── Stage 2：裁图精检（frame ± 35%） ─────────────────────────
    const float FRAME_PAD  = 0.35f;
    const float ANCHOR_EXP = 0.05f;

    int fw = (int)frame_rect.width, fh = (int)frame_rect.height;
    int pad = (int)(std::max(fw, fh) * FRAME_PAD);
    int cx1 = std::max(0,        (int)frame_rect.x - pad);
    int cy1 = std::max(0,        (int)frame_rect.y - pad);
    int cx2 = std::min(img.cols, (int)(frame_rect.x + fw) + pad);
    int cy2 = std::min(img.rows, (int)(frame_rect.y + fh) + pad);
    printf("[Stage2] crop=[%d,%d,%d,%d]\n", cx1, cy1, cx2, cy2);

    cv::Mat crop = img(cv::Rect(cx1, cy1, cx2-cx1, cy2-cy1));
    auto dets2_raw = run_inference(session, crop, CONF_THRESH);
    printf("[Stage2] 检测到 %zu 个\n", dets2_raw.size());

    std::vector<Detection> dets2;
    for (auto& d : dets2_raw) {
        Detection d2 = d;
        d2.box.x += cx1;
        d2.box.y += cy1;
        dets2.push_back(d2);
    }

    bool has_frame2 = false;
    for (auto& d : dets2) if (d.cls == CLS_FRAME) { has_frame2 = true; break; }

    std::vector<Detection> all_dets;
    if (has_frame2) {
        all_dets = dets2;
    } else {
        for (auto& d : dets1) if (d.cls == CLS_FRAME) all_dets.push_back(d);
        for (auto& d : dets2) if (d.cls != CLS_FRAME) all_dets.push_back(d);
    }

    frame_rect = cv::Rect2f(); best_frame_score = 0.f;
    for (auto& d : all_dets)
        if (d.cls == CLS_FRAME && d.score > best_frame_score)
            { frame_rect = d.box; best_frame_score = d.score; }

    // ── 锚点筛选（中心在 frame + 5% 内） ─────────────────────────
    float ex  = frame_rect.width  * ANCHOR_EXP;
    float ey  = frame_rect.height * ANCHOR_EXP;
    float fx1 = frame_rect.x - ex,                     fy1 = frame_rect.y - ey;
    float fx2 = frame_rect.x + frame_rect.width  + ex, fy2 = frame_rect.y + frame_rect.height + ey;

    std::vector<Detection> normal_dets;
    Detection best_br; bool has_br = false;

    for (auto& d : all_dets) {
        if (d.cls != CLS_ANCHOR && d.cls != CLS_ANCHOR_BR) continue;
        float acx = d.box.x + d.box.width  / 2.f;
        float acy = d.box.y + d.box.height / 2.f;
        if (acx < fx1 || acx > fx2 || acy < fy1 || acy > fy2) continue;

        cv::Scalar col = (d.cls == CLS_ANCHOR_BR) ? cv::Scalar(0,0,255) : cv::Scalar(0,128,255);
        cv::Rect ri((int)d.box.x, (int)d.box.y, (int)d.box.width, (int)d.box.height);
        cv::rectangle(vis, ri, col, 2);

        if (d.cls == CLS_ANCHOR_BR) {
            if (!has_br || d.score > best_br.score) { best_br = d; has_br = true; }
            printf("  [BR]     center=(%.0f,%.0f) score=%.3f\n", acx, acy, d.score);
        } else {
            normal_dets.push_back(d);
            printf("  [anchor] center=(%.0f,%.0f) score=%.3f\n", acx, acy, d.score);
        }
    }

    std::sort(normal_dets.begin(), normal_dets.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });
    if (normal_dets.size() > 3) normal_dets.resize(3);

    printf("筛选后: %zu normal + %s BR\n", normal_dets.size(), has_br ? "1" : "0");

    if (!has_br || normal_dets.size() < 3) {
        printf("锚点不足（需要 3 normal + 1 BR），跳过 deskew。\n");
        cv::imwrite(prefix + "detected.png", vis); return 0;
    }

    finish_deskew(vis, img, normal_dets, best_br, frame_rect, prefix);

    cv::imwrite(prefix + "detected.png", vis);
    printf("\n输出文件：\n  %sdetected.png\n  %sdeskewed.png\n",
           prefix.c_str(), prefix.c_str());
    return 0;
}
