#include "vision/deskew.hpp"

#include <array>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "util/errors.hpp"

namespace camdrop::vision {
namespace {

cv::Point2f scale_point(const cv::Point2f& p, const cv::Point2f& center, float ratio) {
    return {
        center.x + (p.x - center.x) * ratio,
        center.y + (p.y - center.y) * ratio,
    };
}

bool compute_homography(const std::array<cv::Point2f, 4>& src,
                        const std::array<cv::Point2f, 4>& dst,
                        std::array<double, 8>& out) {
    double M[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        const double x = src[i].x;
        const double y = src[i].y;
        const double u = dst[i].x;
        const double v = dst[i].y;
        const int row = i * 2;
        M[row][0] = x;
        M[row][1] = y;
        M[row][2] = 1.0;
        M[row][3] = 0.0;
        M[row][4] = 0.0;
        M[row][5] = 0.0;
        M[row][6] = -u * x;
        M[row][7] = -u * y;
        M[row][8] = u;

        M[row + 1][0] = 0.0;
        M[row + 1][1] = 0.0;
        M[row + 1][2] = 0.0;
        M[row + 1][3] = x;
        M[row + 1][4] = y;
        M[row + 1][5] = 1.0;
        M[row + 1][6] = -v * x;
        M[row + 1][7] = -v * y;
        M[row + 1][8] = v;
    }

    for (int c = 0; c < 8; ++c) {
        int pivot = c;
        double max_abs = std::abs(M[c][c]);
        for (int r = c + 1; r < 8; ++r) {
            const double v = std::abs(M[r][c]);
            if (v > max_abs) {
                max_abs = v;
                pivot = r;
            }
        }
        if (max_abs < 1e-12) {
            return false;
        }
        if (pivot != c) {
            for (int j = c; j < 9; ++j) {
                std::swap(M[c][j], M[pivot][j]);
            }
        }
        for (int r = c + 1; r < 8; ++r) {
            const double f = M[r][c] / M[c][c];
            for (int j = c; j < 9; ++j) {
                M[r][j] -= f * M[c][j];
            }
        }
    }

    for (int i = 7; i >= 0; --i) {
        double val = M[i][8];
        for (int j = i + 1; j < 8; ++j) {
            val -= M[i][j] * out[j];
        }
        out[i] = val / M[i][i];
    }
    return true;
}

inline cv::Vec3b sample_nearest_bgr(const cv::Mat& src, double x, double y) {
    int sx = static_cast<int>(std::lround(x));
    int sy = static_cast<int>(std::lround(y));
    sx = std::max(0, std::min(sx, src.cols - 1));
    sy = std::max(0, std::min(sy, src.rows - 1));
    return src.at<cv::Vec3b>(sy, sx);
}

}  // namespace

cv::Mat Deskewer::Deskew(const cv::Mat& frame,
                         const CornerQuad& corners,
                         int out_width,
                         int out_height,
                         float expand,
                         int canonical_inset,
                         int interpolation) {
    if (frame.empty()) {
        throw VisionDeskewError("Input frame is empty");
    }
    
    if (out_width <= 0 || out_height <= 0) {
        throw VisionDeskewError("Invalid output dimensions: " + std::to_string(out_width) + "x" + std::to_string(out_height));
    }
    
    if (!corners.valid()) {
        throw VisionDeskewError("Invalid corner quad");
    }
    
    (void)interpolation;

    cv::Mat src;
    if (frame.type() == CV_8UC3) {
        src = frame;
    } else if (frame.type() == CV_8UC4) {
        cv::cvtColor(frame, src, cv::COLOR_BGRA2BGR);
    } else if (frame.type() == CV_8UC1) {
        cv::cvtColor(frame, src, cv::COLOR_GRAY2BGR);
    } else {
        throw VisionDeskewError("Unsupported input image type: " + std::to_string(frame.type()));
    }

    const cv::Point2f center(
        (corners.tl.x + corners.tr.x + corners.bl.x + corners.br.x) * 0.25f,
        (corners.tl.y + corners.tr.y + corners.bl.y + corners.br.y) * 0.25f);
    const cv::Point2f e_tl = scale_point(corners.tl, center, expand);
    const cv::Point2f e_tr = scale_point(corners.tr, center, expand);
    const cv::Point2f e_bl = scale_point(corners.bl, center, expand);
    const cv::Point2f e_br = scale_point(corners.br, center, expand);

    const float inset_x = static_cast<float>(canonical_inset) * (static_cast<float>(out_width) / static_cast<float>(kImageWidth));
    const float inset_y = static_cast<float>(canonical_inset) * (static_cast<float>(out_height) / static_cast<float>(kImageHeight));
    const float max_x = std::max(inset_x, static_cast<float>(out_width) - inset_x);
    const float max_y = std::max(inset_y, static_cast<float>(out_height) - inset_y);
    std::array<cv::Point2f, 4> src_pts = {
        cv::Point2f(inset_x, inset_y),
        cv::Point2f(max_x, inset_y),
        cv::Point2f(inset_x, max_y),
        cv::Point2f(max_x, max_y),
    };
    std::array<cv::Point2f, 4> dst_pts = {e_tl, e_tr, e_bl, e_br};
    std::array<double, 8> h = {};
    if (!compute_homography(src_pts, dst_pts, h)) {
        throw VisionDeskewError("Failed to compute homography matrix");
    }

    cv::Mat out(out_height, out_width, CV_8UC3, cv::Scalar(38, 38, 38));
    const double src_w = static_cast<double>(src.cols);
    const double src_h = static_cast<double>(src.rows);
    for (int y = 0; y < out_height; ++y) {
        cv::Vec3b* row = out.ptr<cv::Vec3b>(y);
        const double iy = static_cast<double>(y) + 0.5;
        for (int x = 0; x < out_width; ++x) {
            const double ix = static_cast<double>(x) + 0.5;
            const double denom = (h[6] * ix) + (h[7] * iy) + 1.0;
            if (std::abs(denom) < 1e-12) {
                row[x] = cv::Vec3b(38, 38, 38);
                continue;
            }
            const double u = ((h[0] * ix) + (h[1] * iy) + h[2]) / denom;
            const double v = ((h[3] * ix) + (h[4] * iy) + h[5]) / denom;
            if (u < 0.0 || u > src_w || v < 0.0 || v > src_h) {
                row[x] = cv::Vec3b(38, 38, 38);
                continue;
            }
            row[x] = sample_nearest_bgr(src, u, v);
        }
    }
    return out;
}

}  // namespace camdrop::vision
