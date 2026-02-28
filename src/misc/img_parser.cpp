// Author: Coast23
// Date: 2026-02-28
/*
解决图像的定位与标准化问题
*/

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

const int IMG_SIZE = 1024;

const double ANCHOR_CENTER_OFFSET = 30.0;

enum AnchorType {
    ANCHOR_NONE = 0,
    ANCHOR_NORMAL = 1, // 1:1:4:1:1
    ANCHOR_BR = 2      // 1:2:2:2:1
};

struct Anchor {
    Point2d pt;      // 中心点
    AnchorType type; // 类型
    double size;     // 尺寸
    int contourIdx;  // 轮廓索引
};

int COUNT = 0;

float get_ratio_error(const std::vector<uchar>& a, const std::vector<uchar>& b){
    if(a.size() != b.size()) return 999.0f;

    float sumA = 0, sumB = 0;
    for(auto& l : a) sumA += l;
    for(auto& l : b) sumB += l;
    if(sumA == 0 or sumB == 0) return 999.0f;
    
    float error = 0.0f;
    for(size_t i = 0; i < a.size(); ++i){
        float norm_a = (float)a[i] / sumA;
        float norm_b = (float)b[i] / sumB;
        error += std::abs(norm_a - norm_b);
    }
    return error;
}

// 四个角点排序
// 我讨厌 float，但 RotatedRect::points 居然不支持 Point2d 类型，所以不得不用 float
void sort_corner(Point2f* src, Point2f* dst){
    Point2f center(0, 0);
    for(int i = 0; i < 4; ++i) center += src[i];
    center *= 0.25;
    // sum = x + y, diff = y - x;
    // TL 是 sum 最小的, BR 是 sum 最大的, TR 是 diff 最小的, BL 是 diff 最大的
    float min_sum = 9e5, max_sum = -9e5;
    float min_diff = 9e5, max_diff = -9e5;
    int TL = 0, BR = 0, TR = 0, BL = 0;
    for(int i = 0; i < 4; ++i){
        float s = src[i].x + src[i].y;
        float d = src[i].y - src[i].x;
        if(s < min_sum){min_sum = s; TL = i;}
        if(s > max_sum){max_sum = s, BR = i;}
        if(d < min_diff){min_diff = d; TR = i;}
        if(d > max_diff){max_diff = d; BL = i;}
    }

    dst[0] = src[TL];
    dst[1] = src[TR];
    dst[2] = src[BR];
    dst[3] = src[BL];
}

// 规范化 ROI
Mat get_normalized_roi(const Mat& roi, RotatedRect rRect) {
    Point2f corners[4];
    rRect.points(corners);
    const float expansion = 1.5f; // 扩大 1.5 倍 
    for(int i = 0; i < 4; ++i){
        corners[i] -= rRect.center;
        corners[i] *= expansion;
        corners[i] += rRect.center;
    }

    Point2f src_pts[4];
    sort_corner(corners, src_pts);
    
    const float d = 64; // 目标尺寸
    Point2f dst_pts[4] = {
        Point2f(0, 0),
        Point2f(d, 0),
        Point2f(d, d),
        Point2f(0, d)
    };

    Mat M = getPerspectiveTransform(src_pts, dst_pts);
    Mat norm;
    warpPerspective(roi, norm, M, Size(d, d));
    
    // 变换过程似乎会产生灰度，再做个二值化
    threshold(norm, norm, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return norm;
}

AnchorType get_roi_type(const Mat& roi, const bool dir){
    // dir = 0 for Horizontal, 1 for Vertical
    int center = dir ? roi.cols >> 1 : roi.rows >> 1;
    int length = dir ? roi.cols : roi.rows;
    
    if(!(dir & 1)) ++COUNT;

    // 白 - 黑 - 白 - 黑 -白
    // 自创神秘扫描法：在区域块中心，扫宽度为 5 的条带上的像素数量
    int BAND = 5;
    int start = std::max(center - BAND / 2, 0);
    int diff_cnt = 0;
    std::vector<std::vector<std::pair<uchar, uchar>>> lens(BAND);
    std::vector<uchar> last_color(BAND), count(BAND); // roi 不超过 100x100，不担心溢出
    
    auto get_color = [&](int i, int offset) -> uchar {
        int r, c;
        if(dir == 0){
            r = start + offset;
            c = i;
        }
        else{
            c = start + offset;
            r = i;
        }
        if(r < 0 or r >= roi.rows or c < 0 or c >= roi.cols) return 0;
        return roi.at<uchar>(r, c) > 127 ? 1 : 0;
    };

    for(int i = 0; i < length; ++i){

        bool diff = false;
        uchar color0 = get_color(i, 0);

        for(int j = 0; j < BAND; ++j){
           
            uchar color = get_color(i, j);
            if(color != color0) diff = true;
            
            if(i == 0){
                last_color[j] = color;
                count[j] = 1;
                continue;
            }
            if(color == last_color[j]) ++count[j];
            else{
                lens[j].emplace_back(count[j], color ^ 1);
                count[j] = 1;
                last_color[j] = color;
            }
        }
        if(diff) ++diff_cnt;
        if((double)diff_cnt / length > 0.3){
          //  imwrite("./roi/roi" + std::to_string(COUNT) + ".png" , roi);
          //  printf("ROI %u fails in diff test.\n", COUNT);
            return ANCHOR_NONE;
        }
    }
    for(int j = 0; j < BAND; ++j) if(count[j] > 1) lens[j].emplace_back(count[j], last_color[j]);
    
    int min_size = 114514;
    for(int j = 0; j < BAND; ++j) min_size = std::min(min_size, (int)lens[j].size());
    if(min_size < 5){
     //   imwrite("./roi/roi" + std::to_string(COUNT) + ".png" , roi);
     //   printf("ROI %u fails in min-size test.\n", COUNT);
        return ANCHOR_NONE;
    }
    
    // BAND 段都有 白-黑-白-黑-白 且比例正确的特征
    // 放松条件，80% 的 band 符合这个特征即可？
    const float ERROR_THRESHOLD = 0.4f;

    auto get_band_type = [&](const std::vector<std::pair<uchar, uchar>>& band) -> AnchorType {
        if(band.size() < 5) return ANCHOR_NONE;

        for(size_t i = 0; i <= band.size() - 5; ++i){
            if(!band[i].second) continue;
            std::vector<uchar> vec;
            for(int j = 0; j < BAND; ++j) vec.push_back(band[i + j].first);
            auto err = get_ratio_error(vec, {1, 1, 4, 1, 1});
            if(err < ERROR_THRESHOLD) return ANCHOR_NORMAL;
            err = get_ratio_error(vec, {1, 2, 2, 2, 1});
            if(err < ERROR_THRESHOLD) return ANCHOR_BR;
        }
        return ANCHOR_NONE;
    };

    std::vector<uchar> type_count(3);
    for(int i = 0; i < BAND; ++i) ++type_count[get_band_type(lens[i])];
    for(int i = 1; i <= 2; ++i){
        if((double)type_count[i] / BAND >= 0.6) return (AnchorType)i;
    }
 //   imwrite("./roi/roi" + std::to_string(COUNT) + ".png" , roi);
   // printf("ROI %u fails in type test: (%d, %d, %d).\n", COUNT, type_count[0], type_count[1], type_count[2]);
    return ANCHOR_NONE;
}

class Scanner {

public:
    static void threshold_fast(const Mat& img, Mat& out){
        // reserved
        threshold(img, out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    static void threshold_adaptive(const Mat& img, Mat& out){
        // 自适应二值化，应对光照不均
     //   unsigned int unit = std::min(img.cols, img.rows);
     //   unit = get_block_size(unit * 0.05);
     //   adaptiveThreshold(img, out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, unit, -10);
        int blockSize = std::min(img.cols, img.rows) / 20;
        if(blockSize % 2 == 0) ++blockSize;
        adaptiveThreshold(img, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, 5);
    }

    inline void preprocess(const Mat& img, Mat& out, const bool fast = true){
        Mat tmp;
        // 灰度图
        if(img.channels() >= 3) cvtColor(img, tmp, COLOR_BGR2GRAY);
        else tmp = img.clone();

        /*
        // 高斯模糊去噪点
        unsigned int unit = std::min(img.cols, img.rows);
        // wtf what's this param?
        unit = std::max(get_block_size((unsigned)(unit * 0.002)), 3U);
        GaussianBlur(tmp, tmp, Size(unit, unit), 0);
        */
        // 二值化
        if(fast) threshold_fast(tmp, out);
        else threshold_adaptive(tmp, out);

        imwrite("preprocess.png", out);
    }

    std::vector<Anchor> get_anchors(Mat& src){
        Mat img;
        preprocess(src, img);
    
        // 查找轮廓
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        printf("coutours: %u\n", contours.size());
        std::vector<Anchor> candidates;

        // 筛选
        for(int i = 0; i < contours.size(); ++i){
            // 1. 必须有子轮廓
            int child = hierarchy[i][2];
            if(child == -1) continue;
            // 2. 必须有孙子轮廓
         //   int grandChild = hierarchy[child][2];
         //   if(grandChild == -1) continue;
           
            // 3. 孙子轮廓没有子轮廓
            // if(hierarchy[grandChild][2] != -1) continue;
            
            // 4. 简单的几何验证
            RotatedRect rRect = minAreaRect(contours[i]);
            // 太小或太大的不要
            if(rRect.size.width < 30 or rRect.size.height < 30) continue;
            if(rRect.size.width > 100 or rRect.size.height > 100) continue;
            
            Mat roi = get_normalized_roi(img, rRect);
            
            AnchorType typeX = get_roi_type(roi, 0);
            AnchorType typeY = get_roi_type(roi, 1);

            if(typeX == typeY and typeX != ANCHOR_NONE){
                Anchor anchor;
                anchor.pt = rRect.center;
                anchor.type = typeX;
                anchor.size = (rRect.size.width + rRect.size.height) / 2.0f;
                anchor.contourIdx = i;
                candidates.push_back(anchor);
            }
        }
        deduplicate(candidates);
        return candidates;
    }

    // 去重逻辑：合并同心圆，优先保留大的
    void deduplicate(std::vector<Anchor>& cand){
        std::vector<Anchor> res;
        std::vector<bool> used(cand.size());
        for(size_t i = 0; i < cand.size(); ++i){
            if(used[i]) continue;
            Anchor best = cand[i];
            used[i] = true;
            for(size_t j = i + 1; j < cand.size(); ++j){
                if(used[j]) continue;
                if(norm(cand[i].pt - cand[j].pt) < cand[i].size * 0.5){
                    if(cand[j].size > best.size) best = cand[j];
                    used[j] = true;
                }
            }
            res.push_back(best); 
        }
        cand.swap(res);
    }

    void draw_anchors(Mat& img, const std::vector<Anchor>& anchors){
        std::vector<Anchor> norms, brs;
        for(const auto& anchor : anchors){
            if(anchor.type == ANCHOR_NORMAL) norms.push_back(anchor);
            else if(anchor.type == ANCHOR_BR) brs.push_back(anchor);
        }
        printf("Detected Normal: %u\n", norms.size());
        printf("Detected BR: %u\n", brs.size());

        Mat canvas = img.clone();
        for(const auto& anchor : anchors){
            Scalar color = (anchor.type == ANCHOR_BR) ? Scalar(0,0,255) : Scalar(0,255,0);
            circle(canvas, anchor.pt, anchor.size / 2, color, 4);
        }
        imwrite("anchors.png", canvas);
    }

private:
    static inline unsigned get_block_size(unsigned v){
        // 返回 >= v 的最小的 2^p + 1
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return std::max(3U, v + 2);
    }

};

int main(const int argc, const char* argv[]){
    if(argc < 2) return -1;
    Mat srcImg = imread(argv[1]);
    Scanner scanner;
    auto anchors = scanner.get_anchors(srcImg);
    scanner.draw_anchors(srcImg, anchors);
    return 0;
}