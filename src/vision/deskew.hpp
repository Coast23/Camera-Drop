#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vision/types.hpp"

namespace camdrop::vision {

class Deskewer {
public:
    static cv::Mat Deskew(const cv::Mat& frame,
                          const CornerQuad& corners,
                          int out_width = kImageWidth,
                          int out_height = kImageHeight,
                          float expand = 1.0f,
                          int canonical_inset = 2,
                          int interpolation = cv::INTER_NEAREST);
};

}  // namespace camdrop::vision
