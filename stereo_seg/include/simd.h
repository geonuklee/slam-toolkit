#ifndef SIMD_IMGPROC_
#define SIMD_IMGPROC_
#include <immintrin.h> // Required for SSE, AVX, and AVX2
#include <opencv2/opencv.hpp>

void GetDDEdges(const cv::Mat depth, cv::Mat& edges, bool with_simd=true);
void GetGrad(const cv::Mat depth, float fx, float fy, const cv::Mat valid_mask,
             const int offset,
             cv::Mat& gradx, cv::Mat& grady, cv::Mat& valid_grad, bool with_simd=true);
cv::Mat VisualizeGrad(const cv::Mat gradx, const cv::Mat grady);
void GetConcaveEdges(const cv::Mat& gradx,
                     const cv::Mat& grady,
                     const cv::Mat depth,
                     const cv::Mat valid_mask,
                     const int offset,
                     float fx, float fy,
                     float neg_hessian_threshold, // -100.f
                     cv::Mat& edges,
                     bool with_simd=true);

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);

namespace OLD {
cv::Mat Segment(const cv::Mat outline_edge,
                   cv::Mat valid_mask,
                   bool limit_expand_range,
                   cv::Mat rgb4vis);
}

namespace NEW {
void Segment(const cv::Mat outline_edges, cv::Mat& output);
}

#endif
