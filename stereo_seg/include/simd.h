#ifndef SIMD_IMGPROC_
#define SIMD_IMGPROC_
#include <immintrin.h> // Required for SSE, AVX, and AVX2
#include <opencv2/opencv.hpp>

cv::Mat GetDDEdges(const cv::Mat depth, bool with_simd=true);

#endif
