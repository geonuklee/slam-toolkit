#ifndef SEGMENT_UTIL_H_
#define SEGMENT_UTIL_H_
#include <opencv2/opencv.hpp>
#include "camera.h"

cv::Mat VisualizeFlow(const std::vector<cv::Mat>& flow, cv::Mat bgr=cv::Mat() );
void GetExpectedFlow(const Camera& camera, const g2o::SE3Quat& Tc0c1, const cv::Mat depth,
                     cv::Mat& exp_flow, cv::Mat& valid_mask);
cv::Mat GetDisparity(cv::cuda::GpuMat g_gray, cv::cuda::GpuMat g_gray_r);
cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);
cv::Mat GetBoundary(const cv::Mat marker, int w=1);

extern std::vector<cv::Scalar> colors;

#endif
