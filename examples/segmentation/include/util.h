#include <opencv2/opencv.hpp>
#include "camera.h"

cv::Mat VisualizeFlow(cv::Mat flow, cv::Mat bgr=cv::Mat() );
cv::Mat convertDiscrete(cv::Mat input, float istep, int ostep);
cv::Mat convertMat(cv::Mat input, float lower_bound, float upper_bound);
cv::Mat GetDifference(cv::Mat flow, cv::Mat disparity);
cv::Mat GetDivergence(cv::Mat flow);
cv::Mat GetExpectedFlow(const StereoCamera& camera, const g2o::SE3Quat& Tc0c1, cv::Mat disp1);
cv::Mat GetFlowError(const cv::Mat flow, const cv::Mat exp_flow);
cv::Mat GetDisparity(cv::cuda::GpuMat g_gray, cv::cuda::GpuMat g_gray_r);
