#ifndef SEG_H_
#define SEG_H_

#include "stdafx.h"
#include "camera.h"

class Seg {
public:
  Seg();
  bool IsKeyframe(cv::Mat flow, cv::Mat rgb = cv::Mat());
  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray);
  void Put(cv::Mat gray, cv::Mat gray_r, const StereoCamera& camera);

  cv::Mat GetFlow(cv::cuda::GpuMat g_gray);
  cv::Mat GetTextureEdge(cv::Mat gray);

  void NormalizeScale(const cv::Mat disparity, const cv::Mat flow_scale,
                      cv::Mat& flow_difference, cv::Mat& flow_errors);

  g2o::SE3Quat TrackTc0c1(const std::vector<cv::Point2f>& corners,
                          const cv::Mat flow,
                          const cv::Mat disparity,
                          const StereoCamera& camera);

private:
  cv::Ptr<cv::cuda::DenseOpticalFlow> optical_flow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;
};

cv::Mat FlowDifference2Edge(cv::Mat score);
cv::Mat FlowError2Edge(cv::Mat flow_errors);

cv::Mat DistanceWatershed(cv::Mat edges);
cv::Mat EdgeWatershed(const cv::Mat score);
#endif
