#ifndef SEG_H_
#define SEG_H_

#include "stdafx.h"
#include "camera.h"

class Seg {
public:
  Seg();
  bool IsKeyframe(cv::Mat flow, cv::Mat rgb = cv::Mat());
  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray, const g2o::SE3Quat& Tcw);
  void Put(cv::Mat gray, cv::Mat gray_r, const g2o::SE3Quat& Tcw, const StereoCamera& camera);

private:
  cv::Ptr<cv::cuda::DenseOpticalFlow> optical_flow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;
  g2o::SE3Quat Tc0w_;
  std::vector<cv::Point2f> corners0_;

};


#endif
