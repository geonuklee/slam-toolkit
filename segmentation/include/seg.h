#ifndef SEGMENT_SEG_H_
#define SEGMENT_SEG_H_

#include "stdafx.h"
#include "camera.h"


class Shape;
typedef std::shared_ptr<Shape> ShapePtr;
class Shape {
public:
  int label_;
  int n_missing_;
  int n_matching_;
  int n_belief_;
  int stabilized_;
  std::vector<cv::Point2f> outerior_;
  cv::Rect2f outerior_bb_;
  Shape (): label_(-1), n_missing_(0), n_matching_(0), n_belief_(0) {
  }
  void UpdateBB();
  bool HasCollision(const int& x, const int& y, bool check_contour) const;
};

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
  //cv::Mat marker0_;
  cv::cuda::GpuMat g_gray0_;
};

cv::Mat FlowDifference2Edge(cv::Mat score);
cv::Mat FlowError2Edge(const cv::Mat flow_errors, const cv::Mat expd_diffedges, const cv::Mat valid_mask);

cv::Mat Segment(const cv::Mat outline_edge, const cv::Mat rgb4vis=cv::Mat() );
void DistanceWatershed(const cv::Mat dist_fromedge,
                       cv::Mat& markers,
                       cv::Mat& vis_arealimitedflood,
                       cv::Mat& vis_rangelimitedflood,
                       cv::Mat& vis_onedgesflood
                       );

#endif
