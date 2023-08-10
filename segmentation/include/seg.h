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
  bool stabilized_;
  float area_;
  std::vector<cv::Point2f> outerior_;
  cv::Rect2f outerior_bb_;
  Shape (): label_(-1), n_missing_(0), n_matching_(0), n_belief_(0), stabilized_(false),
  area_(-1.){
  }
  void UpdateBB();
  bool HasCollision(const int& x, const int& y, bool check_contour) const;
};

class Segmentor {
public:
  Segmentor();
  const std::map<int, ShapePtr>& Put(cv::Mat gray, cv::Mat depth, const DepthCamera& camera, cv::Mat vis_rgb, cv::Mat& flow0);
  const std::map<int, ShapePtr>& Put(cv::Mat gray, cv::Mat gray_r, const StereoCamera& camera, cv::Mat vis_rgb, cv::Mat& flow0);

private:

  const std::map<int, ShapePtr>& _Put(cv::Mat gray,
            cv::cuda::GpuMat g_gray,
            cv::Mat depth,
            const Camera& camera,
            cv::Mat vis_rgb, // for visualization
            cv::Mat& flow0
            );


  bool IsKeyframe(cv::Mat flow, cv::Mat rgb = cv::Mat());
  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray);

  cv::Mat GetFlow(cv::cuda::GpuMat g_gray);

private:
  cv::Ptr<cv::cuda::DenseOpticalFlow> optical_flow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;

  std::map<int, ShapePtr> global_shapes_; // 최근까지 제대로 추적되던 instance의 모음.
  int n_shapes_;
};

cv::Mat Segment(const cv::Mat outline_edge,
                cv::Mat valid_mask=cv::Mat() ,
                bool limit_expand_range=true,
                cv::Mat rgb4vis=cv::Mat() );

void DistanceWatershed(const cv::Mat dist_fromedge,
                       cv::Mat& markers,
                       bool limit_expand_range,
                       cv::Mat& vis_arealimitedflood,
                       cv::Mat& vis_rangelimitedflood,
                       cv::Mat& vis_onedgesflood
                       );

std::map<int, ShapePtr> ConvertMarker2Instances(const cv::Mat marker);
std::map<int,int> TrackShapes(const std::map<int, ShapePtr>& local_shapes,
                              const cv::Mat& local_marker,
                              const cv::Mat& flow0,
                              const float min_iou,
                              std::map<int, ShapePtr>& global_shapes,
                              int& n_shapes);


#endif
