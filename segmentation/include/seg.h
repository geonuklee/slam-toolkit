#ifndef SEGMENT_SEG_H_
#define SEGMENT_SEG_H_

#include "stdafx.h"

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

class OutlineEdgeDetector {
public:
  virtual void PutDepth(cv::Mat depth, float fx, float fy) = 0;

  cv::Mat GetGradx() const { return gradx_; }
  cv::Mat GetGrady() const { return grady_; }
  cv::Mat GetOutline() const { return outline_edges_; }
  cv::Mat GetValidMask() const { return valid_mask_; }
  

protected:
  cv::Mat gradx_, grady_, outline_edges_, valid_mask_;

};

class OutlineEdgeDetectorWithoutSIMD : public OutlineEdgeDetector {
public:
  OutlineEdgeDetectorWithoutSIMD() { }
  virtual void PutDepth(cv::Mat depth, float fx, float fy);
};

class OutlineEdgeDetectorWithSIMD : public OutlineEdgeDetector {
public:
  OutlineEdgeDetectorWithSIMD() { }
  virtual void PutDepth(cv::Mat depth, float fx, float fy);
private:
  cv::Mat dd_edges_; // assume constant shape.
  cv::Mat concave_edges_;
};

class Segmentor {
public:
  virtual void Put(cv::Mat outline_edges, cv::Mat valid_mask) = 0;
  cv::Mat GetMarker() const { return marker_; }
public:
  cv::Mat marker_;
};

class SegmentorOld : public Segmentor {
  public:
  virtual void Put(cv::Mat outline_edges, cv::Mat valid_mask);
};

class SegmentorNew : public Segmentor {
public:
  SegmentorNew();
  virtual void Put(cv::Mat outline_edges, cv::Mat valid_mask);
};

class ImageTrackerOld {
public:
  ImageTrackerOld();
  virtual void Put(cv::Mat gray, cv::Mat marker);
  cv::Mat GetFlow0() const { return flow0_; }
private:
  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray);
  cv::Mat GetFlow(cv::cuda::GpuMat g_gray);
  cv::Ptr<cv::cuda::DenseOpticalFlow> optical_flow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;
  std::map<int, ShapePtr> global_shapes_; // 최근까지 제대로 추적되던 instance의 모음.
  cv::Mat flow0_;
  int n_shapes_;
};

class ImageTrackerNew {
public:
  ImageTrackerNew();
  virtual void Put(const cv::Mat gray,
                   const cv::Mat unsync_marker,
                   float sync_min_iou);
  const std::vector<cv::Mat>& GetFlow() const { return flow_; }  // delta uv {0}<-{1} on coordinate {0}.
  cv::Mat GetSyncedMarked() const { return prev_sync_marker_; }
private:
  int n_instance_;
  cv::Ptr<cv::DenseOpticalFlow> dof_;
  std::vector<cv::Mat> flow_;
  cv::Mat prev_gray_;
  cv::Mat prev_sync_marker_;
};

namespace OLD{
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
                              const float boundary,
                              std::map<int, ShapePtr>& global_shapes,
                              int& n_shapes);

} // namespace OLD

namespace NEW {
void Segment(const cv::Mat outline_edges, 
             int n_octave,
             int n_downsample,
             cv::Mat& output);
} //namespace NEW

#endif
