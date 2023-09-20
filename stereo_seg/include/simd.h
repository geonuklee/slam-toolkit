#ifndef SIMD_IMGPROC_
#define SIMD_IMGPROC_
#include <opencv2/opencv.hpp>

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

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);


#endif
