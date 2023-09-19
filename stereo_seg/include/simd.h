#ifndef SIMD_IMGPROC_
#define SIMD_IMGPROC_
#include <immintrin.h> // Required for SSE, AVX, and AVX2
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
  ImageTrackerNew() {}
  cv::Mat GetFlow(const cv::Mat gray); // delta uv {0}<-{1} on coordinate {0}.
  cv::Mat GetLogoddsOutline(const cv::Mat& outline_curr);

private:
  cv::Mat gray0_;
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

} // namespace OLD

namespace NEW {
void Segment(const cv::Mat outline_edges, 
             int n_octave,
             int n_downsample,
             cv::Mat& output);
} //namespace NEW


cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);


#endif
