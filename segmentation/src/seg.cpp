#include <opencv2/cudaoptflow.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp> // DIS optical flow

#include "../include/seg.h"
#include "../include/util.h"

SegmentorNew::SegmentorNew() {
}

void SegmentorNew::Put(cv::Mat outline_edges, cv::Mat valid_mask) {
  int n_octave = 6;
  int n_downsample =2;
  NEW::Segment(outline_edges, n_octave, n_downsample, marker_);
}

ImageTrackerNew::ImageTrackerNew() {
  n_instance_ = 0;
  dof_ = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
}

struct MarkerStats{
  size_t samples_;
  std::map<int,size_t> snyc_n_;

};

void ImageTrackerNew::Put(const cv::Mat _gray,
                          const cv::Mat _unsync_marker,
                          float sync_min_iou) {
  std::map<int, int> unsync2sync;
  cv::Size size = _gray.size();
  int i, j, u, v;
  if(! prev_gray_.empty()){
    // optical flow etime : 6~7 milli sec
    cv::Mat flowxy;
    dof_->calc(prev_gray_, _gray, flowxy);
    cv::split(flowxy, flow_);
    
    std::map<int, MarkerStats > unsync2stats;
    std::map<int, size_t> sync_smaples;
    float* flow_u = flow_[0].ptr<float>();
    float* flow_v = flow_[1].ptr<float>();
    int32_t* prev_sync_l = prev_sync_marker_.ptr<int32_t>();
    const int32_t* unsync_l= _unsync_marker.ptr<int32_t>();

    int offset = 4;
    const int offset_step = offset * size.width;
    const int ho = size.height-offset;
    const int wo = size.width-offset;
    for( i = 0; i < ho; i+=offset) {
      for(j = 0; j < wo; j+=offset){
        u = j + int( flow_u[j] );
        v = i + int( flow_v[j] );
        const int32_t& l0 = prev_sync_l[j];
        if(l0 < 1)
          continue;
        if(u > 0 && v > 0 && u < size.width && v < size.height){
          int index = v * size.width + u;
          MarkerStats& stats = unsync2stats[unsync_l[index]];
          stats.samples_++;
          stats.snyc_n_[l0]++;
          sync_smaples[l0]++;
        }
      }
      flow_u += offset_step;
      flow_v += offset_step;
      prev_sync_l += offset_step;
    }

    for(const auto& it : unsync2stats){
      const int& ul = it.first;
      size_t overlap_smaples = 0;
      int sl = -1;
      for(auto it2 : it.second.snyc_n_){
        if(overlap_smaples < it2.second){
          overlap_smaples = it2.second;
          sl = it2.first;
        }
      }
      size_t s_samples = sync_smaples.at(sl);
      float iou = float(overlap_smaples)  / float(it.second.samples_ + s_samples - overlap_smaples);
      if(iou > sync_min_iou)
        unsync2sync[ul] = sl;
    } // unsync2sync
  }
  else{
    // if prev_gray empty
    prev_sync_marker_ = cv::Mat::zeros(size, CV_32SC1);
  }

  int32_t*  curr_sync_l  = prev_sync_marker_.ptr<int32_t>();
  const int32_t*  unsyc_l = _unsync_marker.ptr<int32_t>();
  for(i = 0; i < size.height; i++) {
      for(j = 0; j < size.width; j++){
        const int32_t& ul = unsyc_l[j];
        int32_t* curr_l = curr_sync_l+j;
        if(unsync2sync.count(ul))
          *curr_l = unsync2sync[ul];
        else{
          unsync2sync[ul] = ++n_instance_;
          *curr_l = n_instance_;
        }
      }
      unsyc_l     += size.width;
      curr_sync_l += size.width;
  }

  //cv::imshow("flow label", GetColoredLabel(sync_marker));
  //cv::imshow("curr label", GetColoredLabel(_unsync_marker));
  prev_gray_ = _gray;
  return;
}
void Shape::UpdateBB() {
  {
    cv::Point2f x0(9999999.f,9999999.f);
    cv::Point2f x1(-x0);
    for(const auto& pt : outerior_){
      x0.x = std::min<float>(x0.x, pt.x);
      x0.y = std::min<float>(x0.y, pt.y);
      x1.x = std::max<float>(x1.x, pt.x);
      x1.y = std::max<float>(x1.y, pt.y);
    }
    outerior_bb_ = cv::Rect2f(x0.x,x0.y,x1.x-x0.x,x1.y-x0.y);
  }
  return;
}

static bool BbCollision(const cv::Rect2f& bb, const float& x, const float& y) {
  //bb.width;
  if(x < bb.x )
    return false;
  if(x > bb.x+bb.width)
    return false;
  if(y < bb.y)
    return false;
  if(y > bb.y+bb.width)
    return false;
  return true;
}

bool Shape::HasCollision(const int& _x, const int& _y, bool check_contour) const {
   
  // *[x] global shape에 대해 BB 충돌체크.
  // *[x] contour에 대해 충돌체크
  const float x = _x;
  const float y = _y;
  cv::Point2f pt(x,y);

  if( !BbCollision(outerior_bb_, x, y) )
    return false;
  if(!check_contour)
    return true;

  // Interior contour과 충돌 없는 경우, outerior contour와 충돌체크
  bool b_outerior_collision = cv::pointPolygonTest(outerior_, pt, false) > 0.;
  return b_outerior_collision;
}


bool IsInFrame(const cv::Size size, const cv::Point2f& pt, const float boundary){
  if(pt.x < boundary)
    return false;
  if(pt.y < boundary)
    return false;
  if(pt.x > size.width-boundary)
    return false;
  if(pt.y > size.height-boundary)
    return false;
  return true;
}



