#include <opencv2/cudaoptflow.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp> // DIS optical flow

#include "../include/seg.h"
#include "../include/util.h"

SegmentorNew::SegmentorNew() {
}

void SegmentorNew::Put(cv::Mat outline_edges) {
  int n_octave = 6;
  int n_downsample = 1;
  bool keep_boundary = true;
  NEW::Segment(outline_edges, n_octave, n_downsample, keep_boundary, marker_);
}

cv::Mat GetInvalidDepthMask(const cv::Mat gray, double range) {
  // Apply Sobel operator in the x-direction to highlight vertical edges
  cv::Mat grad_x;
  cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
  // Convert the result to CV_8U
  cv::Mat abs_grad_x;
  cv::convertScaleAbs(grad_x, abs_grad_x);
  // Threshold the result to get edges with value 1 and non-edges with value 0
  cv::Mat v_edges;
  cv::threshold(abs_grad_x, v_edges, 100, 1, cv::THRESH_BINARY);
  // Convert the thresholded image to CV_8UC1
  v_edges.convertTo(v_edges, CV_8UC1);

  // Compute the distance transform
  cv::Mat dist;
  cv::distanceTransform(v_edges < 1, dist, cv::DIST_L2, cv::DIST_MASK_3);
  // Create a mask where pixels with distance greater than max_distance have a value of 1
  cv::Mat invalid_depthmask;
  invalid_depthmask = dist > range;
  return invalid_depthmask;
}

ImageTrackerNew::ImageTrackerNew() {
  n_instance_ = 0;
  dof_ = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
  //dof_ = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_FAST);
  //dof_ = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_MEDIUM);
}

struct MarkerOverlapStats{
  size_t samples_;
  std::map<int,size_t> snyc_n_;
};

struct MarkerStats{
  int synced_label;
  size_t area;
  MarkerStats() : synced_label(-1), area(0) { }
};

void ImageTrackerNew::Put(const cv::Mat _gray,
                          const cv::Mat _unsync_marker,
                          float sync_min_iou) {
  std::map<int, MarkerStats> unsync2sync;
  cv::Size size = _gray.size();
  int i, j, u, v;
  if(! prev_gray_.empty()){
    // optical flow etime : 6~7 milli sec
    // ref) https://stackoverflow.com/questions/38131822/what-is-output-from-opencvs-dense-optical-flow-farneback-function-how-can-th
    cv::Mat flowxy;
    dof_->calc(prev_gray_, _gray, flowxy);
    cv::split(flowxy, flow_);
    
    std::map<int, MarkerOverlapStats > unsync2stats;
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
          MarkerOverlapStats& stats = unsync2stats[unsync_l[index]];
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
      if(iou > sync_min_iou){
        unsync2sync[ul].synced_label = sl;
      }
    } // for unsync2stats
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
        if( ul < 1){
          *curr_l = -1;
          continue;
        }
        MarkerStats& ms = unsync2sync[ul]; // map 참조 최소화.
        ms.area++;
        if(ms.synced_label > 0){
          *curr_l = ms.synced_label;
        }
        else{
          ms.synced_label = ++n_instance_;
          *curr_l = n_instance_;
        }
      }
      unsyc_l     += size.width;
      curr_sync_l += size.width;
  }

  marker_areas_.clear();
  for(auto it : unsync2sync)
    marker_areas_[it.second.synced_label] = it.second.area;
  prev_gray_ = _gray;
  return;
}

