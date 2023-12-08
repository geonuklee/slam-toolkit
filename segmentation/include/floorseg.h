#ifndef FLOORSEG_H_
#define FLOORSEG_H_
#include <opencv2/opencv.hpp>

class Floorseg {
  // 바닥면을 제외한 points cloud에 대해 euclidean clustering
public:
  Floorseg(float fx, float fy, float cx, float cy, float base_line):  fx_(fx), fy_(fy), cx_(cx), cy_(cy),
  base_line_(base_line){
  }

  cv::Mat Put(const cv::Mat depth,
              const cv::Mat vis_rgb=cv::Mat());
  cv::Mat GetMarker() const { return marker_; }
private:
  const float fx_, fy_, cx_, cy_; // Ignore distortion.
  const float base_line_;
  cv::Mat marker_;

};


#endif
