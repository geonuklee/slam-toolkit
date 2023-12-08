#ifndef OCCLUSION_SEGMENT_SEG_H_
#define OCCLUSION_SEGMENT_SEG_H_
#include "stdafx.h"

class TotalSegmentor {
public:
  TotalSegmentor(const float fx, const float fy) : fx_(fx), fy_(fy) { }
  void Put(cv::Mat depth);

private:
  cv::Mat GetEdge(const cv::Mat depth, const cv::Mat invmap, const uchar FG, const uchar BG, const uchar CE,
                  float dd_so, float ce_so, float min_obj_width, float hessian_th
                  ) const;
  void Merge(const cv::Mat& depth,
             const cv::Mat& edge,
             cv::Mat& _marker, float min_direct_contact_ratio_for_merge, bool keep_boundary) const;
  int Dijkstra(cv::Mat& _marker) const;
  const float fx_, fy_;
};

class OutlineEdgeDetectorWithSizelimit {
public:
  enum Direction { VERTICAL=1, HORIZONTAL=2, BOTH=VERTICAL|HORIZONTAL };
  enum DdType    { FG=1, BG=2 };

  OutlineEdgeDetectorWithSizelimit() { }
  void PutDepth(const cv::Mat depth, float fx, float fy);
  cv::Mat GetOutline() const { return outline_edges_; }
  cv::Mat GetDDEdges() const { return dd_edges_; }
  cv::Mat GetConcaveEdges() const { return concave_edges_; }
private:
  cv::Mat ComputeDDEdges(const cv::Mat depth) const;
  cv::Mat ComputeConcaveEdges(const cv::Mat depth, const cv::Mat dd_edges, float fx, float fy) const;

  cv::Mat dd_edges_;
  cv::Mat concave_edges_;
  cv::Mat outline_edges_;
};

cv::Mat MergeOcclusion(const cv::Mat dd_edges, const cv::Mat _marker);
cv::Mat MergeOcclusion(const cv::Mat depth, const cv::Mat dd_edges, const cv::Mat _marker);

/*
  TODO 그냥 바닥면 인식과 euclidean cluster를 설정하자.

*/

#endif
