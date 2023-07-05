#ifndef EPIPCLUSTER_TRACKER_H_
#define EPIPCLUSTER_TRACKER_H_
#include "stdafx.h"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

class DenseTracker {
  struct EpipPoint{
    cv::Point2f img0_;
    cv::Point2f img1_;
    cv::Point3f obj1_;
    int cluster_;
  };

public:
  DenseTracker(const Eigen::Matrix<double, 3,3>& K, const float base);

  void Track(cv::Mat gray, cv::Mat gray_r);

private:
  cv::Mat GetEdge(const cv::Mat gray) const;

  cv::Mat GetFlow(const cv::cuda::GpuMat g_gray,
                  const cv::cuda::GpuMat g_gray_r) const;

  template<typename T>
  void SamplePoints(int stride,
                    const cv::Mat flow,
                    const cv::Mat disparity1,
                    const cv::Mat depth1,
                    std::map<int, EpipPoint>& epip_points
                    ) const;

  void TrackCluster(const cv::Mat flow,
                    const cv::Mat disparity1,
                    const cv::Mat depth0,
                    const cv::Mat depth1,
                    std::set<int>& unclustered,
                    std::map<int, EpipPoint>& epip_points
                   );

  void RansacCluster(const cv::Mat flow,
                     const cv::Mat disparity1,
                     const cv::Mat depth0,
                     const cv::Mat depth1,
                     const std::set<int>& unclustered,
                     std::map<int, EpipPoint>& epip_points
                    );

  cv::Mat MakeMask(const int stride, 
                   const int rows,
                   const int cols,
                   const std::map<int, EpipPoint>& epip_points) const;

  void EuclideanFilter(std::map<int, EpipPoint >&  epip_points,
                       std::set<int>& tracked_points);

  const float base_;
  const cv::Mat cvK_;

  int n_cluster_;

  std::map<int, std::vector<EpipPoint> > clusters_;
  std::map<int, bool> cluster_ground_;

  cv::Ptr<cv::cuda::DenseOpticalFlow> opticalflow_;

#if 0
  template<typename T>
  cv::Mat GetDisparity(const cv::Mat gray,
                       const cv::Mat gray_r,
                       const cv::Mat edge) const;

  cv::Ptr<cv::StereoMatcher> left_matcher_; // StereoSGBM
  cv::Ptr<cv::StereoMatcher> right_matcher_;
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
#else
  template<typename T>
  cv::Mat GetDisparity(const cv::cuda::GpuMat& g_gray,
                       const cv::cuda::GpuMat& g_gray_r,
                       const cv::Mat edge) const;

  cv::Ptr<cv::cuda::StereoBM> left_matcher_; // StereoSGBM
#endif

  cv::Mat gray0_, depth0_;
  cv::Mat mask0_;
  cv::cuda::GpuMat g_gray0_;
};


#endif
