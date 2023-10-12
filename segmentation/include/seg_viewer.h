#ifndef SEG_VIEWER_H_
#define SEG_VIEWER_H_
#include "stdafx.h"
//#include "common.h"
#include <g2o/types/slam3d/se3quat.h>
#include <pangolin/pangolin.h>
#include <mutex>

class SegViewer{
public:
  SegViewer(const EigenMap<int, g2o::SE3Quat>& gt_Tcws, std::string config_fn, cv::Size dst_size,
            std::string viewer_title);
  void Run();
  void Join(bool req_exit);
  bool IsShutDowned() const {
    std::unique_lock<std::mutex> lock(mutex_viewer_);
    return req_exit_;
  }

  void SetCurrCamera(int k,
                     const EigenMap<int, g2o::SE3Quat>& updated_Tcws,
                     const cv::Mat& dst);
  void SetMappoints(const EigenMap<int, Eigen::Vector3d>& mappoints);

private:
  void DrawPoints();
  void DrawPose(const g2o::SE3Quat& Twc);
  void DrawTrajectories(const EigenMap<int,g2o::SE3Quat>& est_Tcws);

  const EigenMap<int, g2o::SE3Quat> gt_Tcws_;

  EigenMap<int, g2o::SE3Quat> est_Tcws_;
  EigenMap<int, g2o::SE3Quat> updated_Tcws_;

  EigenMap<int, Eigen::Vector3d> all_mappoints_;
  EigenMap<int, Eigen::Vector3d> curr_mappoints_;

  cv::Size dst_size_;
  cv::Size trj_size_;
  float vp_f_, z_near_, z_far_, ex_, ey_, ez_, lx_, ly_, lz_, ux_, uy_, uz_, fps_;
  const std::string name_;
  mutable std::mutex mutex_viewer_;
  int curr_k_;
  cv::Mat curr_dst_;
  int is_kf_;
  bool req_exit_;
  std::thread  thread_;
};

void TestPangolin(int argc, char** argv);

#endif
