#ifndef SEG_VIEWER_H_
#define SEG_VIEWER_H_
#include "stdafx.h"
//#include "common.h"
#include <g2o/types/slam3d/se3quat.h>
#include <pangolin/pangolin.h>
#include <mutex>

class SegViewer{
public:
  SegViewer(const EigenMap<int, g2o::SE3Quat>& gt_Tcws, std::string config_fn);
  void Run();
  void Join();
  void SetFrame(int k); // 이거 외에도 그려야할 모든 정보를 여기서 넘겨야겠다.
private:
  void DrawPose(const g2o::SE3Quat& Twc);
  const EigenMap<int, g2o::SE3Quat> gt_Tcws_;

  cv::Size size_;
  float vp_f_, z_near_, z_far_, ex_, ey_, ez_, lx_, ly_, lz_, fps_;
  const std::string name_;
  std::mutex mutex_viewer_;
  int curr_k_;
  std::thread  thread_;
};

#endif
