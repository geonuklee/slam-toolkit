#ifndef SEG_OPTIMIZER_
#define SEG_OPTIMIZER_
#include "camera.h"
#include "segslam.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/se3quat.h>

struct Param {
public:
  Param(const Camera* camera){
    const auto& K = camera->GetK();
    fx = K(0,0);
    fy = K(1,1);
  }
  float fx;
  float fy;
};

namespace NEW_SEG {
class PoseTracker {
public:
  PoseTracker();
  ~PoseTracker();
  g2o::SE3Quat GetTcq(const Camera* camera,
                      Qth qth,
                      Frame* curr_frame,
                      bool vis_verbose
                     );

};

double ChiSquaredThreshold(double p, double dof);

// ICP method
g2o::SE3Quat EstimateTcp(const std::vector<cv::Point3f>& Xp,
                         const std::vector<cv::Point3f>& vec_uvz_curr,
                         const Camera* camera, double uv_info, double invd_info, double delta,
                         std::vector<double>& vec_chi2);

class Mapper {
public:
  Mapper() { }
  ~Mapper() { };
  std::map<Pth,float> ComputeLBA(const Camera* camera,
                        Qth qth,
                        const std::set<Mappoint*>& neighbor_mappoints,
                        const std::map<Jth, Frame*>& neighbor_keyframes,
                        Frame* curr_frame,
                        Frame* prev_frame,
                        const std::set<Pth>& fixed_instances,
                        const cv::Mat& gradx,
                        const cv::Mat& grady,
                        const cv::Mat& valid_grad,
                        bool vis_verbose
                        );
} ;// class Mapper (NEW)

} // namespace NEW_SEG
#endif
