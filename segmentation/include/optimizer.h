#ifndef SEG_OPTIMIZER_
#define SEG_OPTIMIZER_
//#include "camera.h"
#include "segslam.h"
#include <g2o/core/sparse_optimizer.h>

namespace seg {

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

class Mapper {
public:
  Mapper();
  ~Mapper();
  void ComputeLBA(const Camera* camera,
                  Qth qth,
                  const std::set<Mappoint*>& neighbor_mappoints,
                  const std::map<Jth, Frame*>& neighobor_frames,
                  Frame* curr_frame,
                  EigenMap<Jth, g2o::SE3Quat> & kf_Tcqs,
                  g2o::SE3Quat& curr_Tcq,
                  bool verbose
                 );
private:
  g2o::OptimizableGraph::Vertex* CreatePoseVertex(g2o::SparseOptimizer& optimizer,
                                                  Frame* frame,
                                                  const g2o::SE3Quat& Tcq);
  g2o::OptimizableGraph::Vertex* CreateStructureVertex(Qth qth,
                                                       g2o::SparseOptimizer& optimizer,
                                                       Mappoint* mpt);
};

} // namespace seg
#endif
