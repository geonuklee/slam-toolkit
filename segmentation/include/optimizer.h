#ifndef SEG_OPTIMIZER_
#define SEG_OPTIMIZER_
#include "segslam.h"
namespace seg {

class Mapper {
public:
  Mapper();
  ~Mapper();
  void ComputeLBA(const std::set<Mappoint*>& mappoints,
                  const std::map<Jth, Frame*>& frames,
                  const Frame* curr_frame,
                  EigenMap<Jth, g2o::SE3Quat> & kf_Tcqs,
                  g2o::SE3Quat& Tcq
                 );
};

} // namespace seg
#endif
