#ifndef SEG_FRAME_
#define SEG_FRAME_
#include <flann/flann.hpp> // include it before opencv
#include "stdafx.h"
#include "seg.h"

namespace ORB_SLAM2{
  class ORBextractor;
}

class Camera;
class DepthCamera; // TODO 'depth' camera only?

namespace seg {

typedef int Jth; // jth frame
typedef int Qth; // qth rig group
typedef int Pth; // pth instance
typedef int Ith; // ith mappoint

/* Camera coordinate 을 나타내는 클래스
기존 ORB Frame이 mappoints를 instance, group 정보 없이 한꺼번에 가지고 있는 반면,
SegFrame에서는 이를 instance별로 묶고, instance들을 묶은 RigidGroup을 attribute로 가진다.
'
*/
class Frame;
class RigidGroup;
class Instance {
public:
  Instance(Pth pth);
  Pth pth_;
  std::map<Qth,RigidGroup*> rig_groups_;
  std::map<Jth,Frame*>      visible_kfraems_; 
};

class Mappoint;
class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Frame(const Jth id, const cv::Mat vis_rgb=cv::Mat());
  ~Frame();

  int GetIndex(const Mappoint* mp) const;
  void ExtractAndNormalizeKeypoints(const cv::Mat gray,
                                    const Camera* camera,
                                    ORB_SLAM2::ORBextractor* extractor);
  void SetInstances(const std::map<Pth, ShapePtr>& shapes,
                    const std::map<Pth, Instance*>& instances
                    );
  void SetMeasuredDepths(const cv::Mat depth);

  std::set<int> SearchRadius(const Eigen::Vector2d& uv, double radius) const;
  std::vector<std::vector<int> > SearchRadius(const flann::Matrix<double>& points,
                                              double radius) const;

  const cv::Mat GetRgb() const { return rgb_; }
  const std::vector<cv::KeyPoint>& GetKeypoints()      const { return keypoints_; }
  const std::vector<Mappoint*>&    GetMappoints()      const { return mappoints_; }
  const std::vector<Instance*>&    GetInstances()      const { return instances_; }
  const std::vector<float>&        GetMeasuredDepths() const { return measured_depths_; }
  std::vector<Instance*>&          GetInstances() { return instances_; }
  void SetMappoint(Mappoint* mp, int kpt_index);
  const Eigen::Vector3d& GetNormalizedPoint(int index) const { return normalized_[index]; }

  void ReduceMem();
  const cv::Mat GetDescription(int i) const;
  const cv::KeyPoint& GetKeypoint(int i) const {  return keypoints_.at(i); }
  const float& GetDepth(int i) const { return measured_depths_.at(i); }
  const Jth GetId() const { return id_; }

private:
  std::vector<cv::KeyPoint>    keypoints_;
  EigenVector<Eigen::Vector3d> normalized_;
  std::vector<Mappoint*>       mappoints_;
  std::vector<Instance*>       instances_; // 사실 Pth가 Qth를 가리켜야속편한데, 이거때문에 어려워짐.
  std::vector<float>           measured_depths_;
  cv::Mat descriptions_;
  flann::Matrix<double> flann_keypoints_;
  flann::Index<flann::L2<double> >* flann_kdtree_;
  std::map<const Mappoint*, int> mappoints_index_;

  cv::Mat rgb_;
  const Jth id_;
};


/*
class Instance 
ShapePtr과 연결된 seg SLAM class.
Instance 사이의 ... 사이의...  필요없다.
*/

class Mappoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mappoint(Ith id,
           const std::set<Qth>& ref_rigs, 
           Frame* ref);
  const Ith& GetId() const { return id_; }

  /*
  void SetInvD(float invd);
  float GetDepth() const;
  */

  cv::Mat GetDescription() const; // ref에서의 desc
  int GetIndex() const { return id_; }
  Frame* GetRefFrame() const { return ref_; }

  void AddKeyframe(Qth qth, Frame* frame) { keyframes_[qth].insert(frame); }
  const std::map<Qth, std::set<Frame*> >& GetKeyframes() const { return keyframes_; }
  const std::set<Frame*>& GetKeyframes(Qth qth) const { return keyframes_.at(qth); }

  // Ref coordinate에서 본 Xr. Depth Camera의 Measurement
  void SetXr(const Eigen::Vector3d& Xr);
  const Eigen::Vector3d& GetXr() const;

  // Qth rigid coordinate 에 맵핑된 결과값.
  void SetXq(Qth qth, const Eigen::Vector3d& Xq) { Xq_[qth] = Xq; }
  const Eigen::Vector3d& GetXq(Qth qth) const { return Xq_.at(qth); }

public:
  static std::map<Qth, size_t> n_;

private:
  Frame* ref_;
  const Ith id_;

  std::map<Qth, std::set<Frame*> > keyframes_; // GetNeighbors에 필요
  Eigen::Vector3d Xr_; // Measurement
  EigenMap<Qth, Eigen::Vector3d> Xq_; // rig마다 독립된 Mapping값을 가진다.
};

struct RigidGroup {
  // Instance의 묶음 
public:
  RigidGroup(Qth qth);
  ~RigidGroup();
  void IncludeInstance(Instance* ins);
  Instance* GetBgInstance() const { return bg_instance_; }
  const Qth& GetId() const { return id_; }
  void SetLatestKeyframe(Frame* kf) { latest_kf_ = kf; }
  Frame* GetLatestKeyframe() const { return latest_kf_; }
private:
  Frame* latest_kf_;
  const Qth id_;
  Instance*const bg_instance_;
  std::map<Pth, Instance*> excluded_instances_;
  std::map<Pth, Instance*> included_instances_; // TODO Need?
};

class Mapper;
class Pipeline {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pipeline(const Camera* camera,
           ORB_SLAM2::ORBextractor*const extractor
          );
  ~Pipeline();
  void Put(const cv::Mat gray,
           const cv::Mat depth,
           const std::map<Pth, ShapePtr>& shapes,
           const cv::Mat vis_rgb=cv::Mat());

private:
  std::map<Qth,bool> NeedKeyframe(Frame* frame) const;
  // j -> q -> (p) -> i
  std::map<Qth, std::map<Jth, Frame*> >       keyframes_;  // jth {c}amera와 keypointt<->'i'th mappoint correspondence
  // Tcq(j) : {c}amera <- {q}th group for 'j'th frame
  // Jth-1 for 'latest frame - regardless whether keyframe
  EigenMap<Qth, EigenMap<Jth, g2o::SE3Quat> > kf_Tcqs_;    
  Frame*                      prev_frame_;
  EigenMap<Qth, g2o::SE3Quat> prev_Tcqs_; // Qth별 가장 최근에 관찰된 Tcq
  std::set<Qth>               prev_rigs_;

  std::map<Qth, RigidGroup*>                  qth2rig_groups_;
  std::map<Pth, Instance*>                    pth2instances_;
  std::map<Ith, Mappoint* >                   ith2mappoints_;

  const Camera*const camera_;
  ORB_SLAM2::ORBextractor*const extractor_;
  std::shared_ptr<Mapper> mapper_;
};


} // namespace seg

#endif
