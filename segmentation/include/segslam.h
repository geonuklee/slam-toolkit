#ifndef SEG_FRAME_
#define SEG_FRAME_
#include "stdafx.h"
#include "seg.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

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
class RigidGroup;
class Instance {
public:
  Instance(Pth pth);
  Pth pth_;
  std::set<RigidGroup*> rig_groups_;
};

class Mappoint;
class Frame {
public:
  Frame(const Jth id, const cv::Mat vis_rgb=cv::Mat());

  void ExtractKeypoints(const cv::Mat gray, ORB_SLAM2::ORBextractor* extractor);
  void SetInstances(const std::map<Pth, ShapePtr>& shapes,
                    const std::map<Pth, Instance*>& instances
                    );
  void SetMeasuredDepths(const cv::Mat depth);

  const cv::Mat GetRgb() const { return rgb_; }
  const std::vector<cv::KeyPoint>& GetKeypoints()      const { return keypoints_; }
  const std::vector<Mappoint*>&    GetMappoints()      const { return mappoints_; }
  const std::vector<Instance*>&    GetInstances()      const { return instances_; }
  const std::vector<float>&        GetMeasuredDepths() const { return measured_depths_; }
  std::vector<Instance*>&          GetInstances() { return instances_; }
  std::vector<Mappoint*>&          GetMappoints() { return mappoints_; }

  void ReduceMem();
  const Jth GetId() const { return id_; }
private:
  std::vector<cv::KeyPoint>    keypoints_;
  std::vector<Mappoint*>       mappoints_;
  std::vector<Instance*>       instances_; // 사실 Pth가 Qth를 가리켜야속편한데, 이거때문에 어려워짐.
  std::vector<float>           measured_depths_;
  cv::Mat descriptions_;
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
  Mappoint(Ith id,
           const std::map<Qth,RigidGroup*>& visible_rigs,
           Frame* ref, 
           float invd);
  const Ith& GetId() const { return id_; }

  void SetInvD(float invd);
  float GetDepth() const;
  cv::Mat GetDescription() const; // ref에서의 desc, kpt
  cv::KeyPoint GetKeypoint() const;
  int GetIndex() const { return id_; }
  Frame* GetRefFrame() const { return ref_; }

  const std::map<Qth, std::set<Frame*> >& GetKeyframes() const { return keyframes_; }
  const std::set<Frame*>& GetKeyframes(Qth qth) const { return keyframes_.at(qth); }

public:
  static std::map<Qth, size_t> n_;

private:
  Frame* ref_;
  std::map<Qth, std::set<Frame*> > keyframes_;
  const Ith id_;
  float invd_;
};

struct RigidGroup {
  // Instance의 묶음 
public:
  RigidGroup(Qth qth);
  ~RigidGroup();
  void IncludeInstance(Instance* ins);
  Instance* GetBgInstance() const { return bg_instance_; }
  const Qth& GetId() const { return id_; }
private:
  const Qth id_;
  Instance*const bg_instance_;
  std::map<Pth, Instance*> excluded_instances_;
  std::map<Pth, Instance*> included_instances_; // TODO Need?
};

/*
 argmin sig(s(p,q)^) [ Xc(j) - Tcq(j)^ Xq(i∈p)^ ]
 Xq(i∈p)^ : q th group coordinate에서 본 'i'th mappoint의 좌표.
  -> 일시적으로, 하나의 'i'th mappoint가 여러개의 3D position을 가질 수 있다.
  -> Instance 의 모양이 group마다 바뀐다는 모순이 생길순 있지만, 이런 경우가 적으므로,
    사소한 연결때문에 group을 넘나드는 SE(3)사이의 covariance를 늘릴 필요는 없다.( solver dimmension)
 SE(3) 의 dimmension을 줄일려고( n(G) dim, 1iter -> 1dim, n(G) iter ) 
 binary로 group member 가지치기 들어가기.
 */
class Pipeline {
public:
  Pipeline(const Camera* camera,
           ORB_SLAM2::ORBextractor*const extractor
          );
  ~Pipeline();
  void Put(const cv::Mat gray,
           const cv::Mat depth,
           const std::map<Pth, ShapePtr>& shapes,
           const cv::Mat vis_rgb=cv::Mat());

private:
  std::map<Qth,bool> NeedKeyframe(Frame* frame,
                                RigidGroup* new_rig) const;
  // j -> q -> (p) -> i
  std::map<Qth, std::map<Jth, Frame*> >       keyframes_;  // jth {c}amera와 keypointt<->'i'th mappoint correspondence
  Frame* prev_frame_;

  EigenMap<Qth, EigenMap<Jth, g2o::SE3Quat> > Tcqs_;       // Tcq(j) : {c}amera <- {q}th group for 'j'th frame
  std::map<Qth, RigidGroup*>                  qth2rig_groups_;
  std::map<Pth, Instance*>                    pth2instances_;
  std::map<Ith, Mappoint* >                   ith2mappoints_;

  const Camera*const camera_;
  ORB_SLAM2::ORBextractor*const extractor_;
};


} // namespace seg

#endif
