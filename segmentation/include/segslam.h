#ifndef SEG_FRAME_
#define SEG_FRAME_
#include <flann/flann.hpp> // include it before opencv
#include <opencv2/core/mat.hpp>
#include "stdafx.h"
#include "seg.h"

namespace ORB_SLAM2{ class ORBextractor; }

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
struct RigidGroup;
class Mappoint;
class Instance {
public:
  Instance(Pth pth);
  std::map<Qth,RigidGroup*> rig_groups_;
  const Pth& GetId() const { return pth_; }
private:
  const Pth pth_;
};

class FeatureDescriptor {
public:
  virtual void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) = 0;
  virtual double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const = 0;
};

class OrbSlam2FeatureDescriptor : public FeatureDescriptor {
public:
  OrbSlam2FeatureDescriptor(int nfeatures, float scale_factor, int nlevels, int initial_fast_th, int min_fast_th);
  void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  inline double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const;
private:
  std::shared_ptr<ORB_SLAM2::ORBextractor> extractor_;
};

class CvFeatureDescriptor : public FeatureDescriptor {
public:
  CvFeatureDescriptor();
  void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  inline double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const;

  std::vector<cv::Mat> mvImagePyramid;
private:
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
  std::vector<cv::Point> pattern;
  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST; // TODO remove?
  int min_kpt_distance;
  std::vector<int> mnFeaturesPerLevel;
  std::vector<int> umax;
  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

// TODO vToDistributeKeys에 class_id로 keypoint id를 저장해놓고선, 이 함수의 결과물로
// Mappoint를 화면 골고루에 뿌릴 준비를 한다.
// min_distance : 특징점 사이의 최소거리.
std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                            const int &minX,
                                            const int &maxX,
                                            const int &minY,
                                            const int &maxY,
                                            const int &nFeaturesPerLevel,
                                            const int &min_distance = 10
                                           );


class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Frame(const Jth id, const cv::Mat vis_rgb=cv::Mat());
  ~Frame();
  bool IsInFrame(const Camera* camera, const Eigen::Vector2d& uv) const;
  int GetIndex(const Mappoint* mp) const;
  void ExtractAndNormalizeKeypoints(const cv::Mat gray,
                                    const Camera* camera,
                                    FeatureDescriptor* extractor);
  std::map<Pth,float> SetInstances(const std::map<Pth, ShapePtr>& shapes,
                                   const std::map<Pth, Instance*>& instances,
                                   float density_threshold
                                  );
  void SetMeasuredDepths(const cv::Mat depth);

  std::set<int> SearchRadius(const Eigen::Vector2d& uv, double radius) const;
  std::vector<std::vector<int> > SearchRadius(const flann::Matrix<double>& points,
                                              double radius) const;

  const cv::Mat GetRgb() const { return rgb_; }
  const std::vector<cv::KeyPoint>& GetKeypoints()      const { return keypoints_; }
  Mappoint* GetMappoint(int index)                     const { return mappoints_[index]; }
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

  void SetTcq(const Qth& qth, const g2o::SE3Quat& Tcq) { Tcq_[qth] = std::shared_ptr<g2o::SE3Quat>(new g2o::SE3Quat(Tcq)); }
  const g2o::SE3Quat& GetTcq(const Qth& qth) const { return *Tcq_.at(qth); }
  const std::map<Qth, std::shared_ptr<g2o::SE3Quat> >& GetTcqs() const { return Tcq_; }

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

  std::map<Qth, std::shared_ptr<g2o::SE3Quat> > Tcq_;
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
  Mappoint(Ith id, Instance* ins);
  Instance* GetInstance() const { return ins_; }
  const Ith& GetId() const { return id_; }
  //int GetIndex() const { return id_; } // TODO 중복 삭제.
  cv::Mat GetDescription() const; // ref에서의 desc
  Frame* GetRefFrame(const Qth& qth) const { return ref_.at(qth); }
  const std::map<Qth,Frame*>& GetRefFrames() const { return ref_; }
  bool HasEstimate4Rig(const Qth& qth) const { return ref_.count(qth); }

  void AddKeyframe(Qth qth, Frame* frame) { keyframes_[qth].insert(frame); }
  const std::map<Qth, std::set<Frame*> >& GetKeyframes() const { return keyframes_; }
  const std::set<Frame*>& GetKeyframes(Qth qth) const { return keyframes_.at(qth); }

  // Ref coordinate에서 본 Xr. Depth Camera의 Measurement
  void AddReferenceKeyframe(const Qth& qth, Frame* ref);
  void SetXr(const Qth& qth, const Eigen::Vector3d& Xr);
  const Eigen::Vector3d& GetXr(const Qth& qth) const;

  // Qth rigid coordinate 에 맵핑된 결과값.
  void SetXq(Qth qth, const Eigen::Vector3d& Xq);
  const Eigen::Vector3d& GetXq(Qth qth);

public:
  static std::map<Qth, size_t> n_;

private:
  const Ith id_;
  Instance*const ins_;

  // TODO 더이상 ins가 Qth를 포함하지 않을경우 제외해야하는데, 모순이다.
  std::map<Qth, Frame*> ref_;
  std::map<Qth, std::set<Frame*> > keyframes_; // GetNeighbors에 필요
  std::map<Qth, std::shared_ptr<Eigen::Vector3d> >Xr_; // Measurement
  std::map<Qth, std::shared_ptr<Eigen::Vector3d> >Xq_; // rig마다 독립된 Mapping값을 가진다.
};

struct RigidGroup {
  // Instance의 묶음
public:
  RigidGroup(Qth qth, Frame* first_frame);
  ~RigidGroup();
  bool IncludeInstance(Instance* ins);
  bool ExcludeInstance(Instance* ins);
  Instance* GetBgInstance() const { return bg_instance_; }
  const Qth& GetId() const { return id_; }
  const std::map<Pth, Instance*>& GetIncludedInstances() const { return included_instances_; }
  const std::map<Pth, Instance*>& GetExcludedInstances() const { return excluded_instances_; }

private:
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
           FeatureDescriptor*const extractor
          );
  ~Pipeline();
  void Put(const cv::Mat gray,
           const cv::Mat depth,
           const cv::Mat flow0,
           const std::map<Pth, ShapePtr>& shapes,
           const cv::Mat vis_rgb=cv::Mat());

private:
  void SupplyMappoints(Frame* frame, RigidGroup* rig_new);
  void AddNewKeyframesMappoints(Frame* frame,
                                RigidGroup* rig_new);
  std::map<Qth,bool> FrameNeedsToBeKeyframe(Frame* frame,
                                                RigidGroup* rig_new) const;
  void UpdateRigGroups(const std::set<Qth>& curr_rigs, Frame* frame) const;

  // j -> q -> (p) -> i
  std::map<Qth, std::map<Jth, Frame*> >       keyframes_;  // jth {c}amera와 keypointt<->'i'th mappoint correspondence
  std::set<Jth>                         every_keyframes_;

  // Tcq(j) : {c}amera <- {q}th group for 'j'th frame
  Frame*                      prev_frame_;
  std::set<Qth>               prev_rigs_;
  Qth                         prev_dominant_qth_;
  std::map<Qth, RigidGroup*>                  qth2rig_groups_;
  std::map<Pth, Instance*>                    pth2instances_;
  std::map<Ith, Mappoint* >                   ith2mappoints_;

  const Camera*const camera_;
  FeatureDescriptor*const extractor_;
  std::shared_ptr<Mapper> mapper_;
};


} // namespace seg

#endif
