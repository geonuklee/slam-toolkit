#ifndef SEG_FRAME_
#define SEG_FRAME_
#include <flann/flann.hpp> // include it before opencv
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include "Eigen/src/Core/Matrix.h"
#include "stdafx.h"
#include "seg.h"
#include "common.h"

namespace ORB_SLAM2{ class ORBextractor; }

class Camera;
class DepthCamera; // TODO 'depth' camera only?

namespace SEG {
class FeatureDescriptor {
public:
  virtual void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) = 0;
  virtual double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const = 0;
  virtual ~FeatureDescriptor() {}
  double GetScaleFactor() const { return scaleFactor; }
protected:
  double scaleFactor;
};

class OrbSlam2FeatureDescriptor : public FeatureDescriptor {
public:
  OrbSlam2FeatureDescriptor(int nfeatures, float scale_factor, int nlevels, int initial_fast_th, int min_fast_th);
  ~OrbSlam2FeatureDescriptor() {}
  void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  inline double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const;
private:
  std::shared_ptr<ORB_SLAM2::ORBextractor> extractor_;
};

class CvFeatureDescriptor : public FeatureDescriptor {
public:
  CvFeatureDescriptor();
  ~CvFeatureDescriptor() {}
  void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  inline double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const;
  std::vector<cv::Mat> mvImagePyramid;
private:
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, const cv::Mat& mask);
  std::vector<cv::Point> pattern;
  int nfeatures;
  int nlevels;
  int iniThFAST;
  int min_kpt_distance;
  std::vector<int> mnFeaturesPerLevel;
  std::vector<int> umax;
  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

class NewFeatureDescriptor : public FeatureDescriptor {
public:
  NewFeatureDescriptor();
  ~NewFeatureDescriptor() {}
  void Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  inline double GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const;
private:
  cv::Ptr<cv::ORB> orb_;
};

} // namespace SEG

typedef int32_t Jth; // jth frame
typedef int32_t Qth; // qth rig group
typedef int32_t Pth; // pth instance
typedef int32_t Ith; // ith mappoint

namespace NEW_SEG {
class Mappoint;
struct RigidGroup;

class Instance {
public:
  Instance(Pth pth, Qth qth) : pth_(pth), qth_(qth) { }
  const Pth& GetId() const { return pth_; }
  const Qth& GetQth() const { return qth_; }
  void SetQth(const Qth& qth) { qth_ = qth; }

  const std::set<Mappoint*>& GetMappoints() const { return mappoints_; }
  void AddMappoint(Mappoint* mp) { mappoints_.insert(mp); }
  void AddMappoints(const std::set<Mappoint*>& mappoints) { mappoints_.insert(mappoints.begin(), mappoints.end());}

  void SetLatestKfMappoints(const std::set<Mappoint*>& mappoints)  { latest_kf_mappoints_ = mappoints; }
  const std::set<Mappoint*>& GetLatestKfMappoints() const { return latest_kf_mappoints_; }

private:
  Qth qth_;
  std::set<Mappoint*> mappoints_; // all
  std::set<Mappoint*> latest_kf_mappoints_;
  const Pth pth_;
};

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Frame(const Jth id, double sec, const cv::Mat vis_rgb=cv::Mat());
  ~Frame();
  bool IsInFrame(const Camera* camera, const Eigen::Vector2d& uv) const;
  int GetIndex(const Mappoint* mp) const { return mappoints_index_.count(mp) ? mappoints_index_.at(mp) : -1; }

  void ExtractAndNormalizeKeypoints(const cv::Mat gray,
                                    const Camera* camera,
                                    SEG::FeatureDescriptor* extractor,
                                    const cv::Mat& mask);
  void AddKeypoints(const Camera*const camera,
                    const std::vector< cv::KeyPoint >& added_keypoints,
                    const std::vector< Mappoint* >&    added_mappoints,
                    const std::vector< Instance* >&    added_instances,
                    const std::vector< float >&        added_depths
                    );

  void SetInstances(const cv::Mat synced_marker, const std::map<Pth, Instance*>& instances);
  void SetInstance(int index, Instance* ins) { instances_[index] = ins; }
  void SetMeasuredDepths(const cv::Mat depth);

  std::list<std::pair<int, double> > SearchRadius(const Eigen::Vector2d& uv, double radius) const;
  std::vector<std::vector<int> > SearchRadius(const flann::Matrix<double>& points, double radius) const;
  std::vector<std::vector<int> > SearchRadius(const flann::Matrix<double>& points, double radius, 
                                              std::vector<std::vector<double> >& dists) const;

  const cv::Mat GetRgb() const { return rgb_; }
  const std::vector<cv::KeyPoint>& GetKeypoints()      const { return keypoints_; }
  Mappoint* GetMappoint(int index)                     const { return mappoints_[index]; }
  const std::vector<Mappoint*>&    GetMappoints()      const { return mappoints_; }
  const std::vector<Instance*>&    GetInstances()      const { return instances_; }
  const std::vector<float>&        GetMeasuredDepths() const { return measured_depths_; }
  std::vector<Instance*>&          GetInstances() { return instances_; }
  void SetMappoint(Mappoint* mp, int kpt_index);
  void EraseMappoint(int index);
  const Eigen::Vector3d& GetNormalizedPoint(int index) const { return normalized_[index]; }
  const EigenVector<Eigen::Vector3d>& GetNormalizedPoints() const { return normalized_; }

  void ReduceMem() { rgb_ = cv::Mat(); }
  const cv::Mat GetDescription(int i) const { return descriptions_.row(i); }
  const cv::Mat GetDescriptions() const { return descriptions_; }
  const cv::KeyPoint& GetKeypoint(int i) const {  return keypoints_.at(i); }
  const float& GetDepth(int i) const { return measured_depths_.at(i); }
  const Jth GetId() const { return id_; }
  void SetKfId(const Qth qth, int kf_id);
  const int GetKfId(const Qth qth) const { return kf_id_.count(qth) ? kf_id_.at(qth) : -1; }
  bool IsKeyframe() const { return is_kf_; }
  void SetTcq(const Qth& qth, const g2o::SE3Quat& Tcq) { Tcq_[qth] = std::shared_ptr<g2o::SE3Quat>(new g2o::SE3Quat(Tcq)); }
  const g2o::SE3Quat& GetTcq(const Qth& qth) const { return *Tcq_.at(qth); }
  const std::map<Qth, std::shared_ptr<g2o::SE3Quat> >& GetTcqs() const { return Tcq_; }

  EigenMap<Ith, Eigen::Vector3d> Get3dMappoints(Qth qth=0) const;

  double GetSec() const { return sec_; }
private:
  const double sec_;
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
  std::map<Qth, int32_t> kf_id_;
  cv::Mat rgb_;
  const Jth id_;
  bool is_kf_;
};

class Mappoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mappoint(Ith id, Instance* ins, const cv::KeyPoint& kpt, cv::Mat desc) : id_(id), ins_(ins), latest_ins_(ins), kpt_(kpt), desc_(desc) {}
  Instance* GetInstance() const { return ins_; }
  Instance* GetLatestInstance() const { return latest_ins_; }
  void ChangeInstance(Instance* ins) { ins_ = ins; if(latest_ins_==ins) latest_ins_=ins;}
  void SetLatestInstance(Instance* ins) { latest_ins_ = ins; }
  const Ith& GetId() const { return id_; }
  void GetFeature(const Qth& qth, bool latest, cv::Mat& description, cv::KeyPoint& kpt) const;

  Frame* GetRefFrame(const Qth& qth) const { return ref_.at(qth); }
  const std::map<Qth,Frame*>& GetRefFrames() const { return ref_; }
  bool HasEstimate4Rig(const Qth& qth) const { return ref_.count(qth); }
  void AddKeyframe(Qth qth, Frame* frame) { keyframes_[qth].insert(frame); }
  void RemoveKeyframe(Frame* frame);
  const std::map<Qth, std::set<Frame*> >& GetKeyframes() const { return keyframes_; }
  const std::set<Frame*>& GetKeyframes(Qth qth) const { return keyframes_.at(qth); }
  const cv::Mat& GetDescription() const { return desc_; }
  const cv::KeyPoint& GetKeypoint() const { return kpt_; }

  // Ref coordinate에서 본 Xr. Depth Camera의 Measurement
  void AddReferenceKeyframe(const Qth& qth, Frame* ref, const Eigen::Vector3d& Xq) { ref_[qth] = ref; SetXq(qth, Xq); }
  void SetXr(const Qth& qth, const Eigen::Vector3d& Xr) { Xr_[qth] = std::shared_ptr<Eigen::Vector3d>(new Eigen::Vector3d(Xr)); }
  //const Eigen::Vector3d& GetXr(const Qth& qth) const { return *Xr_.at(qth); }

  // Qth rigid coordinate 에 맵핑된 결과값.
  void SetXq(Qth qth, const Eigen::Vector3d& Xq) { Xq_[qth] = std::shared_ptr<Eigen::Vector3d>(new Eigen::Vector3d(Xq)); }
  const Eigen::Vector3d& GetXq(Qth qth) { return *Xq_.at(qth); }

public:
  static std::map<Qth, size_t> n_;

private:
  const Ith id_;
  Instance* ins_;
  Instance* latest_ins_;

  // TODO 더이상 ins가 Qth를 포함하지 않을경우 제외해야하는데, 모순이다.
  std::map<Qth, Frame*> ref_;
  std::map<Qth, std::set<Frame*> > keyframes_; // GetNeighbors에 필요
  std::map<Qth, std::shared_ptr<Eigen::Vector3d> >Xr_; // Measurement
  std::map<Qth, std::shared_ptr<Eigen::Vector3d> >Xq_; // rig마다 독립된 Mapping값을 가진다.
  const cv::KeyPoint kpt_;
  const cv::Mat desc_;
};


struct RigidGroup {
  RigidGroup(Qth qth, Frame* first_frame);
  ~RigidGroup();
  bool ExcludeInstance(Instance* ins);
  Instance* GetBgInstance() const { return bg_instance_; }
  const Qth& GetId() const { return id_; }
  const std::map<Pth, Instance*>& GetExcludedInstances() const { return excluded_instances_; }
  bool RemoveExcludedInstance(Instance* ins);

 private:
  const Qth id_;
  Instance*const bg_instance_;
  std::map<Pth, Instance*> excluded_instances_;
};

class PoseTracker;
class Mapper;
class Pipeline {
public:
  Pipeline(const Camera* camera);
  ~Pipeline();
  Frame* Put(const cv::Mat gray,
             const cv::Mat depth,
             const std::vector<cv::Mat>& flow,
             cv::Mat& synced_marker,
             std::map<int,size_t> marker_areas,
             const cv::Mat gradx,
             const cv::Mat grady,
             const cv::Mat valid_grad,
             double sec,
             const cv::Mat vis_rgb
            );
  void Visualize(const cv::Mat rgb, const cv::Mat gt_dynamic_mask, cv::Mat& dst); // visualize.cpp
  RigidGroup* GetRigidGroup(Qth qth) const { return qth2rig_groups_.count(qth) ? qth2rig_groups_.at(qth) : nullptr; }
  const std::map<Pth, Instance*>& GetInstances()  const { return pth2instances_; }
  const std::map<Pth, Pth>& GetPthRemoved2Replacing() const { return pth_removed2replacing_; }
  EigenMap<Jth, g2o::SE3Quat> GetUpdatedTcqs() const;
private:
  std::map<Instance*,Instance*> MergeEquivalentInstances(const std::map<Instance*, std::set<Mappoint*> >& ins2mappoints,
                                                         const std::map<Pth,float>& density_scores);
  std::set<Qth> FrameNeedsToBeKeyframe(Frame* frame) const;
  void SupplyMappoints(Frame* frame);
  std::set<Pth> FilterOutlierMatches(Frame* curr_frame, const EigenMap<Qth, g2o::SE3Quat>& Tcps, bool verbose);

  SEG::FeatureDescriptor* extractor_;
  const Camera*const camera_;
  std::shared_ptr<PoseTracker> pose_tracker_;
  std::shared_ptr<Mapper> mapper_;

  std::map<Qth, std::map<Jth, Frame*> > keyframes_;
  Frame* prev_frame_;
  std::map<Qth, RigidGroup*> qth2rig_groups_;
  std::map<Pth, Instance*>   pth2instances_;
  std::map<Ith, Mappoint* >  ith2mappoints_;

  std::map<Pth, Pth> pth_removed2replacing_; // log for evaluation
  std::vector<bool> vinfo_supplied_mappoints_;
  float switch_threshold_;
  std::map<Qth, std::map<Pth, float> >  vinfo_switch_states_;
  std::map<Qth, std::map<Jth, Frame*> > vinfo_neighbor_frames_;
  std::map<Qth, std::set<Mappoint*> >   vinfo_neighbor_mappoints_;
  cv::Mat                               vinfo_synced_marker_;
  cv::Mat                               vinfo_match_filter_;
  std::map<Pth,float>                   vinfo_density_socres_;
}; // class Pipeline

Eigen::Matrix4f Estimate3D3DRANSAC( const std::vector<cv::Point3f>& src, 
    const std::vector<cv::Point3f>& dst, 
    int iterations, 
    float threshold); // ransac.cpp

std::map<int, std::pair<Mappoint*, double> > FlowMatch(const Camera* camera,
                                                       SEG::FeatureDescriptor* extractor,
                                                       const std::vector<cv::Mat>& flow,
                                                       const Frame* prev_frame,
                                                       const cv::Mat& curr_depth,
                                                       const cv::Mat& curr_marker,
                                                       const std::map<Pth,Instance*>& pth2instances,
                                                       bool verbose,
                                                       Frame* curr_frame) ;

std::map<int, std::pair<Mappoint*,double> > ProjectionMatch(const Camera* camera,
                                                            const SEG::FeatureDescriptor* extractor,
                                                            const std::set<Mappoint*>& mappoints,
                                                            const Frame* curr_frame, // With predicted Tcq
                                                            const Qth qth);
 
} // namespace NEW_SEG

#endif
