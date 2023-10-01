#include "segslam.h"
#include "Eigen/src/Core/Map.h"
#include "Eigen/src/Core/Matrix.h"
#include "camera.h"
#include "frame.h"
#include "optimizer.h"
#include "orb_extractor.h"
#include "seg.h"
#include "util.h"
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>

namespace NEW_SEG {
Frame::Frame(const Jth id,
             const cv::Mat vis_rgb)
  : id_(id), rgb_(vis_rgb), is_kf_(false)
{

}

Frame::~Frame() {
  if(flann_keypoints_.ptr())
    delete flann_keypoints_.ptr();
  if(flann_kdtree_)
    delete flann_kdtree_;
}

void Frame::SetMeasuredDepths(const cv::Mat depth) {
  for(size_t n=0; n<keypoints_.size(); n++){
    const cv::KeyPoint& kpt = keypoints_[n];
    const cv::Point2i pt(kpt.pt);
    if(pt.x < 0 || pt.y < 0 || pt.x >= depth.cols || pt.y >= depth.rows)
      continue;
    measured_depths_[n] = depth.at<float>(pt);
  }
  return;
}

void Frame::SetInstances(const cv::Mat synced_marker,
                         const std::map<Pth, Instance*>& pth2instance
                        ) {
  // 아직 mappoint 할당이 안된상태이므로 density상관없이 일단 keypoint 할당.
  for(size_t n=0; n<keypoints_.size(); n++){
    const cv::KeyPoint& kpt = keypoints_[n];
    const Pth& pth = synced_marker.at<int32_t>(kpt.pt);
    if(pth < 1)
      continue; // keepp instances_[n] as nullptr
    else
      instances_[n] = pth2instance.at(pth);
  }
  return;
}

bool Frame::IsInFrame(const Camera* camera, const Eigen::Vector2d& uv) const {
  double width = camera->GetWidth();
  double height = camera->GetHeight();
  const double ulr_boundary = 20.; // TODO fsettings 로 옮기기
  const double b_boundary = 50.;
  if( uv.y() <  ulr_boundary )
    return false;
  if( uv.x() < ulr_boundary )
    return false;
  if( width-uv.x() < ulr_boundary )
    return false;
  if( height - uv.y() < b_boundary) // 땅바닥에서 결과가 너무 안좋았다.
    return false;
  return true;
}

void Frame::SetMappoint(Mappoint* mp, int index) {
  if(!mp)
    throw std::invalid_argument("mp can't be nullptr!");
  if(mappoints_index_.count(mp))
    throw std::invalid_argument("Already matched mappoint!");
  if(mappoints_.at(index))
    throw std::invalid_argument("Erase previous mp before overwrite new mp.");
  mappoints_[index] = mp;
  mappoints_index_[mp] = index;
  return;
}

std::list<int> Frame::SearchRadius(const Eigen::Vector2d& uv, double radius) const {
  flann::Matrix<double> query((double*)uv.data(), 1, 2);
  std::list<int> inliers;
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;
  flann_kdtree_->radiusSearch(query, indices, dists, radius*radius, param);
  for(int idx :  indices[0])
    inliers.push_back(idx);
  return inliers;
}

std::vector<std::vector<int> > Frame::SearchRadius(const flann::Matrix<double>& points,
                                                   double radius) const {
  std::vector<std::vector<double> > dists;
  return this->SearchRadius(points, radius, dists);
}

std::vector<std::vector<int> > Frame::SearchRadius(const flann::Matrix<double>& points,
                                                   double radius,
                                                   std::vector<std::vector<double> >& dists
                                                   ) const {
  const flann::SearchParams param;
  std::vector<std::vector<int> > indices;
  flann_kdtree_->radiusSearch(points, indices, dists, radius*radius, param);
  return indices;
}

void Frame::ExtractAndNormalizeKeypoints(const cv::Mat gray,
                                         const Camera* camera,
                                         SEG::FeatureDescriptor* extractor,
                                         const cv::Mat& mask) {
  extractor->Extract(gray, mask, keypoints_, descriptions_);
  mappoints_.resize(keypoints_.size(), nullptr);
  instances_.resize(keypoints_.size(), nullptr);
  measured_depths_.resize(keypoints_.size(), 0.f);
  normalized_.reserve(keypoints_.size());
  for(int n=0; n<keypoints_.size(); n++){
    const auto& pt = keypoints_[n].pt;
    normalized_[n] = camera->NormalizedUndistort(Eigen::Vector2d(pt.x,pt.y));
  }
  // ref : https://github.com/mariusmuja/flann/blob/master/examples/flann_example.cpp
  flann_keypoints_
    = flann::Matrix<double>(new double[2*keypoints_.size()], keypoints_.size(), 2);
  flann_kdtree_
    = new flann::Index< flann::L2<double> >(flann_keypoints_, flann::KDTreeSingleIndexParams());
  for(size_t i = 0; i < keypoints_.size(); i++){
    const cv::Point2f& pt = keypoints_.at(i).pt;
    flann_keypoints_[i][0] = pt.x;
    flann_keypoints_[i][1] = pt.y;
  }
  flann_kdtree_->buildIndex();
  return;
}

void Frame::SetKfId(const Qth qth, int kf_id) { 
  if(kf_id_.count(qth))
    throw  -1;
  kf_id_[qth] = kf_id;
  is_kf_ = true;
}

void Frame::EraseMappoint(int index) {
  Mappoint* mp = mappoints_[index];
  mappoints_index_.erase(mp);
  mappoints_[index] = nullptr;
  return;
}


EigenMap<Ith, Eigen::Vector3d> Frame::Get3dMappoints(Qth qth) const {
  EigenMap<Ith, Eigen::Vector3d> vec3d;
  for(Mappoint* mp : mappoints_){
    if(!mp)
      continue;
    Instance* ins = mp->GetInstance();
    if(ins->GetQth() != qth)
      continue;
    if(mp->GetKeyframes(qth).size() < 2)
      continue;
    vec3d[mp->GetId()] = mp->GetXq(qth);
  }
  return vec3d;
}

void Mappoint::GetFeature(const Qth& qth, bool latest, cv::Mat& description, cv::KeyPoint& kpt) const {
  Frame* frame = nullptr;
  if(qth < 0)
    frame = ref_.begin()->second;
  else
    if(latest)
      frame = *keyframes_.at(qth).rbegin();
    else
      frame = ref_.at(qth);
  int kpt_idx = frame->GetIndex(this);
  description = frame->GetDescription(kpt_idx);
  kpt         = frame->GetKeypoint(kpt_idx);
  return;
}

RigidGroup::RigidGroup(Qth qth, Frame* first_frame)
: id_(qth),
  bg_instance_(new Instance(-1, qth))
{
}

RigidGroup::~RigidGroup() {
  delete bg_instance_;
}

bool RigidGroup::ExcludeInstance(Instance *ins) {
  const Pth pth = ins->GetId();
  excluded_instances_[pth] = ins;
  /* bool count = included_instances_.count(pth);
  if(count) 
    included_instances_.erase(pth);
  if(ins->rig_groups_.count(id_))
    ins->rig_groups_.erase(id_);
  else
    count = false;;
    return count;
  */
  return true;
}

Pipeline::Pipeline(const Camera* camera,
                   SEG::FeatureDescriptor*const extractor
                  )
  :extractor_(extractor),
  camera_(camera), 
  prev_frame_(nullptr),
  pose_tracker_(new PoseTracker)
{
}

Pipeline::~Pipeline() {

}

void Pipeline::SupplyMappoints(Frame* frame) {
  const double min_mpt_distance = 5.;
  const auto& keypoints = frame->GetKeypoints();
  const auto& instances = frame->GetInstances();
  const auto& depths    = frame->GetMeasuredDepths();
  const auto& mappoints = frame->GetMappoints();

  const Qth qth = 0;
  vinfo_supplied_mappoints_.clear();
  vinfo_supplied_mappoints_.resize(keypoints.size(), false);
  for(int n=0; n < keypoints.size(); n++){
    Instance* ins = instances[n];
    if(Mappoint* mp = mappoints[n]){
      mp->AddKeyframe(qth, frame); // TODO 모든 qth에 keyframe추가하는건 좋은생각이 아니다.
      continue;
    }
    static Ith nMappoints = 0;
    const Eigen::Vector3d Xr
      = depths[n] <1e-5 ? 1e+2*frame->GetNormalizedPoint(n) : depths[n]*frame->GetNormalizedPoint(n);
    const Eigen::Vector3d Xq = frame->GetTcq(qth).inverse()*Xr;

    const cv::KeyPoint& kpt = keypoints[n];
    std::list<int> neighbors = frame->SearchRadius(Eigen::Vector2d(kpt.pt.x, kpt.pt.y), min_mpt_distance);
    bool too_close_mp_exist = false;
    for(int nn : neighbors){
      if(mappoints[nn]){
        too_close_mp_exist=true;
        break;
      }
    }
    if(too_close_mp_exist)
      continue;

    Mappoint* mp = new Mappoint(nMappoints++, ins);
    mp->SetXr(qth, Xr);  // new mappoint를 위한 reference frame 지정.
    mp->AddReferenceKeyframe(qth, frame, Xq);
    mp->AddKeyframe(qth, frame);
    frame->SetMappoint(mp, n);
    ith2mappoints_[mp->GetId()] = mp;
    vinfo_supplied_mappoints_[n] = true;
  } // for n < keypoints.size()
  return;
}

void GetMappoints4Qth(Frame* frame,
                      const Qth& qth,
                      std::set<Mappoint*>& _mappoints
                      ){
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  for(size_t n=0; n<keypoints.size(); n++){
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    Instance* ins = mpt->GetInstance();
    if(!ins)
      continue;
    if(ins->GetQth() != qth)
      continue;
    const auto& keyframes = mpt->GetKeyframes();
    if(!keyframes.count(qth))
      continue;
    if(!keyframes.at(qth).count(frame))
      continue;
    mpt->GetXr(qth);
    _mappoints.insert(mpt);
  }
  return;
}

void GetNeighbors(Frame* keyframe,
                  const Qth& qth,
                  std::set<Mappoint*>   &neighbor_mappoints,
                  std::map<Jth, Frame*> &neighbor_keyframes) {
  int min_kf_id = keyframe->GetKfId(qth) - 10; // TODO covisiblity로 대체.
  // 1. frame에서 보이는 mappoints
  GetMappoints4Qth(keyframe, qth, neighbor_mappoints);
  // 2. '1'의 mappoint에서 보이는 nkf
  for(Mappoint* mpt : neighbor_mappoints){
    for(Frame* nkf : mpt->GetKeyframes(qth) ){
      // nkf가 qth에 대해 keyframe이 아닐경우 -1이므로, neighbor_keyframes에 여기서 추가되진 않는다.
      if(nkf->GetKfId(qth) > min_kf_id)
        neighbor_keyframes[nkf->GetId()] = nkf;
    }
  }

  // 3. '2'의 nkf에서 보이는 mappoints
  for(auto it_nkf = neighbor_keyframes.rbegin(); it_nkf != neighbor_keyframes.rend(); it_nkf++)
    GetMappoints4Qth(it_nkf->second, qth, neighbor_mappoints);
  return;
}

std::map<Qth, size_t> CountMappoints(Frame* frame){
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  std::map<Qth, size_t> n_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    Instance* ins = mpt->GetInstance();
    if(!ins)
      continue;
    if(ins->GetQth() < 0)
      continue;
    n_mappoints[ins->GetQth()]++;
  }
  return n_mappoints;
}

std::set<Qth> Pipeline::FrameNeedsToBeKeyframe(Frame* curr_frame) const {
  std::map<Qth, size_t> curr_n_mappoints = CountMappoints(curr_frame);
  std::set<Qth> need_keyframes;
  for(auto it : curr_n_mappoints){
    const Qth& qth = it.first;
    Frame* lkf = keyframes_.at(qth).rbegin()->second; // Not count 일 경우?
    if(lkf == curr_frame){
      need_keyframes.insert(qth); // ComputLBA 호출하는 while loop에서 qth rig가 생성됨.
      continue;
    }
    std::map<Qth, size_t> lkf_n_mappoints = CountMappoints(lkf);
    const size_t& n_mappoints_curr = it.second;
    if(!lkf_n_mappoints.count(qth) ){
      need_keyframes.insert(qth);
      continue;
    }
    const size_t& n_mappoints_lkf = lkf_n_mappoints.at(qth);
    size_t min_mpt_threshold = std::max<size_t>(10, .5 * n_mappoints_lkf);
    //printf("KF test. Jth %d, Qth %d,  %ld / %ld (%ld)\n", curr_frame->GetId(), qth,n_mappoints_curr, min_mpt_threshold, n_mappoints_lkf);
    if(n_mappoints_curr < min_mpt_threshold)
      need_keyframes.insert(qth);
  }
  return need_keyframes;
}

void Pipeline::FilterOutlierMatches(Frame* curr_frame) {
  const std::vector<Mappoint*>&    mappoints = curr_frame->GetMappoints();
  const std::vector<cv::KeyPoint>& keypoints = curr_frame->GetKeypoints();

  // instance 별로..
  std::vector<double> all_errors;
  all_errors.reserve(mappoints.size());

  std::map<Instance*, std::list<double> > valid_errors;
  for(int n = 0; n < mappoints.size(); n++){
    Mappoint* mp = mappoints.at(n);
    if(!mp){
      all_errors.push_back(0.);
      continue;
    }
    Instance* ins = mp->GetInstance();
    const Qth& qth = ins->GetQth();
    if(qth < 0){
      all_errors.push_back(0.);
      continue;
    }
    const g2o::SE3Quat& Tcq = curr_frame->GetTcq(qth);

    const Eigen::Vector3d Xr = mp->GetXr(qth);
    Frame* ref = mp->GetRefFrame(qth);
    const g2o::SE3Quat& Trq = ref->GetTcq(qth);
    const Eigen::Vector3d Xc = Tcq * Trq.inverse() * Xr;
    Eigen::Vector2d rprj_uv = camera_->Project(Xc);
    const auto& pt = keypoints.at(n).pt;
    const Eigen::Vector2d uv(pt.x, pt.y);
    double err = (uv-rprj_uv).norm();
    all_errors.push_back(err);
    valid_errors[ins].push_back(err);
  }

  std::map<Instance*, double > err_thresholds;
  for(auto it : valid_errors){
    std::list<double>& errors = it.second;
    errors.sort();
    auto it_median = errors.begin();
    std::advance(it_median, errors.size()/2);
    err_thresholds[it.first] = 4. * (*it_median);
  }
  //double err_threshold = 5.;

  for(int n = 0; n < mappoints.size(); n++){
    Mappoint* mp = mappoints.at(n);
    if(!mp)
      continue;
    Instance* ins = mp->GetInstance();
    if(ins->GetQth()<0)
      continue;
    const double& err = all_errors.at(n);
    const double& err_th = err_thresholds.at(ins);
    if(err < 10.) // TODO
      continue;
    //if(err < err_th)
    //  continue;
    curr_frame->EraseMappoint(n); // keyframe이 아니라서 mp->RemoveKeyframe 등을 호출하지 않는다.
  }
  return;
}

void SetMatches(std::map<int, std::pair<Mappoint*, double> >& flow_matches,
                std::map<int, std::pair<Mappoint*, double> >& proj_matches,
                Frame* curr_frame){
  std::map<Mappoint*,int> matched_mappoints;
  for(auto it : flow_matches){
    //curr_frame->SetMappoint(it.second.first, it.first); // 이미 Posetracking을 위해 연결했다.
    matched_mappoints[it.second.first] = it.first;
  }
  const std::vector<Mappoint*>& mappoints = curr_frame->GetMappoints();
  for(auto it : proj_matches){
    const int& prj_n = it.first;
    Mappoint*const prj_mp = it.second.first;
    bool c1 = flow_matches.count(prj_n);
    bool c2 = matched_mappoints.count(prj_mp);
    if(c1 && c2) // flow match 와 결과가 똑같은 경우.
      continue;
    else if (c2) // flow_match가 연결시킨 mappoint를 다른 keypoint에 연결하려하는경우.
      continue;
    if(mappoints[prj_n]) // 이미 다른 mappoint가 연결된 keypoint
      continue;
    curr_frame->SetMappoint(prj_mp, prj_n);
  }
  return;
}

void CountMappoints(Frame* frame,
                    std::map<Pth, size_t>& ins2mappoints){
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  for(size_t n=0; n<keypoints.size(); n++){
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    Instance* ins = mpt->GetInstance();
    if(!ins)
      continue;
    Pth pth = ins->GetId();
    if(pth > -1)
      ins2mappoints[pth]++;
  }
  return;
}

Frame* Pipeline::Put(const cv::Mat gray,
                     const cv::Mat depth,
                     const std::vector<cv::Mat>& flow,
                     const cv::Mat synced_marker,
                     const std::map<int,size_t>& marker_areas,
                     const cv::Mat gradx,
                     const cv::Mat grady,
                     const cv::Mat valid_grad,
                     const cv::Mat vis_rgb
                    ) {
  switch_threshold_ = .3;
  float density_threshold = 1. / 70. / 70.; // NxN pixel에 한개 이상의 mappoint가 존재해야 dense instance

  vinfo_switch_states_.clear();
  vinfo_neighbor_frames_.clear();
  vinfo_neighbor_mappoints_.clear();
  vinfo_synced_marker_ = synced_marker;

  const float proj_search_radius = 30.;
  const Qth qth_default = 0;
  cv::Mat outline_mask = synced_marker < 1;
  std::map<Qth, std::set<Pth> > segmented_instances;
  for(const auto& it : marker_areas){
    Instance* ins = nullptr;
    if(!pth2instances_.count(it.first) ){
      ins = new Instance(it.first, qth_default);
      pth2instances_[it.first] = ins;
    }
    else
      ins = pth2instances_.at(it.first);
    segmented_instances[ins->GetQth()].insert(ins->GetId());
  }
  if(segmented_instances.empty())
    throw -1;

  static Jth nFrames = 0;
  Frame* curr_frame = new Frame(nFrames++, vis_rgb);
  curr_frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_, outline_mask);
  curr_frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_, outline_mask);
  curr_frame->SetMeasuredDepths(depth);
  curr_frame->SetInstances(synced_marker, pth2instances_);

  if(!prev_frame_){
    RigidGroup* rig_new = new RigidGroup(0, curr_frame);
    qth2rig_groups_[rig_new->GetId()] = rig_new;
    curr_frame->SetTcq(qth_default, g2o::SE3Quat() );
    SupplyMappoints(curr_frame);
    keyframes_[qth_default][curr_frame->GetId()] = curr_frame;
    curr_frame->SetKfId(qth_default,0);
    prev_frame_ = curr_frame;
    return curr_frame;
  } 

  // intiial pose prediction
  const auto& prev_Tcqs = prev_frame_->GetTcqs();
  for(auto it : prev_Tcqs)
    curr_frame->SetTcq(it.first, *it.second);

  bool verbose_flowmatch = false;
  std::map<int, std::pair<Mappoint*, double> > flow_matches
    = FlowMatch(camera_, extractor_, flow, prev_frame_, verbose_flowmatch, curr_frame);
  std::map<Mappoint*,int> matched_mappoints;
  for(auto it : flow_matches){
    curr_frame->SetMappoint(it.second.first, it.first);
    matched_mappoints[it.second.first] = it.first;
  }

  static std::map<Qth, std::map<Pth,size_t> > n_consecutiv_switchoff;
  int nq = 1;
  { // while !segmented_instances.empty()
    const Qth qth = 0;
    RigidGroup* rig = qth2rig_groups_.at(qth);
    bool verbose_track = true;
    g2o::SE3Quat Tcq = pose_tracker_->GetTcq(camera_, qth_default, curr_frame, verbose_track);
    curr_frame->SetTcq(qth_default, Tcq);

    Frame* latest_kf = keyframes_.at(qth).rbegin()->second;
    std::set<Mappoint*>     & neighbor_mappoints = vinfo_neighbor_mappoints_[qth];
    std::map<Jth, Frame* >  & neighbor_frames    = vinfo_neighbor_frames_[qth];
    GetNeighbors(latest_kf, qth_default, neighbor_mappoints, neighbor_frames);
    if(neighbor_mappoints.empty()) {
      std::cerr << "qth = " << qth << ", failure to get mappoints" << std::endl;
      throw -1;
    }

    // ProjectionMatch는 flow로부터 motion update와 pth(k_0)!= pth(k) 인 경우에 예외처리가 팔요해보인다.
    std::map<int, std::pair<Mappoint*, double> > proj_matches = ProjectionMatch(camera_, extractor_, neighbor_mappoints, curr_frame, qth, proj_search_radius);
    SetMatches(flow_matches, proj_matches, curr_frame);
    FilterOutlierMatches(curr_frame); // LBA 전에 호출되야 accuracy가 높았다.

    std::set<Pth> fixed_instances;
    bool vis_verbose = false;
    std::map<Pth,float>& switch_states = vinfo_switch_states_[qth];
    switch_states\
      = mapper_->ComputeLBA(camera_,qth, neighbor_mappoints, neighbor_frames, curr_frame, prev_frame_, fixed_instances, gradx, grady, valid_grad, vis_verbose);

    std::map<Pth, size_t> ins2mappoints;
    CountMappoints(curr_frame, ins2mappoints);
    std::set<Pth> instances4next_rig;
    std::set<Pth> switchoff_instances;
    if(!n_consecutiv_switchoff.count(qth))
      n_consecutiv_switchoff[qth]; // empty map 생성.
    for(auto it_switch : switch_states){
      const Pth& pth = it_switch.first;
      if(it_switch.second > switch_threshold_)
        continue;
      size_t& n_consec = n_consecutiv_switchoff[qth][pth];
      n_consec++;
      if(n_consec > 2){ // N 번 연속 switch off 판정을 받은경우,
        Instance* ins = pth2instances_.at( pth );
        rig->ExcludeInstance(ins);
        instances4next_rig.insert(pth);
        ins->SetQth(-1);
      }
      else
        switchoff_instances.insert(pth); // 현재 프레임에서 switchoff가 발생하지 않은 pthqth를 삭제하기위해.
    }

    size_t n_mappoints4next_rig = 0;
    for(Pth pth : instances4next_rig)
      if(ins2mappoints.count(pth))// mp 없는 instance도 있으니까.
         n_mappoints4next_rig += ins2mappoints.at(pth);
  }

  std::set<Qth> need_keyframe = FrameNeedsToBeKeyframe(curr_frame);
  for(const Qth& qth : need_keyframe){
    keyframes_[qth][curr_frame->GetId()] = curr_frame;
    if(keyframes_.at(qth).size() > 1)// ComputLBA 를 포함한 while loop문에서 이미 추가된 keyframe.
      curr_frame->SetKfId(qth, keyframes_[qth].size() );
  }
  if(!need_keyframe.empty())
    SupplyMappoints(curr_frame);
  if(prev_frame_ && ! prev_frame_->IsKeyframe() )
    delete prev_frame_;
  prev_frame_ = curr_frame;
  return curr_frame;
}

} // namespace NEW_SEG
