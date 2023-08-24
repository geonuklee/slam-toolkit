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

namespace seg {

RigidGroup::RigidGroup(Qth qth, Frame* first_frame)
: id_(qth),
  bg_instance_(new Instance(-1))
{
  AddInlierInstace(bg_instance_);
}

bool RigidGroup::AddInlierInstace(Instance* ins) {
  const Pth pth = ins->GetId();
  if(excluded_instances_.count(pth) )
    return false;
  ins->rig_groups_[id_] = this;
  included_instances_[pth] = ins;
  return true;
}

RigidGroup::~RigidGroup() {
  delete bg_instance_;
}

Instance::Instance(Pth pth)
  : pth_(pth)
{

}

Frame::Frame(const Jth id,
             const cv::Mat vis_rgb)
  : id_(id), rgb_(vis_rgb)
{

}

Frame::~Frame() {
  if(flann_keypoints_.ptr())
    delete flann_keypoints_.ptr();
  if(flann_kdtree_)
    delete flann_kdtree_;
}

bool Frame::IsInFrame(const Camera* camera, const Eigen::Vector2d& uv) const {
  double width = camera->GetWidth();
  double height = camera->GetHeight();
  const double ulr_boundary = 20.;
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

int Frame::GetIndex(const Mappoint* mp) const {
  if(mappoints_index_.count(mp))
    return mappoints_index_.at(mp);
  return -1;
}

void Frame::EraseMappoint(int index) {
  Mappoint* mp = mappoints_[index];
  mappoints_index_.erase(mp);
  mappoints_[index] = nullptr;
  return;
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
                                         FeatureDescriptor* extractor,
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

void Frame::SetInstances(const std::map<Pth, ShapePtr>& shapes,
                         const std::map<Pth, Instance*>& pth2instance
                        ) {
  // 아직 mappoint 할당이 안된상태이므로 density상관없이 일단 keypoint 할당.
  for(size_t n=0; n<keypoints_.size(); n++){
    const cv::KeyPoint& kpt = keypoints_[n];
    for(auto it_shape : shapes){
      ShapePtr s_ptr = it_shape.second;
      if(s_ptr->n_missing_ > 0)
        continue;
      const auto& pt = kpt.pt;
      if(! s_ptr->HasCollision(pt.x, pt.y, true) )
        continue;
      const Pth& pth = it_shape.first;
      instances_[n] = pth2instance.at(pth);
      break;
    }
  }
  return;
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

void Frame::ReduceMem() {
  rgb_ = cv::Mat();
  return;
}

const cv::Mat Frame::GetDescription(int i) const {
  cv::Mat desc = descriptions_.row(i);
  return desc;
}

void Frame::SetKfId(const Qth qth, int kf_id) { 
  if(kf_id_.count(qth))
    throw  -1;
  kf_id_[qth] = kf_id;
}


const int Frame::GetKfId(const Qth qth) const { 
  if(kf_id_.count(qth))
    return kf_id_.at(qth);
  return -1;
}

Mappoint::Mappoint(Ith id, Instance* ins)
  : id_(id), ins_(ins)
{
}

void Mappoint::AddReferenceKeyframe(const Qth& qth, Frame* ref, const Eigen::Vector3d& Xq) {
  ref_[qth] = ref;
  SetXq(qth, Xq);
}
void Mappoint::SetXr(const Qth& qth, const Eigen::Vector3d& Xr) {
  Xr_[qth] = std::shared_ptr<Eigen::Vector3d>(new Eigen::Vector3d(Xr));
  return;
}

const Eigen::Vector3d& Mappoint::GetXr(const Qth& qth) const {
  return *Xr_.at(qth);
}

void Mappoint::SetXq(Qth qth, const Eigen::Vector3d& Xq) {
  Xq_[qth] = std::shared_ptr<Eigen::Vector3d>(new Eigen::Vector3d(Xq));
}

const Eigen::Vector3d& Mappoint::GetXq(Qth qth) {
  if(!Xq_.count(qth) ){
    //Frame* ref = ref_.at(qth);
    //const auto Tqc = ref->GetTcq(qth).inverse();
    //Xq_[qth] = Tqc * (*Xr_.at(qth));
  }
  return *Xq_.at(qth);
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

bool RigidGroup::ExcludeInstance(Instance* ins) {
  const Pth pth = ins->GetId();
  bool count = included_instances_.count(pth);
  if(count) 
    included_instances_.erase(pth);
  if(ins->rig_groups_.count(id_))
    ins->rig_groups_.erase(id_);
  else
   count = false;;
  excluded_instances_[pth] = ins;
  return count;
}

void RigidGroup::ExcludeMappoint(Mappoint* mp) {
  excluded_mappoints_[mp->GetId()] = mp;
}

inline double lerp(double x0, double y0, double x1, double y1, double x) {
  return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

std::map<int, std::pair<Mappoint*, double> > FlowMatch(const Camera* camera,
                                                       const FeatureDescriptor* extractor,
                                                       const cv::Mat& flow0,
                                                       const Frame* prev_frame,
                                                       bool verbose,
                                                       Frame* curr_frame) {
  const double best12_threshold = .7;
  const double search_radius_min = 5.; // [pixel]
  const double far_depth         = 50.; // [meter] for min radius.
  const double search_radius_max = 20.; // [pixel]
  const double near_depth        = 15.; // [meter] for max radius
  const auto& keypoints0 = prev_frame->GetKeypoints();
  const auto& mappoints0 = prev_frame->GetMappoints();
  const auto& depths0    = prev_frame->GetMeasuredDepths();
  const auto& keypoints1 = curr_frame->GetKeypoints();
  cv::Mat dst0, dst1;
  if(verbose){
    dst0 = prev_frame->GetRgb().clone();
    dst1 = curr_frame->GetRgb().clone();
    for(int n=0; n<keypoints0.size(); n++)
      cv::circle(dst0,  keypoints0[n].pt, 3, CV_RGB(150,150,150), 1);
    for(int n=0; n<keypoints1.size(); n++)
      cv::circle(dst1,  keypoints1[n].pt, 3, CV_RGB(150,150,150), 1);
  }
  std::map<int, std::pair<Mappoint*, double> > matches;
  for(int n0=0; n0 < mappoints0.size(); n0++){
    Mappoint* mp0 = mappoints0[n0];
    if(!mp0)
      continue;
    const cv::KeyPoint& kpt0 = keypoints0[n0];
    const cv::Point2f&  pt0  = kpt0.pt;
    const double z = depths0[n0] < 1e-5 ? far_depth : depths0[n0];
    double search_radius = lerp(near_depth, search_radius_max, far_depth, search_radius_min, z);
    search_radius = std::max(search_radius, search_radius_min);
    search_radius = std::min(search_radius, search_radius_max);
    const auto& dpt01 = flow0.at<cv::Point2f>(pt0);
    Eigen::Vector2d eig_pt1(pt0.x+dpt01.x, pt0.y+dpt01.y);
    if(!curr_frame->IsInFrame(camera,eig_pt1))  //  땅바닥 제외
      continue;

    if(verbose){
      cv::Point2f pt1(eig_pt1.x(), eig_pt1.y());
      cv::line(  dst1, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(0,0,255), -1);
      cv::line(  dst0, pt0, pt1, CV_RGB(0,0,255), 1);
      cv::circle(dst0, pt0, 3,  CV_RGB(0,0,255), -1);
      char text[100];
      sprintf(text,"%2.1f",z);
      cv::putText(dst0, text, pt0, cv::FONT_HERSHEY_SIMPLEX, .4,  CV_RGB(255,0,0) );
      cv::circle(dst0, pt1, search_radius,  CV_RGB(0,255,0), 1);
    }

    std::list<int> candidates = curr_frame->SearchRadius(eig_pt1, search_radius);
    if(candidates.empty())
      continue;
    const cv::Mat desc0 = prev_frame->GetDescription(n0);
    double dist0 = 1e+9;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(const int& n1 : candidates){
      const cv::Mat desc1 = curr_frame->GetDescription(n1);
      const cv::KeyPoint& kpt1 = keypoints1[n1];
      double dist = extractor->GetDistance(desc0,desc1);
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = n1;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = n1;
      }
    }
    if(champ0 < 0)
      continue;
    {
      const cv::KeyPoint& kpt1 = keypoints1[champ0];
      if(std::abs(kpt0.angle - kpt1.angle) > 20.) // flow match는 낮은 angle 오차 허용.
        continue;
      if(kpt0.octave != kpt1.octave)
        continue;
    }
    bool c2 = dist0 < dist1 * best12_threshold;
    if(candidates.size() < 6 || c2  ){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(matches.at(champ0).second < dist0)
          continue;
      }
      matches[champ0] = std::make_pair(mp0,dist0);
    }
  }

  if(!dst1.empty()){ // Visualization
    for(auto it : matches){
      const int& i1 = it.first;
      const cv::Point2f& pt1 = curr_frame->GetKeypoint(i1).pt;
      const int& i0 = prev_frame->GetIndex(it.second.first);
      const cv::Point2f& pt0 = prev_frame->GetKeypoint(i0).pt;
      cv::line(  dst1, pt0, pt1, CV_RGB(255,255,0), 1);
      cv::circle(dst1, pt1, 3,  CV_RGB(255,255,0), 1);
    }
    char buffer[200];
    snprintf(buffer, sizeof(buffer), "Jth #%d, #%d", prev_frame->GetId(), curr_frame->GetId() );
    cv::putText(dst1, buffer, cv::Point(10, dst1.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1.,  CV_RGB(255,0,0), 2);
    cv::imshow("flow 01", dst0);
    cv::imshow("flow 10", dst1);
  }
  return matches;
}


std::map<int, std::pair<Mappoint*,double> > ProjectionMatch(const Camera* camera,
                                                            const FeatureDescriptor* extractor,
                                                            const std::set<Mappoint*>& mappoints,
                                                            const Frame* curr_frame, // With predicted Tcq
                                                            const Qth qth,
                                                            double search_radius) {
  const bool use_latest_desc = false;
  const double best12_threshold = 1.;
  double* ptr_projections = new double[2 * mappoints.size()];
  std::map<int, Mappoint*> key_table;
  int n = 0;
  for(Mappoint* mp : mappoints){
    if(curr_frame->GetIndex(mp) >= 0){
      // Loopclosing이 2번 이상 한 지점에서 발생하면
      // LoopCloser::CombineNeighborMappoints()에서 발생하는 경우.
      continue;
    }
    const Eigen::Vector3d Xr = mp->GetXr(qth);
    Frame* ref = mp->GetRefFrame(qth);
    const g2o::SE3Quat& Trq = ref->GetTcq(qth);
    const Eigen::Vector3d Xc = curr_frame->GetTcq(qth) * Trq.inverse() * Xr;
    if(Xc.z() < 0.)
      continue;
    Eigen::Vector2d uv = camera->Project(Xc);
    if(!curr_frame->IsInFrame(camera,uv))  //  땅바닥 제외
      continue;
    ptr_projections[2*n]   = uv[0];
    ptr_projections[2*n+1] = uv[1];
    key_table[n++] = mp;
  }

  std::map<int, std::pair<Mappoint*,double> > matches;
  flann::Matrix<double> queries(ptr_projections, n, 2);
  std::vector<std::vector<int> > batch_search = curr_frame->SearchRadius(queries, search_radius);
  for(size_t key_mp = 0; key_mp < batch_search.size(); key_mp++){
    Mappoint* query_mp = key_table.at(key_mp);
    cv::Mat desc0; cv::KeyPoint kpt0;
    query_mp->GetFeature(qth, use_latest_desc, desc0, kpt0);
    const std::vector<int>& each_search = batch_search.at(key_mp);
    double dist0 = 999999999.;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(int idx : each_search){
      cv::Mat desc1 = curr_frame->GetDescription(idx);
      const cv::KeyPoint& kpt1 = curr_frame->GetKeypoint(idx);
      if(! curr_frame->IsInFrame(camera, Eigen::Vector2d(kpt1.pt.x, kpt1.pt.y)) )
        continue;
      double dist = extractor->GetDistance(desc0,desc1);
      if(std::abs(kpt0.angle - kpt1.angle) > 40.)
        continue;
      if(kpt0.octave != kpt1.octave)
        continue;
      if(dist < dist0){
        dist1 = dist0;
        champ1 = champ0;
        dist0 = dist;
        champ0 = idx;
      }
      else if(dist < dist1){
        dist1 = dist;
        champ1 = idx;
      }
    }

    if(champ0 < 0)
      continue;
    if(dist0 < dist1 * best12_threshold){
      if(matches.count(champ0)){
        // 이미 matching candidate가 있는 keypoint가 선택된 경우,
        // 오차를 비교하고 교체 여부 결정
        if(matches.at(champ0).second < dist0)
          continue;
      }
      matches[champ0] = std::make_pair(query_mp, dist0);
    }
  }
  delete[] ptr_projections;
  return matches;
}

Pipeline::Pipeline(const Camera* camera,
                   FeatureDescriptor*const extractor
                  )
  : camera_(camera), prev_frame_(nullptr), prev_dominant_qth_(-1), extractor_(extractor), mapper_(new Mapper()) {
}

Pipeline::~Pipeline() {

}
  
void Pipeline::UpdateRigGroups(const Qth& dominant_qth,
                               Frame* frame) const {
  std::set<Instance*> nongroup_instances;
  for(Instance* ipt : frame->GetInstances() ){
    if(!ipt)
      continue;
    if(ipt->rig_groups_.empty())
      nongroup_instances.insert(ipt);
  }
  RigidGroup* rig = qth2rig_groups_.at(dominant_qth);    
  for(Instance* ins : nongroup_instances)
    rig->AddInlierInstace(ins);
  return;
}

void SetMatches(std::map<int, std::pair<Mappoint*, double> >& flow_matches,
                std::map<int, std::pair<Mappoint*, double> >& proj_matches,
                Frame* curr_frame){
  std::map<Mappoint*,int> matched_mappoints;
  for(auto it : flow_matches){
    curr_frame->SetMappoint(it.second.first, it.first);
    matched_mappoints[it.second.first] = it.first;
  }

  for(auto it : proj_matches){
    const int& prj_n = it.first;
    Mappoint*const prj_mp = it.second.first;
    bool c1 = flow_matches.count(prj_n);
    bool c2 = matched_mappoints.count(prj_mp);
    if(c1 && c2) // flow match 와 결과가 똑같은 경우.
      continue;
    else if(c1){
      if( flow_matches.at(prj_n).second < it.second.second ) // 기존 matching이 더 description error가 적을 경우.
        continue;
      else{
        Mappoint* mp0 = curr_frame->GetMappoint(prj_n);
        curr_frame->EraseMappoint(prj_n);
        flow_matches.erase(prj_n);
        matched_mappoints.erase(mp0);
      }
    }
    else if(c2){
      const int& flow_n = matched_mappoints.at(prj_mp);
      if(flow_matches.at(flow_n).second < it.second.second)
        continue;
      else{
        curr_frame->EraseMappoint(flow_n);
        flow_matches.erase(flow_n);
        matched_mappoints.erase(prj_mp);
      }
    }
    curr_frame->SetMappoint(prj_mp, prj_n);
  }
  return;
}

std::vector< std::pair<Qth, size_t> > CountRigKeypoints(Frame* frame,
                                                     const Camera* camera,
                                                     bool fill_bg_with_dominant,
                                                     Qth prev_dominant_qth,
                                                     const std::map<Qth,RigidGroup*> qth2rig_groups
                                                     ){
  std::map<Qth, size_t> num_points; {
    const auto& instances = frame->GetInstances();
    const auto& keypoints = frame->GetKeypoints();
    for(size_t n=0; n<instances.size(); n++){
      Instance* ins = instances[n];
      if(!ins)
        continue;
      const cv::Point2f& pt = keypoints[n].pt;
      if(! frame->IsInFrame(camera, Eigen::Vector2d(pt.x, pt.y) ) ) // 화면 구석에만 몰린 instance와 rig group은 tracking에서 제외한다.
        continue;
      for(auto it_rig : ins->rig_groups_)
        num_points[it_rig.first]++;
    }
  }
  std::vector< std::pair<Qth, size_t> >  sorted_results;
  sorted_results.reserve(num_points.size());
  for(auto it : num_points)
    sorted_results.push_back(std::pair<Qth, size_t>(it.first, it.second) );
  std::sort(sorted_results.begin(), sorted_results.end(),
            [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b)
              { return a.second > b.second; }
           );
  if(fill_bg_with_dominant){
    Qth dominant = sorted_results.empty() ? prev_dominant_qth : sorted_results.begin()->first;
    RigidGroup* dominant_rig = qth2rig_groups.at(dominant);
    auto& instances = frame->GetInstances();
    for(size_t n=0; n<instances.size(); n++){
      Instance*& ins = instances[n];
      if(ins)
        continue;
      ins = dominant_rig->GetBgInstance();
    }
  }
  return sorted_results;
}

void GetMappoints4Qth(Frame* frame,
                      const Qth& qth,
                      std::set<Mappoint*>& _mappoints
                      ){
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    if(!ins)
      continue;
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    if(!ins->rig_groups_.count(qth))
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
  int min_kf_id = keyframe->GetKfId(qth) - 10; // TODO frame 간격이 아니라 keyframe 간격
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
  const auto& instances = frame->GetInstances();
  std::map<Qth, size_t> n_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    if(!ins)
      continue;
    Mappoint* mpt = mappoints[n];
    if(mpt){
      for(auto it_rig : ins->rig_groups_)
        n_mappoints[it_rig.first]++;
    }
  }
  return n_mappoints;
}

void CountMappoints(Frame* frame,
                    std::map<Qth, std::set<Pth> >& segmented_instances,
                    std::map<Qth, size_t>& rig2bgmappoints,
                    std::map<Pth, size_t>& ins2mappoints){
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    if(!ins)
      continue;
    Pth pth = ins->GetId();
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    for(auto it_rig : ins->rig_groups_){
      segmented_instances[it_rig.first].insert(ins->GetId());
      if(pth < 0)
        rig2bgmappoints[it_rig.first]++;
    }
    if(pth > -1)
      ins2mappoints[pth]++;
  }
}

std::map<Qth,bool> Pipeline::FrameNeedsToBeKeyframe(Frame* curr_frame) const {
  std::map<Qth, size_t> curr_n_mappoints = CountMappoints(curr_frame);
  std::map<Qth,bool> need_keyframes;
  for(auto it : curr_n_mappoints){
    const Qth& qth = it.first;
    Frame* lkf = keyframes_.at(qth).rbegin()->second; // Not count 일 경우?
    if(lkf == curr_frame){
      need_keyframes[qth] = true; // ComputLBA 호출하는 while loop에서 qth rig가 생성됨.
      continue;
    }
    std::map<Qth, size_t> lkf_n_mappoints = CountMappoints(lkf);
    const size_t& n_mappoints_curr = it.second;
    if(!lkf_n_mappoints.count(qth) ){
      need_keyframes[qth] = true;
      continue;
    }
    const size_t& n_mappoints_lkf = lkf_n_mappoints.at(qth);
    size_t min_mpt_threshold = std::max<size_t>(10, .5 * n_mappoints_lkf);
    //printf("KF test. Jth %d, Qth %d,  %ld / %ld (%ld)\n", curr_frame->GetId(), qth,n_mappoints_curr, min_mpt_threshold, n_mappoints_lkf);
    need_keyframes[qth] = n_mappoints_curr < min_mpt_threshold;
  }

  return need_keyframes;
}

/*
void AddKeyframesAtMappoints(Frame* keyframe) {
  // ins->rig_qth 가 일하는 mappoint에 한해서 mpt->AddKeyframe(qth, frame)
  const auto& mappoints = keyframe->GetMappoints();
  const auto& instances = keyframe->GetInstances();
  for(size_t n=0; n<mappoints.size(); n++) {
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    for(auto it_ref : mpt->GetRefFrames() )
      mpt->AddKeyframe(it_ref.first, keyframe);
  }
  return;
}
*/

Eigen::Vector3d ComputeXr(float z, const Eigen::Vector3d& normalized_pt) {
  if(z < 1e-5) // No depth too far
    z = 1e+2;
  return z*normalized_pt;
}

void Pipeline::SupplyMappoints(const Qth& qth, Frame* frame) {
  const double min_mpt_distance = 10.;
  const auto& keypoints = frame->GetKeypoints();
  const auto& instances = frame->GetInstances();
  const auto& depths    = frame->GetMeasuredDepths();
  const auto& mappoints = frame->GetMappoints();
  RigidGroup* rig = qth2rig_groups_.at(qth);

  for(int n=0; n < keypoints.size(); n++){
    Instance* ins = instances[n];
    if(! rig->IsInlierInstances(ins) )
      continue;
    const Eigen::Vector3d Xr = ComputeXr(depths[n], frame->GetNormalizedPoint(n));
    const Eigen::Vector3d Xq = frame->GetTcq(qth).inverse()*Xr;
    if(Mappoint* mp = mappoints[n]){
      if(!mp->GetRefFrames().count(qth)){
        // 기존 mp에, 새로운 qth RigidGroup을 위한  Xr, Xq를 입력
        mp->SetXr(qth, Xr);
        mp->AddReferenceKeyframe(qth, frame, Xq);
      }
      mp->AddKeyframe(qth, frame);
      continue;
    }
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
    static Ith nMappoints = 0;
    if(!ins)
      throw -1;
    Mappoint* mp = new Mappoint(nMappoints++, ins);
    mp->SetXr(qth, Xr);  // new mappoint를 위한 reference frame 지정.
    mp->AddReferenceKeyframe(qth, frame, Xq);
    mp->AddKeyframe(qth, frame);
    frame->SetMappoint(mp, n);
    ith2mappoints_[mp->GetId()] = mp;
  }

  bool verbose = false;
  if(verbose){
    cv::Mat dst = frame->GetRgb().clone();
    for(int n=0; n<keypoints.size(); n++){
      const auto& kpt = keypoints[n];
      const Mappoint* mp = mappoints[n];
      float z = depths[n];
      if(mp){
        if( z > 1e-2)
          cv::circle(dst,kpt.pt,5,CV_RGB(0,255,0),-1);
        else
          cv::circle(dst,kpt.pt,5,CV_RGB(255,0,0),-1);
      }
      else
        cv::circle(dst,kpt.pt,5,CV_RGB(150,150,150),1);
    }
    cv::imshow("supply mappoints "+std::to_string(qth), dst);
  }
  return;
}

cv::Mat Pipeline::VisualizeRigInfos(Frame* frame,
                                    const Qth& qth,
                                    const std::map<Jth, Frame* >& neighbor_frames,
                                    const std::set<Mappoint*>& neighbor_mappoints,
                                    std::set<Pth> given_instances,
                                    const float& switch_threshold,
                                    const std::map<Pth,float>& switch_states,
                                    const std::map<Pth,ShapePtr>& curr_shapes
                                   ) const {
  const cv::Mat rgb = frame->GetRgb();
  RigidGroup* rig = qth2rig_groups_.at(qth);
  cv::Mat dst_frame; {
    cv::Mat dst_fill  = rgb.clone();
    std::set<Pth> excluded;
    for(auto it : rig->GetExcludedInstances())
      excluded.insert(it.first);
    for(auto it : switch_states){
      if(it.second > switch_threshold)
        continue;
      excluded.insert(it.first);
    }
    for(auto pth : excluded)
      if(given_instances.count(pth))
        given_instances.erase(pth);

    for(Pth pth : given_instances){
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(0,255,0), 3);
    }
    for(Pth pth : excluded){
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(255,0,0), -1);
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(0,255,0), 3);
    }
    cv::addWeighted(rgb, .5, dst_fill, .5, 1., dst_frame);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .3; int fontThick = 1; int baseline = 0;
    for(Pth pth : given_instances){
      if(!curr_shapes.count(pth))
        continue;
      auto msg = std::to_string(pth);
      const auto ptr = curr_shapes.at(pth);
      const auto& bb = ptr->outerior_bb_;
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      auto size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
      cv::Point dpt(.5*size.width, .5*size.height);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      cv::putText(dst_frame, msg, cp-dpt, fontFace, fontScale, CV_RGB(0,0,0));
    }
    for(Pth pth : excluded){
      if(!curr_shapes.count(pth))
        continue;
      auto msg = std::to_string(pth);
      const auto ptr = curr_shapes.at(pth);
      const auto& bb = ptr->outerior_bb_;
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      auto size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
      cv::Point dpt(.5*size.width, .5*size.height);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      cv::putText(dst_frame, msg, cp-dpt, fontFace, fontScale, CV_RGB(0,0,0));
    }
    for(Mappoint* mp : neighbor_mappoints){
      int n = frame->GetIndex(mp);
      if(n < 0)
        continue;
      cv::KeyPoint kpt = frame->GetKeypoint(n);
      cv::circle(dst_frame, kpt.pt, 3, CV_RGB(0,255,0) );
    }
  }
  cv::pyrDown(dst_frame, dst_frame);

  cv::Mat dst_texts = cv::Mat::zeros( dst_frame.rows, 400, CV_8UC3); {
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .4;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    /* Neighbor frame을 여기에 나열. */
    std::map<Jth, Frame*> keyframes = neighbor_frames;
    keyframes[frame->GetId()] = frame;
    for(auto it_kf : keyframes){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      auto t = kf->GetTcq(qth).inverse().translation();
      char text[100];
      sprintf(text,"est : F#%2d, KF#%2d, (%4.3f, %4.3f, %4.3f)", kf->GetId(), kf->GetKfId(qth), t[0], t[1], t[2] );
      auto size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }

    y+= 10;
    std::ostringstream buffer;
    buffer << "Include> ";
    for(auto it : rig->GetIncludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 3;
    buffer.str("");
    buffer << "Exclude> ";
    for(auto it : rig->GetExcludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 10;
    buffer.str("");
    buffer << "Pth:Switch state> ";
    for(auto it : switch_states){
      buffer << "(" << it.first <<  ":" << std::fixed << std::setprecision(3) << it.second << "), ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 100)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
  }

  cv::Mat dst = cv::Mat::zeros(std::max<int>(dst_frame.rows, dst_texts.rows),
                               dst_frame.cols + dst_texts.cols,
                               CV_8UC3);

  {
    cv::Rect rect(0, 0, dst_frame.cols, dst_frame.rows);
    cv::Mat roi(dst, rect);
    dst_frame.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  {
    cv::Rect rect(dst_frame.cols,0,
                  dst_texts.cols, dst_texts.rows);
    cv::Mat roi(dst, rect);
    dst_texts.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  return dst;
}

cv::Mat Pipeline::VisualizeStates(Frame* frame,
                        const std::map<Pth,float>& density_scores,
                        const std::map<Pth, float>& switch_states,
                        const float& switch_threshold,
                        const std::map<Jth, Frame* >& neighbor_frames,
                        const std::map<Pth,ShapePtr>& curr_shapes,
                        const cv::Mat& outline_mask,
                        const EigenMap<int,g2o::SE3Quat>* gt_Tcws
                        ) const {
  const Qth dominant_qth = 0;
  const RigidGroup* dominant_rig = qth2rig_groups_.at(dominant_qth);
  const cv::Mat rgb = frame->GetRgb();
  cv::Mat dst_frame; {
    cv::Mat dst_fill  = rgb.clone();
    std::set<Pth> excluded;
    for(auto it : dominant_rig->GetExcludedInstances())
      excluded.insert(it.first);
    for(auto it : switch_states){
      if(it.second > switch_threshold)
        continue;
      excluded.insert(it.first);
    }

    for(auto it : excluded){
      const Pth& pth = it;
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(255,0,0), -1);
    }
    cv::addWeighted(rgb, .5, dst_fill, .5, 1., dst_frame);
    cv::Mat dst_outline;
    cv::cvtColor(outline_mask, dst_outline, cv::COLOR_GRAY2BGR);
    cv::addWeighted(dst_frame, 1., dst_outline, .3, 1., dst_frame);
  }

  for(auto it : curr_shapes) {
    const Pth& pth = it.first;
    //bool sparse = density_scores.count(pth) ? density_scores.at(pth) < 1. : true;
    //bool dynamics = switch_states.count(pth) ? switch_states.at(pth) > switch_threshold : false;
    ShapePtr s_ptr = it.second;
    if(s_ptr->n_missing_ > 0)
      continue;
    if(s_ptr->n_belief_ < 2)
      continue;
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
    const auto& color = colors[pth % colors.size()];
    cv::drawContours(dst_frame, cnts, 0, color, 2);
  }

  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    Mappoint* mp = mappoints[n];
    const cv::Point2f& pt = keypoints[n].pt;
    if(mp){
      std::stringstream ss;
      ss << std::hex << mp->GetId(); // Convert to hexadecimal
      std::string msg = ss.str();
      if(mp->HasEstimate4Rig(dominant_qth)) {
        const std::set<Frame*>& keyframes = mp->GetKeyframes(dominant_qth);
        for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
          const cv::Point2f& pt0 = (*it)->GetKeypoint( (*it)->GetIndex(mp) ).pt;
          cv::circle(dst_frame, pt, 3, CV_RGB(0,255,0), -1);
          cv::line(dst_frame, pt, pt0, CV_RGB(255,255,0),  1);
          break;
        }
      }
      else
        cv::circle(dst_frame, pt, 3, CV_RGB(255, -1,0), -1);
    }
    else {
      cv::circle(dst_frame, pt, 2, CV_RGB(150,150,150), 1);
    }
  }
  {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .3;
    int fontThick = 1;
    int baseline = 0;
    for(auto it : curr_shapes){
      const Pth& pth = it.first;
      ShapePtr ptr = it.second;
      const auto& bb = ptr->outerior_bb_;
      std::string msg[3];
      cv::Scalar txt_colors[3];
      msg[0] = "#" + std::to_string(pth);
      for(int k=0; k<3;k++)
        txt_colors[k] = CV_RGB(0,0,0);
      msg[1] = "d: "; msg[2] = "s: ";
      if(density_scores.count(pth) ){
        std::ostringstream oss; oss << std::fixed << std::setprecision(2);
        oss<< density_scores.at(pth);
        msg[1] += oss.str();
        if(density_scores.at(pth) < 1.)
          txt_colors[1] = CV_RGB(255,0,0);
      }
      else
        msg[1] += "-";
      if(switch_states.count(pth) ){
        std::ostringstream oss; oss << std::fixed << std::setprecision(2);
        oss<< switch_states.at(pth);
        msg[2] += oss.str();
        if(switch_states.at(pth) < switch_threshold)
          txt_colors[2] = CV_RGB(255,0,0);
      }
      else
        msg[2] += "-";
      int offset = 2;
      int w, h;
      w = h = 0;
      for(int k=0; k<3; k++){
        auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
        h += size.height;
        w = std::max(size.width, w);
      }
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      cv::Point dpt(.5*w, .5*h);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);

      int x = cp.x - .5*w;
      int y = cp.y - .5*h;
      for(int k=0;k<3;k++){
        cv::putText(dst_frame, msg[k], cv::Point(x,y),fontFace, fontScale, txt_colors[k], fontThick);
        auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
        y += size.height+offset;
      }

    }
  }

  cv::Mat dst_texts = cv::Mat::zeros( rgb.rows, 400, CV_8UC3); {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .4;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    std::map<Jth, Frame*> keyframes = neighbor_frames;
    keyframes[frame->GetId()] = frame;
    Frame* prev_kf = nullptr;
    for(auto it_kf : keyframes){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      auto t = kf->GetTcq(dominant_qth).inverse().translation();
      char text[100];

      sprintf(text,"est : F#%2d, KF#%2d  (%4.3f, %4.3f, %4.3f)", kf->GetId(), kf->GetKfId(dominant_qth), t[0], t[1], t[2] );
      auto size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);

      if(!gt_Tcws)
        continue;
      const g2o::SE3Quat& Tcw = gt_Tcws->at(jth);
      t = Tcw.inverse().translation();
      sprintf(text,"true: F#%2d,  (%4.3f, %4.3f, %4.3f)", kf->GetId(), t[0], t[1], t[2] );
      size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(120,120,120), thickness);

      if(!prev_kf){
        prev_kf = kf;
        continue;
      }
      // dT =Tc0 c1 = Tc0 w * Tc1 w.inverse()
      g2o::SE3Quat est_dT = prev_kf->GetTcq(dominant_qth) * kf->GetTcq(dominant_qth).inverse();
      g2o::SE3Quat true_dT = gt_Tcws->at(prev_kf->GetId()) * gt_Tcws->at(kf->GetId()).inverse();
      g2o::SE3Quat err_dT = true_dT.inverse() * est_dT;
      if(true_dT.translation().norm() < 1e-2)
        continue;
      float norm_ratio = est_dT.translation().norm() / true_dT.translation().norm();
      float err_ratio = err_dT.translation().norm() / true_dT.translation().norm();
      sprintf(text,"norm ratio: %4.3f", norm_ratio);
      cv::putText(dst_texts, text, cv::Point(20+size.width, y), fontFace, fontScale, CV_RGB(255,0,0), thickness);

      y += 2;
      prev_kf = kf;
    }

    y+= 10;
    std::ostringstream buffer;
    buffer << "Include> ";
    for(auto it : dominant_rig->GetIncludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 3;
    buffer.str("");
    buffer << "Exclude> ";
    for(auto it : dominant_rig->GetExcludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }

    y+= 10;
    buffer.str("");
    buffer << "Pth:Switch state> ";
    for(auto it : switch_states){
      buffer << "(" << it.first <<  ":" << std::fixed << std::setprecision(3) << it.second << "), ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 100)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
  }

  cv::Mat dst = cv::Mat::zeros(std::max<int>(dst_frame.rows, dst_texts.rows),
                               dst_frame.cols + dst_texts.cols,
                               CV_8UC3);

  {
    cv::Rect rect(0, 0, dst_frame.cols, dst_frame.rows);
    cv::Mat roi(dst, rect);
    dst_frame.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  {
    cv::Rect rect(dst_frame.cols,0,
                  dst_texts.cols, dst_texts.rows);
    cv::Mat roi(dst, rect);
    dst_texts.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }

  return dst;
}

cv::Mat GetClosedOutlineMask(const cv::Size size, const std::map<Pth, ShapePtr>& curr_shapes){
  cv::Mat outline_mask = cv::Mat::zeros(size, CV_8UC1);
  for(auto it_shape : curr_shapes){
    ShapePtr s_ptr = it_shape.second;
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(s_ptr->outerior_.size() );
    for( auto pt: s_ptr->outerior_)
      cnts[0].push_back(cv::Point(pt.x,pt.y));
    cv::drawContours(outline_mask, cnts, 0, 255, 20);
  }
  return outline_mask;
}

void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const cv::Mat flow0,
                   const std::map<Pth, ShapePtr>& curr_shapes,
                   const cv::Mat& gradx, const cv::Mat& grady, const cv::Mat& valid_grad,
                   const cv::Mat vis_rgb,
                   const EigenMap<int, g2o::SE3Quat>* gt_Tcws
                   )
{
  const bool fill_bg_with_dominant = true;
  const float search_radius = 30.;
  const float switch_threshold = .3;
  float density_threshold = 1. / 70. / 70.; // NxN pixel에 한개 이상의 mappoint가 존재해야 dense instance
  const bool verbose_flowmatch = false;

  cv::Mat outline_mask = GetClosedOutlineMask(gray.size(), curr_shapes);
  for(auto it_shape : curr_shapes)
    if(!pth2instances_.count(it_shape.first) )
      pth2instances_[it_shape.first] = new Instance(it_shape.first);

  static Jth nFrames = 0;
  Frame* curr_frame = new Frame(nFrames++, vis_rgb);
  curr_frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_, outline_mask);
  curr_frame->SetInstances(curr_shapes, pth2instances_);
  curr_frame->SetMeasuredDepths(depth);
#if 1
  Qth dominant_qth = -1; {
    // Initiate pipeline for first frame.
    auto& instances = curr_frame->GetInstances();
    if(!prev_frame_){
      RigidGroup* rig_new = new RigidGroup(0, curr_frame);
      qth2rig_groups_[rig_new->GetId()] = rig_new;
      const Qth& qth = rig_new->GetId();
      curr_frame->SetTcq(rig_new->GetId(), g2o::SE3Quat() );
      for(size_t n=0; n<instances.size(); n++)
        if(!instances[n])
          instances[n] = rig_new->GetBgInstance(); // Fill bg instance
      std::set<Qth> curr_rigs = {qth, };
      UpdateRigGroups(qth, curr_frame);
      SupplyMappoints(rig_new->GetId(), curr_frame);

      keyframes_[qth][curr_frame->GetId()] = curr_frame;
      curr_frame->SetKfId(qth, keyframes_.at(qth).size()-1 ); // kf_id는 0이어야한다.
      every_keyframes_.insert(curr_frame->GetId());
      prev_frame_ = curr_frame;
      prev_dominant_qth_ = qth;
      return;
    } 

    // Set non instance points as a member of dominant rigid body group.
    RigidGroup* dominant_rig = qth2rig_groups_.at(0);// TODO 일반화.
    dominant_qth = dominant_rig->GetId();
    for(size_t n=0; n<instances.size(); n++)
      if(!instances[n])
        instances[n] = dominant_rig->GetBgInstance(); // Fill bg instance
    UpdateRigGroups(dominant_qth, curr_frame);
  }
#endif

  // intiial pose prediction
  const auto& Tcqs = prev_frame_->GetTcqs();
  for(auto it : Tcqs)
    curr_frame->SetTcq(it.first, *it.second);

  // Optical flow를 기반으로한 feature matching은 Tcq가 필요없다.
  std::map<int, std::pair<Mappoint*, double> > flow_matches
    = FlowMatch(camera_, extractor_, flow0, prev_frame_, verbose_flowmatch, curr_frame);
  std::map<Mappoint*,int> matched_mappoints;
  for(auto it : flow_matches){
    curr_frame->SetMappoint(it.second.first, it.first);
    matched_mappoints[it.second.first] = it.first;
  }

  std::map<Qth, std::set<Pth> > segmented_instances;
  std::map<Qth, size_t> rig2bgmappoints;
  std::map<Pth, size_t> ins2mappoints;
  CountMappoints(curr_frame, segmented_instances, rig2bgmappoints, ins2mappoints);
  if(segmented_instances.empty())
    throw -1;

  std::map<Pth,float> density_scores;
  for(auto it_shape : curr_shapes){
    const Pth& pth = it_shape.first;
    float npoints = ins2mappoints.count(pth) ? ins2mappoints.at(pth) : 0.;
    float dense = npoints > 0 ? npoints/it_shape.second->area_ : 0.; // 최소 점의 개수도 고려.
    density_scores[pth] = dense / density_threshold;
  }

  static std::map<Qth, std::map<Pth,size_t> > n_consecutiv_switchoff;
  int nq = 1;
  while(!segmented_instances.empty()){
    auto it_rig = segmented_instances.begin();
    const Qth qth = it_rig->first;
    std::set<Pth> instances = it_rig->second;
    segmented_instances.erase(it_rig);
    RigidGroup* rig = qth2rig_groups_.at(qth);
    Frame* latest_kf = keyframes_.at(qth).rbegin()->second;
    std::set<Mappoint*>     neighbor_mappoints;
    std::map<Jth, Frame* >  neighbor_frames;
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    if(neighbor_mappoints.empty()) {
      std::cerr << "qth = " << qth << ", failure to get mappoints" << std::endl;
      throw -1;
    }
#if 0
    // ProjectionMatch는 flow로부터 motion update와 pth0 != pth_curr 인 경우에 예외처리가 팔요해보인다.
    std::map<int, std::pair<Mappoint*, double> > proj_matches = ProjectionMatch(camera_, extractor_, neighbor_mappoints, curr_frame, qth, search_radius);
    SetMatches(flow_matches, proj_matches, curr_frame);
#endif
    std::set<Pth> fixed_instances;
    if(qth == dominant_qth)
      for(auto it_density : density_scores)
        if(it_density.second < 1.)
          fixed_instances.insert(it_density.first);

    bool vis_verbose = false;
    std::map<Pth,float> switch_states\
      = mapper_->ComputeLBA(camera_,qth, neighbor_mappoints, neighbor_frames,
                            curr_frame, prev_frame_, fixed_instances, gradx, grady, valid_grad, vis_verbose);
    std::set<Pth> instances4next_rig;
    std::set<Pth> switchoff_instances;
    if(!n_consecutiv_switchoff.count(qth))
      n_consecutiv_switchoff[qth]; // empty map 생성.
    for(auto it_switch : switch_states){
      const Pth& pth = it_switch.first;
      if(it_switch.second > switch_threshold)
        continue;
      size_t& n_consec = n_consecutiv_switchoff[qth][pth];
      n_consec++;
      if(n_consec > 2){ // N 번 연속 switch off 판정을 받은경우,
        Instance* ins = pth2instances_.at( pth );
        rig->ExcludeInstance(ins);
        printf("Exclude Q#%d P#%d", qth, pth);
        instances4next_rig.insert(pth);
      }
      else
        switchoff_instances.insert( pth); // 현재 프레임에서 switchoff가 발생하지 않은 pthqth를 삭제하기위해.
    }

    int nmappoints = 0;
    for(Pth pth : instances4next_rig)
      if(ins2mappoints.count(pth))// mp 없는 instance도 있으니까.
         nmappoints += ins2mappoints.at(pth);

    Qth qth_next = -1;
    if(segmented_instances.empty()){
#if 1
      if(nmappoints > 10) {
#else
      if(nmappoints > 1){ // neighbor_mappoints.empty()를 예방하기 위해 필요한 최소 조건.
#endif
        // 새로운 rig를 생성
        qth_next = qth2rig_groups_.rbegin()->first + 1;
        RigidGroup* rig_new = new RigidGroup(qth_next, curr_frame);
        qth2rig_groups_[rig_new->GetId()] = rig_new;
        curr_frame->SetTcq(rig_new->GetId(), g2o::SE3Quat() );
        for(Pth pth : instances4next_rig){
          Instance* ins = pth2instances_.at(pth);
          rig_new->AddInlierInstace(ins);
          segmented_instances[qth_next].insert(pth);
        }
        keyframes_[qth_next][curr_frame->GetId()] = curr_frame;
        curr_frame->SetKfId(qth_next, keyframes_.at(qth_next).size()-1); // kf_id 0이 필요
        const auto& instances = curr_frame->GetInstances();
        const auto& depths    = curr_frame->GetMeasuredDepths();
        const auto& mappoints = curr_frame->GetMappoints();
        for(int n=0; n < mappoints.size(); n++){
          Mappoint* mp = mappoints[n];
          if(!mp)
            continue;
          Instance* ins = instances[n];
          if(!ins)
            continue;
          if(!instances4next_rig.count(ins->GetId()))
            continue;
          Eigen::Vector3d Xr = ComputeXr(depths[n], curr_frame->GetNormalizedPoint(n));
          mp->SetXr(qth_next, Xr);
          const Eigen::Vector3d Xq = curr_frame->GetTcq(qth_next).inverse()*Xr;
          mp->AddReferenceKeyframe(qth_next, curr_frame, Xq);
          mp->AddKeyframe(qth_next, curr_frame);
        }
      }
    } 
    else { // if(segmented_instances.empty()
      auto it_rig_next = segmented_instances.begin();
      qth_next = it_rig_next->first;
      RigidGroup* rig_next = qth2rig_groups_.at(qth_next);
      for(Pth pth : instances4next_rig){
        Instance* ins = pth2instances_.at(pth);
        rig_next->AddInlierInstace(ins);
        segmented_instances[qth_next].insert(pth);
      }
    }

    if(false){
      std::set<Pth> missing_outlier;
      for(auto it_conseq_so : n_consecutiv_switchoff.at(qth) ){
        const Pth& pth = it_conseq_so.first;
        if(! switchoff_instances.count(pth) )
          missing_outlier.insert(pth);
      }
      for(const Pth& it : missing_outlier)
        n_consecutiv_switchoff[qth].erase(it);
      if(n_consecutiv_switchoff.count(qth) && n_consecutiv_switchoff.at(qth).empty())
        n_consecutiv_switchoff.erase(qth);
    }

    if(qth == 0){
      cv::Mat dst = VisualizeStates(curr_frame, density_scores, switch_states, switch_threshold, neighbor_frames,
                                    curr_shapes, outline_mask, gt_Tcws);
      cv::imshow("segslam", dst);
    } else {
      cv::Mat dst = VisualizeRigInfos(curr_frame, qth, neighbor_frames, neighbor_mappoints,
                                      instances, switch_threshold, switch_states, curr_shapes);
      cv::imshow("Rig #"+std::to_string(qth), dst);
    }
    if(++nq > 3) // N 번만 수행.
      break;
  }

  static bool stop = false;
  char c = cv::waitKey(stop?0:1);
  if(c == 'q')
    exit(1);
  else if (c == 's')
    stop = !stop;

  std::map<Qth,bool> need_keyframe = FrameNeedsToBeKeyframe(curr_frame);
  for(auto it : need_keyframe){
    if(!it.second)
      continue;
    SupplyMappoints(it.first, curr_frame);
    every_keyframes_.insert(curr_frame->GetId());
    keyframes_[it.first][curr_frame->GetId()] = curr_frame;
    if(keyframes_.at(it.first).size() > 1)// ComputLBA 를 포함한 while loop문에서 이미 추가된 keyframe.
      curr_frame->SetKfId(it.first, keyframes_[it.first].size() );
  }

  prev_frame_ = curr_frame;
  prev_dominant_qth_ = dominant_qth;
  return;
}

    /* 
     * [ ] 3D visualization
     * Mappoint가 가리키는 instance가 많이 겹치는 새로운 ins에 대해, 병합처리가 필요할것은데??
     - Mappoint의 instance는 생성될 시점의 ID를 가지더라도, 현재의 instance ID는 다를 수 있다.
     주어진 shape에 해당 instance가 없다면 matching이 가장 많은 케이스와 연결해야,. 적절한 visualzation 가능.
     *  Futureworks? : SupplyRigGroup을 frame당 한번만 하는게 타당한가? LBA에서 필터링 하고나서 소속없는 instance 모두 모아서 새로 만드는거 반복해야하지 않냐?
     - 일단은 Compute LBA 반복하다가, lba 결과물의 rigid body의 크기가 일정값 이하가 될때 반복 종료하는것으로 대충 끝내기.
     - 나머지 '자잘' 한것은 그냥 독립된 instance로 취급.
     - '자잘' 한것은 일단 medain length tracking으로 관찰하다가 일정프레임 이상 유지되면 병합.
     */


} // namespace seg
