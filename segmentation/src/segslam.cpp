#include "segslam.h"
#include "Eigen/src/Core/Matrix.h"
#include "camera.h"
#include "frame.h"
#include "optimizer.h"
#include "orb_extractor.h"
#include "seg.h"
#include "util.h"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>

namespace seg {

OrbSlam2FeatureDescriptor::OrbSlam2FeatureDescriptor(int nfeatures, float scale_factor, int nlevels, int initial_fast_th, int min_fast_th)
  : FeatureDescriptor()
{
  auto ptr = new ORB_SLAM2::ORBextractor(nfeatures, scale_factor, nlevels, initial_fast_th, min_fast_th);
  extractor_ = std::shared_ptr<ORB_SLAM2::ORBextractor>(ptr);
}

void OrbSlam2FeatureDescriptor::Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  extractor_->extract(gray, mask, keypoints, descriptors);
  return;
}

double OrbSlam2FeatureDescriptor::GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const {
  return ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
}

CvFeatureDescriptor::CvFeatureDescriptor() 
  : FeatureDescriptor() {
  int 	nfeatures = 4000;
  float scaleFactor = 1.2f;
  int 	nlevels = 3;
  int 	edgeThreshold = 15;
  int 	firstLevel = 0;
  int 	WTA_K = 2;
  int 	scoreType = cv::ORB::HARRIS_SCORE;
  int 	patchSize = 31;
  int 	fastThreshold = 15;
  orb_ = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
}

void CvFeatureDescriptor::Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  orb_->detect(gray, keypoints);
  orb_->compute(gray, keypoints, descriptors);
  return;
}

double CvFeatureDescriptor::GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const {
  return ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
}

RigidGroup::RigidGroup(Qth qth, Frame* first_frame)
: id_(qth),
  bg_instance_(new Instance(-1))
{
  IncludeInstance(bg_instance_);
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
  if( uv.y() < 20. )
    return false;
  if( uv.x() < 20. )
    return false;
  if( width-uv.x() < 20. )
    return false;
  if( height - uv.y() < 50. ) // 땅바닥에서 결과가 너무 안좋았다.
    return false;
  return true;
}
int Frame::GetIndex(const Mappoint* mp) const {
  if(mappoints_index_.count(mp))
    return mappoints_index_.at(mp);
  return -1;
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

std::set<int> Frame::SearchRadius(const Eigen::Vector2d& uv, double radius) const {
  flann::Matrix<double> quary((double*)uv.data(), 1, 2);
  std::set<int> inliers;
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;
  flann_kdtree_->radiusSearch(quary, indices, dists, radius*radius, param);
  for(int idx :  indices[0])
    inliers.insert(idx);
  return inliers;
}

std::vector<std::vector<int> > Frame::SearchRadius(const flann::Matrix<double>& points,
                                                   double radius) const {
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;
  flann_kdtree_->radiusSearch(points, indices, dists, radius*radius, param);
  return indices;
}

void Frame::ExtractAndNormalizeKeypoints(const cv::Mat gray,
                                         const Camera* camera,
                                         FeatureDescriptor* extractor) {
  extractor->Extract(gray, cv::noArray(), keypoints_, descriptions_);
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

std::map<Pth,float> Frame::SetInstances(const std::map<Pth, ShapePtr>& shapes,
                                  const std::map<Pth, Instance*>& pth2instance,
                                  float density_threshold
                                 ) {
  std::map<Pth, std::set<int> > ins2nset;
  for(size_t n=0; n<keypoints_.size(); n++){
    const cv::KeyPoint& kpt = keypoints_[n];
    for(auto it_shape : shapes){
      ShapePtr s_ptr = it_shape.second;
      if(s_ptr->n_missing_ > 0)
        continue;
      const auto& pt = kpt.pt;
      if(! s_ptr->HasCollision(pt.x, pt.y, true) )
        continue;
      ins2nset[it_shape.first].insert(n);
      break;
    }
  }
  std::map<Pth,float> density_scores;
  //std::cout << "DenseScore = (";
  for(auto it_nset : ins2nset){
    const Pth& pth = it_nset.first;
    ShapePtr shape = shapes.at(pth);
    float dense = (float) it_nset.second.size() / shape->area_;
    float score =  dense / density_threshold;
    density_scores[pth] = score;
    //std::cout << "(#" << pth << ":" << score <<"),";
  }
  //std::cout << std::endl;
  for(auto it : density_scores){
    if(it.second < 1.) // sparse instance는 keypoint에 할당하지 않는다.
      continue;
    const Pth& pth = it.first;
    for(const int& n : ins2nset.at(pth) )
      instances_[n] = pth2instance.at(pth);
  }
  return density_scores;
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

Mappoint::Mappoint(Ith id, Instance* ins)
  : id_(id), ins_(ins)
{
}

void Mappoint::AddReferenceKeyframe(const Qth& qth, Frame* ref) {
  ref_[qth] = ref;
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

cv::Mat Mappoint::GetDescription() const {
  Frame* ref = ref_.begin()->second;
  int kpt_idx = ref->GetIndex(this);
  return ref->GetDescription(kpt_idx);
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

bool RigidGroup::IncludeInstance(Instance* ins) {
  const Pth pth = ins->GetId();
  if(excluded_instances_.count(pth) )
    return false;
  ins->rig_groups_[id_] = this;
  included_instances_[pth] = ins;
  return true;
}

std::map<int, Mappoint*> ProjectionMatch(const Camera* camera,
                                         const FeatureDescriptor* extractor,
                                         const std::set<Mappoint*>& mappoints,
                                         const Frame* curr_frame, // With predicted Tcq
                                         const Qth qth,
                                         double search_radius) {
  std::map<int, Mappoint*> matches;
  const double best12_threshold = 0.5;
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

  flann::Matrix<double> quaries(ptr_projections, n, 2);
  std::vector<std::vector<int> > batch_search = curr_frame->SearchRadius(quaries, search_radius);
  std::map<int, double> distances;
  for(size_t key_mp = 0; key_mp < batch_search.size(); key_mp++){
    Mappoint* quary_mp = key_table.at(key_mp);
    cv::Mat desc0 = quary_mp->GetDescription();
    const std::vector<int>& each_search = batch_search.at(key_mp);
    double dist0 = 999999999.;
    double dist1 = dist0;
    int champ0 = -1;
    int champ1 = -1;
    for(int idx : each_search){
      cv::Mat desc1 = curr_frame->GetDescription(idx);
      double dist = extractor->GetDistance(desc0,desc1);
      //if(std::abs(kpt0.angle - kpt.angle) > 40.)
      //  continue;
      //if(kpt0.octave != kpt.octave)
      //  continue;
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
        if(distances.at(champ0) < dist0)
          continue;
      }
      matches[champ0] = quary_mp;
      distances[champ0] = dist0;
    }
  }
  delete[] ptr_projections;
  return matches;
}

Pipeline::Pipeline(const Camera* camera,
                   FeatureDescriptor*const extractor
                  )
  : camera_(camera), prev_frame_(nullptr), extractor_(extractor), mapper_(new Mapper()) {
}

Pipeline::~Pipeline() {

}
  
void Pipeline::UpdateRigGroups(const std::set<Qth>& curr_rigs, Frame* frame) const {
  std::set<Instance*> nongroup_instances;
  for(Instance* ipt : frame->GetInstances() ){
    if(!ipt)
      continue;
    if(ipt->rig_groups_.empty())
      nongroup_instances.insert(ipt);
  }
  for(const Qth& qth : curr_rigs){
    RigidGroup* rig = qth2rig_groups_.at(qth);    
    for(Instance* ins : nongroup_instances)
      rig->IncludeInstance(ins);
  }
  return;
}

RigidGroup* SupplyRigGroup(Frame* frame,
                           std::map<Qth, RigidGroup*>& rig_groups) {
  size_t N = 0;
  size_t n = 0;
  for(Instance* ipt : frame->GetInstances() ){
    if(!ipt)
      continue;
    N++;
    if(ipt->rig_groups_.empty()){
      n++;
    }
  }
  float ng_ratio = (float)n / (float) N;
  // TODO 제대로 n(qth) > 1 에서 문제없는것 확인하고 삭제.
#if 0
  if(ng_ratio < .2)
    return nullptr;
#else
  if(frame->GetId() > 0)
    return nullptr;
#endif
  static Qth nRiggroups = 0;
  RigidGroup* rig = new RigidGroup(nRiggroups++, frame);
  rig_groups[rig->GetId()] = rig;
  std::cout << "Add new group! #" << rig->GetId() << std::endl;
  return rig;
}

std::vector< std::pair<Qth, size_t> > CountRigPoints(Frame* frame,
                                                     const Camera* camera,
                                                     bool fill_bg_with_dominant,
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
    Qth dominant = sorted_results.begin()->first;
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
  // 1. frame에서 보이는 mappoints
  GetMappoints4Qth(keyframe, qth, neighbor_mappoints);
  // 2. '1'의 mappoint에서 보이는 nkf
  for(Mappoint* mpt : neighbor_mappoints){
    for(Frame* nkf : mpt->GetKeyframes(qth) )
      neighbor_keyframes[nkf->GetId()] = nkf;
  }
  // 3. '2'의 nkf에서 보이는 mappoints
  int n = 0;
  for(auto it_nkf = neighbor_keyframes.rbegin(); it_nkf != neighbor_keyframes.rend(); it_nkf++){
    if(++n > 5) // N keyframe 이상 보이지 않은 mappoint는 제외.
      break;
    GetMappoints4Qth(it_nkf->second, qth, neighbor_mappoints);
  }
  return;
}

std::map<Qth,bool> Pipeline::FrameNeedsToBeKeyframe(Frame* frame,
                                                        RigidGroup* rig_new) const {
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  std::map<Qth, std::pair<size_t, size_t> > n_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    if(!ins)
      continue;
    Mappoint* mpt = mappoints[n];
    for(auto it_rig : ins->rig_groups_){
      n_mappoints[it_rig.first].second++;
      if(mpt)
        n_mappoints[it_rig.first].first++;
    }
  }

  std::map<Qth,bool> need_keyframes;
  const Qth qth_new = rig_new? rig_new->GetId() : 0;
  for(auto it : n_mappoints){
    const Qth& qth = it.first;
    if(qth==qth_new) // 다음 if(rig_new에서 'frame'이 qth의 kf로 추가 된다.
      continue;
    float valid_matches_ratio = ((float) it.second.first) / ((float)it.second.second);
    if(valid_matches_ratio < .5){
      need_keyframes[qth] = true;
      continue;
    }
    bool lot_flow = false;
    if(lot_flow){
      // 나~중에 valid match가 많아도 flow가 크면 미리 kf를 추가할 필요가 있다.
      need_keyframes[qth] = true;
      continue;
    }
    need_keyframes[qth] = false;
  }
  return need_keyframes;
}

void AddKeyframesAtMappoints(Frame* keyframe, RigidGroup* rig_new) {
  // ins->rig_qth 가 일하는 mappoint에 한해서 mpt->AddKeyframe(qth, frame)
  const auto& mappoints = keyframe->GetMappoints();
  const auto& instances = keyframe->GetInstances();
  for(size_t n=0; n<mappoints.size(); n++) {
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    for(auto it_ref : mpt->GetRefFrames() ){
      //if(it_rig.second == rig_new)
      //  continue;
      mpt->AddKeyframe(it_ref.first, keyframe);
      mpt->GetXq(it_ref.first);
    }
  }
  return;
}

void Pipeline::SupplyMappoints(Frame* frame,
                               RigidGroup* rig_new) {
  const auto& keypoints = frame->GetKeypoints();
  const auto& instances = frame->GetInstances();
  const auto& depths    = frame->GetMeasuredDepths();
  const auto& mappoints = frame->GetMappoints();
  /*
    현재 frame에서 관찰되는 (모든 잠재적 연결가능성있는) Qth를 모두 받기
     - 이거 나중에는 dominant group 과만 연결하는 쪽으로 가야할 순 있지만,. 일단은.
  */
  const auto& Tcqs = frame->GetTcqs();
  for(size_t n=0; n<keypoints.size(); n++){
    // depth값이 관찰되지 않는 uv only point를 구분해서 처리하면 SLAM 결과가 더 정확할텐데..
    float z = depths[n];
    if(z < 1e-5) // No depth too far
      z = 1e+5;
    const cv::KeyPoint& kpt = keypoints[n];
    Instance* ins = instances[n];
    Eigen::Vector3d Xr = z*frame->GetNormalizedPoint(n);

    if(Mappoint* mp = mappoints[n]){
      if(rig_new){
        // 기존 mp에 대해, 새로운 rig_new를 위한  Xr, Xq를 입력
        const Qth& qth = rig_new->GetId();
        mp->SetXr(qth, Xr);
        mp->AddReferenceKeyframe(qth, frame);
        const Eigen::Vector3d Xq = frame->GetTcq(qth).inverse()*Xr;
        mp->SetXq(qth, Xq);
      }
      continue;
    }
    static Ith nMappoints = 0;
    Mappoint* mp = new Mappoint(nMappoints++, ins);
    for(auto it_Tcq : Tcqs){
      mp->SetXr(it_Tcq.first, Xr);
      mp->AddReferenceKeyframe(it_Tcq.first, frame);
      const Eigen::Vector3d Xq = it_Tcq.second->inverse()*Xr;
      mp->SetXq(it_Tcq.first, Xq);
    }
    frame->SetMappoint(mp, n);
    ith2mappoints_[mp->GetId()] = mp;
  }

  if(rig_new)
    keyframes_[rig_new->GetId()][frame->GetId()] = frame;
  AddKeyframesAtMappoints(frame, rig_new); // Call after supply mappoints
  every_keyframes_.insert(frame->GetId());
  return;
}

cv::Mat VisualizeStates(const RigidGroup* rig,
                        const Frame* frame,
                        const std::map<Pth,float>& density_scores,
                        const std::map<Pth, float>& switch_states,
                        const float& switch_threshold,
                        const std::map<Jth, Frame* >& neighbor_frames,
                        const std::map<Pth,ShapePtr>& curr_shapes) {
  const Qth qth = rig->GetId();
  const cv::Mat rgb = frame->GetRgb();
  cv::Mat dst_frame; {
    cv::Mat dst_fill  = rgb.clone();
    for(auto it : curr_shapes) {
      const Pth& pth = it.first;
      bool sparse = density_scores.count(pth) ? density_scores.at(pth) < 1. : true;
      bool dynamics = switch_states.count(pth) ? switch_states.at(pth) > switch_threshold : false;

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
      /*
      TODO contour 채우는 색상.
      * [x] density score가 1. 이하인 sparse인 경우 - 파랑(안정될때까진 체크는필요)
      * [x] vertes state - dynamics인 경우 - shade
                            rigid인 경우   - 투명
      */
      //const auto& color0 = colors.at(it.first % colors.size() );
      if(!sparse && !dynamics)
        cv::drawContours(dst_fill, cnts, 0, CV_RGB(255,0,0), -1);
      //else
      //  cv::drawContours(dst_fill, cnts, 0, CV_RGB(0,0,255), -1);
    }
    cv::addWeighted(rgb, .5,
                    dst_fill, 1.,
                    1., dst_frame);
    for(auto it : curr_shapes){
      ShapePtr ptr = it.second;
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(ptr->outerior_.size() );
      for( auto pt: ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      const auto& color0 = colors.at(it.first % colors.size() );
      cv::drawContours(dst_frame, cnts, 0, color0, 2);
    }
  }

  cv::Mat dst_texts = cv::Mat::zeros( rgb.rows, 400, CV_8UC3); {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .5;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    for(auto it_kf : neighbor_frames){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      const auto t = kf->GetTcq(qth).inverse().translation();
      char text[100];
      sprintf(text,"F#%2d,  (%4.3f, %4.3f, %4.3f)", kf->GetId(), t[0], t[1], t[2] );
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

  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    Mappoint* mp = mappoints[n];
    const cv::Point2f& pt = keypoints[n].pt;
    cv::Scalar c; int thickness;
    bool b_circle = true;
    if(mp){
      thickness = -1;
      if(!ins)
        c = CV_RGB(120,120,120);
      else{
        const Pth& pth = ins->GetId();
        bool outlier = switch_states.count(pth) ?  switch_states.at(pth) < switch_threshold :  false;
        c = outlier?CV_RGB(255,0,0) : CV_RGB(0,255,0);
        const Pth& pth0 = mp->GetInstance()->GetId();
        b_circle = pth == pth0;
      }
    }
    else {
      thickness = 1;
      c = CV_RGB(120,120,120);
    }

    if(b_circle)
      cv::circle(dst_frame, pt, 4, c, thickness);
    else{
      const float hw = 6.;
      cv::Rect rec(pt.x-hw, pt.y-hw, 2*hw, 2*hw);
      cv::rectangle(dst_frame, rec, c, 2);
    }

    if(mp){
      Frame* ref = mp->GetRefFrame(qth);
      const cv::Point2f& pt0 = ref->GetKeypoint( ref->GetIndex(mp) ).pt;
      cv::line(dst_frame, pt0, pt, CV_RGB(255,255,0), 1);
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
      /*
      const auto& color0 = colors.at(pth % colors.size() );
      std::string txt = "#" + std::to_string(pth);
      if(density_scores.count(pth) ){
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << density_scores.at(pth);
        txt += " : "+ oss.str();
      }
      int baseline=0;
      auto size = cv::getTextSize(txt, fontFace, fontScale, fontThick, &baseline);
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      cv::Point dpt(.5*size.width, .5*size.height);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      cv::putText(dst_frame, txt, cp-dpt,fontFace, fontScale, color0, fontThick);
      TODO Bounding box 에서 순서대로 #Ith, d: desntiy score, s: switch score
      */
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

void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const std::map<Pth, ShapePtr>& curr_shapes,
                   const cv::Mat vis_rgb)
{
  const bool fill_bg_with_dominant = true;
  const float search_radius = 30.;
  const float switch_threshold = .3;
  float density_threshold = 1. / 20. / 20.; // NxN pixel에 한개 이상의 feature point가 존재해야 dense instance

  for(auto it_shape : curr_shapes){
    const Pth& pth = it_shape.first;
    if(pth2instances_.count(pth) )
      continue;
    pth2instances_[pth] = new Instance(pth);
  }

  static Jth nFrames = 0;
  Frame* curr_frame = new Frame(nFrames++, vis_rgb);
  curr_frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_);
  std::map<Pth,float> density_scores = curr_frame->SetInstances(curr_shapes, pth2instances_, density_threshold);
  curr_frame->SetMeasuredDepths(depth);

  RigidGroup* rig_new = SupplyRigGroup(curr_frame, qth2rig_groups_);
  std::set<Qth> curr_rigs_pred = prev_rigs_;
  if(rig_new){
    curr_rigs_pred.insert(rig_new->GetId());
    curr_frame->SetTcq(rig_new->GetId(), g2o::SE3Quat() );
  }
  if(!prev_frame_){
    const Qth& qth = rig_new->GetId();
    SupplyMappoints(curr_frame, rig_new);
    UpdateRigGroups(curr_rigs_pred, curr_frame);
    prev_frame_ = curr_frame;
    return;
  }
  // curr_frame의 instnace 중, 속한 group이 없는 ins는 기존 모든 rig_new + prev_rigs에 포함시킨다.
  UpdateRigGroups(curr_rigs_pred, curr_frame);

  std::vector<std::pair<Qth,size_t> > rig_counts;
  std::set<Qth> curr_rigs; {
    rig_counts  = CountRigPoints(curr_frame, camera_, fill_bg_with_dominant, qth2rig_groups_);
    for(auto it : rig_counts)
      curr_rigs.insert(it.first);
    //Qth qth_dominant = rig_counts.begin()->first; // Future work : rigid_tree를 만들어 부모 자식관계로 관리할때 사용
  }

  // intiial pose prediction
  const auto& Tcqs = prev_frame_->GetTcqs();
  for(auto it : Tcqs)
    curr_frame->SetTcq(it.first, *it.second);

  for(auto q_it : rig_counts){
    const Qth& qth = q_it.first;
    RigidGroup* rig  = qth2rig_groups_.at(qth);
    if(rig == rig_new)
      continue;
    Frame* latest_kf = keyframes_.at(qth).rbegin()->second;
    std::set<Mappoint*>     neighbor_mappoints;
    std::map<Jth, Frame* >  neighbor_frames;
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    std::map<int, Mappoint*> matches = ProjectionMatch(camera_, extractor_, neighbor_mappoints, curr_frame, qth, search_radius);
    for(auto it : matches){
      if(curr_frame->GetMappoint(it.first)) // 이전에 이미 matching이 된 keypoint
        continue;
      curr_frame->SetMappoint(it.second, it.first);
    }
    bool verbose = true;
    std::map<Pth,float> switch_states\
      = mapper_->ComputeLBA(camera_, qth, neighbor_mappoints, neighbor_frames, curr_frame, verbose);
    for(auto it : switch_states){
      if(it.second > switch_threshold)
        continue;
      Instance* ins = pth2instances_.at( it.first );
      rig->ExcludeInstance(ins);
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
    cv::Mat dst = VisualizeStates(rig,
                                  curr_frame,
                                  density_scores,
                                  switch_states,
                                  switch_threshold,
                                  neighbor_frames,
                                  curr_shapes);
    cv::imshow("segslam", dst);

  }

  if(true){ // Mappoint 보충이 매 프레임미다 있어야 한다는게 타당한가?
#if 0
    std::map<Qth,bool> need_keyframe = FrameNeedsToBeKeyframe(curr_frame, rig_new);
    for(auto it : need_keyframe){
      if(!it.second)
        continue;
      keyframes_[it.first][curr_frame->GetId()] = curr_frame;
    }
    SupplyMappoints(curr_frame, rig_new);
#else
    // TODO 제대로 된 keyframe 선택
    keyframes_[0][curr_frame->GetId()] = curr_frame;
    SupplyMappoints(curr_frame, rig_new);
#endif
  }
  prev_frame_ = curr_frame;
  prev_rigs_ = curr_rigs;
  return;
}

} // namespace seg
