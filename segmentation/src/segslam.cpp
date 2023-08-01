#include "segslam.h"
#include "camera.h"
#include "frame.h"
#include "optimizer.h"
#include "orb_extractor.h"
#include "seg.h"
#include "util.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>

namespace seg {

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
                                         ORB_SLAM2::ORBextractor* extractor) {
  extractor->extract(gray, cv::noArray(), keypoints_, descriptions_);
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
  for(size_t n=0; n<keypoints_.size(); n++){
    const cv::KeyPoint& kpt = keypoints_[n];
    for(auto it_shape : shapes){
      ShapePtr s_ptr = it_shape.second;
      if(s_ptr->n_missing_ > 0)
        continue;
      const auto& pt = kpt.pt;
      if(! s_ptr->HasCollision(pt.x, pt.y, true) )
        continue;
      instances_[n] = pth2instance.at(it_shape.first);
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

void RigidGroup::IncludeInstance(Instance* ins) {
  ins->rig_groups_[id_] = this;
  included_instances_[ins->GetId()] = ins;
  return;
}

#if 0
cv::Mat VisualizeMatches(const Frame* frame, const std::map<int, Mappoint*>& matches){
  cv::Mat dst = frame->GetRgb().clone();
  const auto& keypoints = frame->GetKeypoints();
  for(int n=0; n<keypoints.size(); n++){
    const auto& pt = keypoints.at(n).pt;
    if( !matches.count(n) ){
      cv::circle(dst, pt, 3, CV_RGB(255,0,0));
      continue;
    }
    Mappoint* mpt = matches.at(n);
    Frame* ref = mpt->GetRefFrame();
    const cv::Point2f& pt0 = ref->GetKeypoint( ref->GetIndex(mpt) ).pt;
    std::string msg = std::to_string( ref->GetId() );
    cv::putText(dst, msg, pt0, cv::FONT_HERSHEY_SIMPLEX, .5, CV_RGB(0,255,0) );
    cv::circle(dst, pt, 3, CV_RGB(0,0,255), -1);
    cv::line(dst, pt0, pt, CV_RGB(0,255,0), 1);
  }
  return dst;
}
#endif

std::map<int, Mappoint*> ProjectionMatch(const Camera* camera,
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
      double dist = ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
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
                   ORB_SLAM2::ORBextractor*const extractor
                  )
  : camera_(camera),
  prev_frame_(nullptr),
  extractor_(extractor),
  mapper_(new Mapper())
{
}

Pipeline::~Pipeline() {

}

cv::Mat visualize_frame(Frame* frame) {
  cv::Mat dst = frame->GetRgb().clone();
  const auto& keypoints = frame->GetKeypoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    const cv::KeyPoint& kpt = keypoints[n];
    const Instance* ins = instances[n];
    Qth qth = -1;
    cv::Scalar color = CV_RGB(255,0,0);
    if(ins){
      // pth 를 visualization할꺼냐? 아니면 qth를 visualizatino할꺼냐?
      if(!ins->rig_groups_.empty() ){
        std::set<Qth> qths;
        for(auto it_rig : ins->rig_groups_)
          qths.insert(it_rig.first);
        Qth qth = *qths.begin();
        color = colors.at( qth % colors.size() );
      }
    }
    cv::circle(dst, kpt.pt, 4, color, qth?1:-1);
  }
  return dst;
}

RigidGroup* SupplyRigGroup(Frame* frame,
                    std::map<Qth, RigidGroup*>& rig_groups) {
  size_t N = 0;
  size_t n = 0;
  std::set<Instance*> nongroup_instances;
  for(Instance* ipt : frame->GetInstances() ){
    if(!ipt)
      continue;
    N++;
    if(ipt->rig_groups_.empty()){
      n++;
      nongroup_instances.insert(ipt);
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
  for(Instance* ins : nongroup_instances)
    rig->IncludeInstance(ins);
  std::cout << "Add new group! #" << rig->GetId() << std::endl;
  return rig;
}

std::vector< std::pair<Qth, size_t> > CountRigPoints(Frame* frame,
                                                     bool fill_bg_with_dominant,
                                                     const std::map<Qth,RigidGroup*> qth2rig_groups
                                                     ){
  std::map<Qth, size_t> num_points; {
    const auto& instances = frame->GetInstances();
    for(size_t n=0; n<instances.size(); n++){
      Instance* ins = instances[n];
      if(!ins)
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
    if(++n > 3) // N keyframe 이상 보이지 않은 mappoint는 제외.
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

void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const std::map<Pth, ShapePtr>& shapes,
                   const cv::Mat vis_rgb)
{
  const bool fill_bg_with_dominant = true;
  const float search_radius = 30.;

  for(auto it_shape : shapes){
    const Pth& pth = it_shape.first;
    if(pth2instances_.count(pth) )
      continue;
    pth2instances_[pth] = new Instance(pth);
  }

  static Jth nFrames = 0;
  Frame* frame = new Frame(nFrames++, vis_rgb);
  frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_);
  frame->SetInstances(shapes, pth2instances_);
  frame->SetMeasuredDepths(depth);

  RigidGroup* rig_new = SupplyRigGroup(frame, qth2rig_groups_);
  std::vector<std::pair<Qth,size_t> > rig_counts;
  std::set<Qth> curr_rigs; {
    rig_counts  = CountRigPoints(frame, fill_bg_with_dominant, qth2rig_groups_);
    for(auto it : rig_counts)
      curr_rigs.insert(it.first);
    //Qth qth_dominant = rig_counts.begin()->first; // Future work : rigid_tree를 만들어 부모 자식관계로 관리할때 사용
  }
  if(rig_new)
    frame->SetTcq(rig_new->GetId(), g2o::SE3Quat() );

  if(!prev_frame_){
    const Qth& qth = rig_new->GetId();
    SupplyMappoints(frame, rig_new);
    prev_frame_ = frame;
    return;
  } else {
    // intiial pose prediction
    const auto& Tcqs = prev_frame_->GetTcqs();
    for(auto it : Tcqs)
      frame->SetTcq(it.first, *it.second);
  }

  for(auto q_it : rig_counts){
    const Qth& qth = q_it.first;
    RigidGroup* rig  = qth2rig_groups_.at(qth);
    if(rig == rig_new)
      continue;
    Frame* latest_kf = keyframes_.at(qth).rbegin()->second;
    std::set<Mappoint*>     neighbor_mappoints;
    std::map<Jth, Frame* >  neighbor_frames;
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    std::map<int, Mappoint*> matches = ProjectionMatch(camera_,
                                                       neighbor_mappoints, 
                                                       frame,
                                                       qth,
                                                       search_radius);
    for(auto it : matches){
      if(frame->GetMappoint(it.first)) // 이전에 이미 matching이 된 keypoint
        continue;
      frame->SetMappoint(it.second, it.first);
    }
    bool verbose = true;
    mapper_->ComputeLBA(camera_, qth, neighbor_mappoints, neighbor_frames,
                        frame, verbose);
  }

  // if(frame->GetId() > 10){
  //   std::cout << "terminate for frame" << std::endl;
  //   cv::waitKey();
  //   exit(1);
  // }

  {
    std::map<Qth,bool> need_keyframe = FrameNeedsToBeKeyframe(frame, rig_new);
    for(auto it : need_keyframe){
      if(!it.second)
        continue;
      keyframes_[it.first][frame->GetId()] = frame;
      SupplyMappoints(frame, rig_new);
    }
  }

  prev_frame_ = frame;
  return;
}

} // namespace seg
