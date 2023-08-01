#include "segslam.h"
#include "camera.h"
#include "frame.h"
#include "optimizer.h"
#include "orb_extractor.h"
#include "seg.h"
#include "util.h"
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>

/*
std::vector<cv::Scalar> colors = {
  CV_RGB(0,180,0),
  CV_RGB(0,100,0),
  CV_RGB(255,0,255),
  CV_RGB(100,0,255),
  CV_RGB(100,0,100),
  CV_RGB(0,0,180),
  CV_RGB(0,0,100),
  CV_RGB(255,255,0),
  CV_RGB(100,255,0),
  CV_RGB(100,100,0),
  CV_RGB(100,0,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};
*/

namespace seg {

RigidGroup::RigidGroup(Qth qth)
: id_(qth),
  bg_instance_(new Instance(-1)),
  latest_kf_(nullptr)
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

Mappoint::Mappoint(Ith id,
                   const std::set<Qth>& ref_rigs, 
                   Frame* ref)
  : ref_(ref),
  id_(id)
{
}
/*
float Mappoint::GetDepth() const {
  return 1./invd_;
}
void Mappoint::SetInvD(float invd){
  invd_ = std::max<float>(1e-5, invd);
  return;
}
*/
void Mappoint::SetXr(const Eigen::Vector3d& Xr) {
  Xr_ = Xr;
  return;
}

const Eigen::Vector3d& Mappoint::GetXr() const {
  return Xr_;
}


cv::Mat Mappoint::GetDescription() const {
  int kpt_idx = ref_->GetIndex(this);
  return ref_->GetDescription(kpt_idx);
}

void RigidGroup::IncludeInstance(Instance* ins) {
  ins->rig_groups_[id_] = this;
  included_instances_[ins->pth_] = ins;
  return;
}

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
    cv::circle(dst, pt, 3, CV_RGB(0,0,255), -1);
    cv::line(dst, pt0, pt, CV_RGB(0,255,0), 1);
  }
  return dst;
}

std::map<int, Mappoint*> ProjectionMatch(const Camera* camera,
                                         const std::set<Mappoint*>& mappoints,
                                         const g2o::SE3Quat& predicted_Tcq,
                                         const EigenMap<Jth, g2o::SE3Quat> & kf_Tcqs,
                                         const Frame* curr_frame,
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

    const Eigen::Vector3d Xr = mp->GetXr();
    const g2o::SE3Quat& Trq = kf_Tcqs.at(mp->GetRefFrame()->GetId());
    const Eigen::Vector3d Xc = predicted_Tcq * Trq.inverse() * Xr;
    if(Xc.z() < 0.)
      continue;
    Eigen::Vector2d uv = camera->Project(Xc);
    // 이거 체크해야할까? 밖으로 넘어가도, radius만 만족하면..
    //if(!curr_frame->IsInFrame(uv)) 
    //  continue;
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
  if(ng_ratio < .2)
    return nullptr;
  static Qth nRiggroups = 0;
  RigidGroup* rig = new RigidGroup(nRiggroups++);
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

void SupplyMappoints(Frame *ref_frame,
                     const std::set<Qth>& ref_rigs,
                     const EigenMap<Qth, EigenMap<Jth, g2o::SE3Quat> >& kf_Tcqs,   
                     std::map<Ith, Mappoint*>& ith2mappoints
                     ) {
  const auto& keypoints = ref_frame->GetKeypoints();
  const auto& instances = ref_frame->GetInstances();
  const auto& depths    = ref_frame->GetMeasuredDepths();
  const auto& mappoints = ref_frame->GetMappoints();
  /*
    현재 frame에서 관찰되는 (모든 잠재적 연결가능성있는) Qth를 모두 받기
     - 이거 나중에는 dominant group 과만 연결하는 쪽으로 가야할 순 있지만,. 일단은.
  */
  for(size_t n=0; n<keypoints.size(); n++){
    if(mappoints[n])
      continue;
    // depth값이 관찰되지 않는 uv only point를 구분해서 처리하면 SLAM 결과가 더 정확할텐데..
    float z = depths[n];
    if(z < 1e-5) // No depth too far
      z = 1e+3;
    const cv::KeyPoint& kpt = keypoints[n];
    Instance* ins = instances[n];
    assert(ins);
    static Ith nMappoints = 0;
    auto ptr = new Mappoint(nMappoints++, ref_rigs, ref_frame);
    Eigen::Vector3d Xr = z*ref_frame->GetNormalizedPoint(n);
    ptr->SetXr(Xr);
    for(Qth qth : ref_rigs){
      const g2o::SE3Quat& Trq = kf_Tcqs.at(qth).at(ref_frame->GetId());
      const Eigen::Vector3d Xq = Trq.inverse()*Xr;
      ptr->SetXq(qth, Xq);
    }
    ref_frame->SetMappoint(ptr, n);
    ith2mappoints[ptr->GetId()] = ptr;
  }
  return;
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
  for(auto it_nkf : neighbor_keyframes){
    GetMappoints4Qth(it_nkf.second, qth, neighbor_mappoints);
  }
  return;
}

std::map<Qth,bool> Pipeline::NeedKeyframe(Frame* frame) const {
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
  for(auto it : n_mappoints){
    const Qth& qth = it.first;
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

void AddKeyframesAtMappoints(Frame* keyframe) {
  // ins->rig_qth 가 일하는 mappoint에 한해서 mpt->AddKeyframe(qth, frame)
  const auto& mappoints = keyframe->GetMappoints();
  const auto& instances = keyframe->GetInstances();
  for(size_t n=0; n<mappoints.size(); n++) {
    Mappoint* mpt = mappoints[n];
    if(!mpt)
      continue;
    Instance* ins = instances[n];
    if(!ins)
      continue;
    for(auto it_rig : ins->rig_groups_)
      mpt->AddKeyframe(it_rig.first, keyframe);
  }
  return;
}

void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const std::map<Pth, ShapePtr>& shapes,
                   const cv::Mat vis_rgb)
{
  const float search_radius = 30.;
  /*
  *[ ] 3) 기존 mappoint 와 matching,
    * Matching은 qth고려 없이 가능한가? rprj 참고하려면 필요는 한데..
  *[ ] 4) qth group 별로 LBA
  */
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

  // nullptr if no new rigid group
  RigidGroup* rig_new = nullptr;
  if(frame->GetId() == 0) // TODO 제거 
    rig_new = SupplyRigGroup(frame, qth2rig_groups_);

  bool fill_bg_with_dominant = true;
  std::vector<std::pair<Qth,size_t> > rig_counts 
    = CountRigPoints(frame, fill_bg_with_dominant, qth2rig_groups_);
  std::set<Qth> curr_rigs;
  for(auto it : rig_counts)
    curr_rigs.insert(it.first);
  //Qth qth_dominant = rig_counts.begin()->first; // Future work : rigid_tree를 만들어 부모 자식관계로 관리할때 사용

  EigenMap<Qth, g2o::SE3Quat> current_Tcq;
  if(rig_new)
    current_Tcq[rig_new->GetId()] = g2o::SE3Quat();

  if(frame->GetId() == 0){
    const Qth& qth = rig_new->GetId();
    kf_Tcqs_[qth][frame->GetId()]   = current_Tcq.at(qth);
    qth2rig_groups_.at(qth)->SetLatestKeyframe(frame);
    SupplyMappoints(frame, curr_rigs, kf_Tcqs_, ith2mappoints_);
    AddKeyframesAtMappoints(frame); // Call after supply mappoints
    keyframes_[qth][frame->GetId()] = frame;

    prev_frame_ = frame;
    for(auto it : current_Tcq)
      prev_Tcqs_[it.first] = it.second;
    prev_rigs_ = curr_rigs;
    return;
  }

  for(auto q_it : rig_counts){
    const Qth& qth = q_it.first;
    if(prev_Tcqs_.count(qth))
      current_Tcq[qth] = prev_Tcqs_.at(qth);
    else
      assert( qth == rig_new->GetId() );
  }

  for(auto q_it : rig_counts){
    const Qth& qth = q_it.first;
    g2o::SE3Quat Tcq = current_Tcq.at(qth);
    RigidGroup* rig  = qth2rig_groups_.at(qth);
    Frame* latest_kf = rig->GetLatestKeyframe();
    if(!latest_kf) {
      assert(rig == rig_new);
      continue;
    }

    std::cout << "F#" << frame->GetId() << " Tqc 0.t() = " << Tcq.inverse().translation().transpose() << std::endl;
    std::set<Mappoint*>     neighbor_mappoints;
    std::map<Jth, Frame* >  neighbor_frames;
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    std::map<int, Mappoint*> matches = ProjectionMatch(camera_, neighbor_mappoints, 
                                                       Tcq, kf_Tcqs_.at(qth), frame, search_radius);
    for(auto it : matches)
      frame->SetMappoint(it.second, it.first);
    //std::cout << "l_kf#" << latest_kf->GetId() << ", n(nkf)= " << neighbor_frames.size() << ", n(mpt)" << neighbor_mappoints.size() << std::endl;
    if(qth == 0){
      printf("curr F# %d neighbor keyframes = {", frame->GetId());
      for(auto it_kf : neighbor_frames)
        printf("#%d, ", it_kf.first);
      printf("}\n");
      cv::Mat dst = VisualizeMatches(frame, matches);
      cv::imshow("matches", dst);
    }
    bool verbose = qth==0;
    mapper_->ComputeLBA(camera_, qth, neighbor_mappoints, neighbor_frames,
                        frame, kf_Tcqs_[qth], Tcq, verbose);
    current_Tcq[qth] = Tcq;
  }
  /*
  cv::Mat dst = VisualizeMatches(frame, matches);
  cv::imshow("matches", dst);
    cv::waitKey(0);
    */
    //std::cout << "curr f#" <<  frame->GetId() << " q#" << qth << ", Tqc = " << Tcq.inverse().translation().transpose() << std::endl;
  //std::cout << "Qth dominant = " << qth_dominant << std::endl;
  if(frame->GetId() > 1){
    std::map<Qth,bool> need_keyframes = NeedKeyframe(frame); // 판정은 current로, kf 승경은 prev로 
    bool is_kf = false;
    for(auto it : need_keyframes){
      const Qth& qth = it.first;
      const bool& is_kf4qth = it.second;
      if(!is_kf4qth)
        continue;
      kf_Tcqs_[qth][prev_frame_->GetId()]   = prev_Tcqs_.at(qth);
      qth2rig_groups_.at(qth)->SetLatestKeyframe(prev_frame_);
      keyframes_[qth][prev_frame_->GetId()] = prev_frame_;
      is_kf = true;
    }
    if(is_kf){
      SupplyMappoints(prev_frame_, prev_rigs_, kf_Tcqs_, ith2mappoints_);
      AddKeyframesAtMappoints(prev_frame_); // Call after supply mappoints
    }
    else
      delete prev_frame_;
  }
  prev_frame_ = frame;
  for(auto it : current_Tcq)
    prev_Tcqs_[it.first] = it.second;
  prev_rigs_ = curr_rigs;

  {
    cv::Mat dst =  visualize_frame(frame);
    cv::imshow("slam frame", dst);
  }
  return;
}

} // namespace seg
