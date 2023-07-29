#include "segslam.h"
#include "camera.h"
#include "frame.h"
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

void Frame::ExtractKeypoints(const cv::Mat gray,
                             ORB_SLAM2::ORBextractor* extractor) {
  extractor->extract(gray, cv::noArray(), keypoints_, descriptions_);
  mappoints_.resize(keypoints_.size(), nullptr);
  instances_.resize(keypoints_.size(), nullptr);
  measured_depths_.resize(keypoints_.size(), 0.f);
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

Mappoint::Mappoint(Ith id,
                   const std::map<Qth,RigidGroup*>& visible_rigs,
                   Frame* ref, 
                   float invd)
  : ref_(ref),
  id_(id)
{
  for(auto it : visible_rigs)
    keyframes_[it.first].insert(ref);
  SetInvD(invd);
}

float Mappoint::GetDepth() const {
  return 1./invd_;
}

void Mappoint::SetInvD(float invd){
  /*
  if(invd < 1e-5){
    // Ignore too smal inverse depth depth which cause NaN error.
    invd = 1e-5;
  }
  invd_ = invd;
  */
  invd_ = std::max<float>(1e-5, invd);
  return;
}

void RigidGroup::IncludeInstance(Instance* ins) {
  ins->rig_groups_.insert(this);
  included_instances_[ins->pth_] = ins;
  return;
}

Pipeline::Pipeline(const Camera* camera,
                   ORB_SLAM2::ORBextractor*const extractor
                  )
  : camera_(camera),
  prev_frame_(nullptr),
  extractor_(extractor)
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
      // TODO pth 를 visualization할꺼냐? 아니면 qth를 visualizatino할꺼냐?
      if(!ins->rig_groups_.empty() ){
        std::set<Qth> qths;
        for(auto rig : ins->rig_groups_)
          qths.insert(rig->GetId());
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
      for(RigidGroup* rig : ins->rig_groups_)
        num_points[rig->GetId()]++;
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
                     const std::map<Qth,RigidGroup*>& visible_rigs,
                     std::map<Ith, Mappoint*>& ith2mappoints
                     ) {
  const auto& keypoints = ref_frame->GetKeypoints();
  const auto& instances = ref_frame->GetInstances();
  const auto& depths    = ref_frame->GetMeasuredDepths();
  auto&       mappoints = ref_frame->GetMappoints();
  /*
    현재 frame에서 관찰되는 (모든 잠재적 연결가능성있는) Qth를 모두 받기
     - 이거 나중에는 dominant group 과만 연결하는 쪽으로 가야할 순 있지만,. 일단은.
  */
  for(size_t n=0; n<keypoints.size(); n++){
    Mappoint*& mpt = mappoints[n];
    if(mpt)
      continue;

    // depth값이 관찰되지 않는 uv only point를 구분해서 처리하면 SLAM 결과가 더 정확할텐데
    // 근데 이건 LBA g2o graph 만드는 단계에서 판정이 가능.

    float z = std::max<float>(1e-5, depths[n]);
    const cv::KeyPoint& kpt = keypoints[n];
    Instance* ins = instances[n];
    if(!ins) // TODO ins 없는 mappoint를 ground로 연결시켜도 좋은 결과가 나올텐데.
      continue;
    static Ith nMappoints = 0;
    mpt = new Mappoint(nMappoints++, visible_rigs, ref_frame, 1./z);
    ith2mappoints[mpt->GetId()] = mpt;
  }
  return;
}

void GetMappoints4Rig(Frame* frame,
                      std::map<Qth, std::set<Mappoint*> >& _mappoints){
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
    for(RigidGroup* rig : ins->rig_groups_){
      const Qth& qth = rig->GetId();
      _mappoints[qth].insert(mpt);
    }
  }
  return;
}

void GetNeighbors(Frame* frame,
                  const std::map<Qth,RigidGroup*>     &rig_groups,
                  std::map<Qth, RigidGroup* >         &visible_rigs,
                  std::map<Qth, std::set<Mappoint*> > &neighbor_mappoints,
                  std::map<Qth, std::set<Frame*> >    &neighbor_frames) {
  // 1. frame에서 보이는 mappoints
  GetMappoints4Rig(frame, neighbor_mappoints);
  // frame에서 보이는 rig로 제한.
  for(auto it : neighbor_mappoints)
    visible_rigs[it.first] = rig_groups.at(it.first);
  // 2. '1'의 mappoint에서 보이는 nkf
  for(auto it : neighbor_mappoints){
    const Qth& qth = it.first;
    for(Mappoint* mpt : it.second){
      for(Frame* nkf : mpt->GetKeyframes(qth) )
        neighbor_frames[qth].insert(nkf);
    }
  }
  // 3. '2'의 nkf에서 보이는 mappoints
  for(auto it : neighbor_frames){
    for(Frame* nkf : it.second)
      GetMappoints4Rig(frame, neighbor_mappoints);
  }
  std::set<Qth> erase_list;
  for(auto it : neighbor_mappoints){
    if(! visible_rigs.count(it.first) )
      erase_list.insert(it.first);
  }
  for(Qth qth : erase_list)
    neighbor_mappoints.erase(qth);
  return;
}

std::map<Qth, RigidGroup*> GetCurrRigidgroups(Frame* frame){
  std::map<Qth, RigidGroup*> curr_rigs;
  const std::vector<Instance*>& instances = frame->GetInstances();
  for(Instance* ins : instances){
    if(!ins)
      continue;
    for(RigidGroup* rig : ins->rig_groups_)
      curr_rigs[rig->GetId()] = rig;
  }
  return curr_rigs;
}

std::map<Qth,bool> Pipeline::NeedKeyframe(Frame* frame,
                                          RigidGroup* new_rig) const {
  std::map<Qth,bool> need_keyframes;
  if(new_rig)
    need_keyframes[new_rig->GetId()] = true;
  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  std::map<Qth, std::pair<size_t, size_t> > n_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    if(!ins)
      continue;
    Mappoint* mpt = mappoints[n];
    for(RigidGroup* rig : ins->rig_groups_){
      n_mappoints[rig->GetId()].second++;
      if(mpt)
        n_mappoints[rig->GetId()].first++;
    }
  }

  for(auto it : n_mappoints){
    const Qth& qth = it.first;
    if(need_keyframes.count(qth)){
      assert(b_keyframes.at(qth) == true);
      continue;
    }
    float valid_matches_ratio = ((float) it.second.first) / ((float)it.second.second);
    if(valid_matches_ratio < .5){
      need_keyframes[qth] = true;
      continue;
    }
    bool lot_flow = false;
    if(lot_flow){
      // TODO 나~중에 valid match가 많아도 flow가 크면 미리 kf를 추가할 필요가 있다.
      need_keyframes[qth] = true;
      continue;
    }
    need_keyframes[qth] = false;
  }
  return need_keyframes;
}

void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const std::map<Pth, ShapePtr>& shapes,
                   const cv::Mat vis_rgb)
{
  camera_->GetK();
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
  frame->ExtractKeypoints(gray, extractor_);
  frame->SetInstances(shapes, pth2instances_);
  frame->SetMeasuredDepths(depth);

  std::map<Qth, RigidGroup* >         visible_rigs;
  std::map<Qth, std::set<Mappoint*> > neighbor_mappoints;
  std::map<Qth, std::set<Frame*> >    neighbor_frames;
  GetNeighbors(frame, qth2rig_groups_, visible_rigs, neighbor_mappoints, neighbor_frames);

  if(!keyframes_.empty() ){
    /* Inprogress
    * [ ] TODO 기존 keyframe, latest keyframe Tcw를 참고해서 rprj matching 수행
     - matching 결과물의 visualization
     - matching 후 LBA (epipolar부터) 수행

     * 필요한것 - latest frame에서 각 qth의 좌표가 필요.
    


    */
    std::cout << "Exit : 2D-3D correspondence matching is done" << std::endl;
    exit(1);
  }

  // nullptr if no new rigid group
  RigidGroup* rig_new = SupplyRigGroup(frame, qth2rig_groups_);
  bool fill_bg_with_dominant = true;
  std::vector<std::pair<Qth,size_t> > rig_counts 
    = CountRigPoints(frame, fill_bg_with_dominant, qth2rig_groups_);
  Qth qth_dominant = rig_counts.begin()->first;
  //std::cout << "Qth dominant = " << qth_dominant << std::endl;
  std::map<Qth,bool> need_keyframes = NeedKeyframe(frame, rig_new);

  if(prev_frame_){
    bool is_kf = false;
    for(auto it : need_keyframes){
      const Qth& qth = it.first;
      const bool& is_kf4qth = it.second;
      if(!is_kf4qth)
        continue;
      if(rig_new){
        assert(Tcqs_[qth].empty());
        Tcqs_[qth][frame->GetId()] = g2o::SE3Quat(); // Initial coordinate
      }
      keyframes_[qth][prev_frame_->GetId()] = prev_frame_;
      is_kf = true;
    }
    if(is_kf)
      SupplyMappoints(prev_frame_,visible_rigs, ith2mappoints_);
    else
      delete prev_frame_;
  }
  prev_frame_ = frame;

  {
    cv::Mat dst =  visualize_frame(frame);
    cv::imshow("slam frame", dst);
  }
  return;
}

} // namespace seg
