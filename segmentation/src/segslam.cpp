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

/*
namespace OLD_SEG {
  
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

Frame* Pipeline::Put(const cv::Mat gray,
                     const cv::Mat depth,
                     const std::vector<cv::Mat>& flow,
                     const cv::Mat synced_marker,
                     const std::map<int,size_t>& marker_areas,
                     const cv::Mat gradx,
                     const cv::Mat grady,
                     const cv::Mat valid_grad,
                     const cv::Mat vis_rgb
                    )
{
  const bool fill_bg_with_dominant = true;
  const float search_radius = 30.;
  switch_threshold_ = .3;
  float density_threshold = 1. / 70. / 70.; // NxN pixel에 한개 이상의 mappoint가 존재해야 dense instance
  const bool verbose_flowmatch = false;

  vinfo_switch_states_.clear();
  vinfo_neighbor_frames_.clear();
  vinfo_neighbor_mappoints_.clear();
  vinfo_synced_marker_ = synced_marker;

  for(const auto& it : marker_areas)
    if(!pth2instances_.count(it.first) )
      pth2instances_[it.first] = new Instance(it.first);

  cv::Mat outline_mask = synced_marker < 1;
  static Jth nFrames = 0;
  Frame* curr_frame = new Frame(nFrames++, vis_rgb);
  curr_frame->ExtractAndNormalizeKeypoints(gray, camera_, extractor_, outline_mask);
  curr_frame->SetInstances(synced_marker, pth2instances_);
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
      return curr_frame;
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
    = FlowMatch(camera_, extractor_, flow, prev_frame_, verbose_flowmatch, curr_frame);
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

  std::map<Pth,float>& density_scores = vinfo_density_socres_;
  for(const auto& it : marker_areas){
    const Pth& pth = it.first;
    const size_t& area = it.second;
    float npoints = ins2mappoints.count(pth) ? ins2mappoints.at(pth) : 0.;
    float dense = npoints > 0 ? float(npoints) / float(area) : 0.; // 최소 점의 개수도 고려.
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
    std::set<Mappoint*>&     neighbor_mappoints = vinfo_neighbor_mappoints_[qth];
    std::map<Jth, Frame* >&  neighbor_frames = vinfo_neighbor_frames_[qth];
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    if(neighbor_mappoints.empty()) {
      std::cerr << "qth = " << qth << ", failure to get mappoints" << std::endl;
      throw -1;
    }

#if 0
    // ProjectionMatch는 flow로부터 motion update와 pth(k_0)!= pth(k) 인 경우에 예외처리가 팔요해보인다.
    std::map<int, std::pair<Mappoint*, double> > proj_matches = ProjectionMatch(camera_, extractor_, neighbor_mappoints, curr_frame, qth, search_radius);
    SetMatches(flow_matches, proj_matches, curr_frame);
#endif
    std::set<Pth> fixed_instances;
    if(qth == dominant_qth)
      for(auto it_density : density_scores)
        if(it_density.second < 1.)
          fixed_instances.insert(it_density.first);

    bool vis_verbose = false;
    std::map<Pth,float>& switch_states = vinfo_switch_states_[qth];
    switch_states\
      = mapper_->ComputeLBA(camera_,qth, neighbor_mappoints, neighbor_frames,
                            curr_frame, prev_frame_, fixed_instances, gradx, grady, valid_grad, vis_verbose);

    // LocalBA로 structure, motion 모두 추정한 다음에, rpjr error가 동일 instance내 다른 feature에 비해 유난히 큰 matching을 제거.
    FilterOutlierMatches(qth, curr_frame);

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
        printf("Exclude Q#%d P#%d", qth, pth);
        instances4next_rig.insert(pth);
      }
      else
        switchoff_instances.insert( pth); // 현재 프레임에서 switchoff가 발생하지 않은 pthqth를 삭제하기위해.
    }

    size_t n_mappoints4next_rig = 0;
    for(Pth pth : instances4next_rig)
      if(ins2mappoints.count(pth))// mp 없는 instance도 있으니까.
         n_mappoints4next_rig += ins2mappoints.at(pth);

    Qth qth_next = -1;
    if(segmented_instances.empty()){
#if 1
      if(n_mappoints4next_rig > 10) {
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

    if(true){
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
    else{
      n_consecutiv_switchoff.clear();
    }

   if(++nq > 3) // N 번만 수행.
      break;
  } // if(segmented_instances.empty())

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
  if(prev_frame_ && ! prev_frame_->IsKeyframe() )
    delete prev_frame_;
  prev_frame_ = curr_frame;
  prev_dominant_qth_ = dominant_qth;
  return curr_frame;
}

} // namespace OLD_SEG
*/
