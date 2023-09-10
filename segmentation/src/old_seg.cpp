#include <opencv2/core/types.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
// #include <opencv2/cudafilters.hpp>

#include "../include/seg.h"
#include "../include/util.h"

#if 0
void Pipeline::Put(const cv::Mat gray,
                   const cv::Mat depth,
                   const std::map<Pth, ShapePtr>& shapes,
                   const cv::Mat vis_rgb)
{
  const float search_radius = 30.;
  /*
  *[x] 3) 기존 mappoint 와 matching,
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
#if 1
  // Seg SLAM
  RigidGroup* rig_new = SupplyRigGroup(frame, qth2rig_groups_);
#else
  // 그냥 SLAM
  RigidGroup* rig_new = frame->GetId() == 0 ? SupplyRigGroup(frame, qth2rig_groups_) : nullptr;
#endif

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
    //kf_Tcqs_[qth][frame->GetId()]   = current_Tcq.at(qth);
    frame->SetTcq(qth, curret.at(qth));
    qth2rig_groups_.at(qth)->SetLatestKeyframe(frame);
    SupplyMappoints(frame, ith2mappoints_);
    AddKeyframesAtMappoints(frame); // Call after supply mappoints
    keyframes_[qth][frame->GetId()] = frame;
    every_keyframes_.insert(frame->GetId());

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
      if(rig != rig_new)
        throw;
      continue;
    }

    std::cout << "F#" << frame->GetId() << " Tqc 0.t() = " << Tcq.inverse().translation().transpose() << std::endl;
    std::set<Mappoint*>     neighbor_mappoints;
    std::map<Jth, Frame* >  neighbor_frames;
    GetNeighbors(latest_kf, qth, neighbor_mappoints, neighbor_frames);
    std::map<int, Mappoint*> matches = ProjectionMatch(camera_,
                                                       neighbor_mappoints, 
                                                       Tcq,
                                                       kf_Tcqs_.at(qth),
                                                       frame,
                                                       search_radius);
    for(auto it : matches){
      if(frame->GetMappoint(it.first)) // 이전에 이미 matching이 된 keypoint
        continue;
      frame->SetMappoint(it.second, it.first);
    }
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

  /* 
    * TODO prev_frame이 kf 로 승격되야하는가?
      * [x] frame의 maapoints여부로 판정.
      * [x] frame에 새로운 qth가 추가되었는가 여부.
  */
  if( !every_keyframes_.count(prev_frame_->GetId()) ) {
    bool convert_prev2kf = false;
    // frame의 mappoint 부족으로 prev_frame을 kf로 변환해야함.
    std::map<Qth,bool> prev_need_to_be_keyframe = PrevFrameNeedsToBeKeyframe(frame); // 판정은 current로, kf 승경은 prev로 
    for(auto it : prev_need_to_be_keyframe){
      const Qth& qth = it.first;
      if(rig_new && rig_new->GetId() == qth) // 다음 if(rig_new에서 'frame'이 qth의 kf로 추가 된다.
        continue;
      const bool& is_kf4qth = it.second;
      if(!is_kf4qth)
        continue;
      kf_Tcqs_[qth][prev_frame_->GetId()]   = prev_Tcqs_.at(qth);
      qth2rig_groups_.at(qth)->SetLatestKeyframe(prev_frame_);
      keyframes_[qth][prev_frame_->GetId()] = prev_frame_;
      convert_prev2kf = true;
    }
    if(convert_prev2kf){
      SupplyMappoints(prev_frame_, ith2mappoints_);
      AddKeyframesAtMappoints(prev_frame_); // Call after supply mappoints
      every_keyframes_.insert(prev_frame_->GetId());
    }
    else
      delete prev_frame_;
  }
  if(rig_new){
    const Qth& qth = rig_new->GetId();
    for(auto it : current_Tcq)
      kf_Tcqs_[it.first][frame->GetId()] = it.second;
    qth2rig_groups_.at(qth)->SetLatestKeyframe(frame);
    keyframes_[qth][frame->GetId()] = frame;
    std::cout << "Test " << kf_Tcqs_.at(qth).at(frame->GetId()) << std::endl;
    SupplyMappoints(frame, ith2mappoints_);
    AddKeyframesAtMappoints(frame); // Call after supply mappoints
    every_keyframes_.insert(frame->GetId());
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


#endif
