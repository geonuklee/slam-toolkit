/*
Copyright (c) 2020 Geonuk Lee

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#include "pipeline.h"
#include "matcher.h"
#include "frame.h"
#include "mappoint.h"
#include "camera.h"
#include "orb_extractor.h"
#include "posetracker.h"
#include "localmapper.h"
#include "pipeline_map.h"
#include "method.h"

#include "ORBVocabulary.h"

//#include "memento.h"
#include "loopcloser.h"
#include "loopdetector.h"

//#include "config.h"
#include <unistd.h> //sleep
#include <chrono>

ORB_SLAM2::ORBextractor* CreateExtractor(){
  int nfeatures = 2000;
  float scale_factor = 1.2;
  int nlevels = 8;
  int initial_fast_th = 20;
  int min_fast_th = 7;

  ORB_SLAM2::ORBextractor* extractor = new ORB_SLAM2::ORBextractor(nfeatures,
                                                                   scale_factor,
                                                                   nlevels,
                                                                   initial_fast_th,
                                                                   min_fast_th);
  return extractor;
}

ORB_SLAM2::ORBVocabulary* CreateVocabulary(){
  std::string fn = GetPackageDir()+"/thirdparty/ORBvoc.txt";
  ORB_SLAM2::ORBVocabulary* ptr = new ORB_SLAM2::ORBVocabulary();
  if(!ptr->loadFromTextFile(fn) ){
    throw std::invalid_argument("Load ORBVoc failure");
  }
  return ptr;
}

Pipeline::~Pipeline(){
  if(extractor_)
    delete extractor_;
  if(pose_tracker_)
    delete pose_tracker_;
  if(local_mapper_)
    delete local_mapper_;
  if(map_)
    delete map_;
  if(latest_keyframe_)
    delete latest_keyframe_;
}

Pipeline::Pipeline(const Camera* camera)
  : camera_(camera),
  extractor_(CreateExtractor()),
  map_(new PipelineMap(CreateVocabulary() )),
  latest_keyframe_(nullptr),
  exit_flag_(false)
{
  const auto& inv_sigma2 = extractor_->GetInverseScaleSigmaSquares();
  pose_tracker_ = new IndirectPoseTracker(inv_sigma2);
  local_mapper_ = new StandardLocalMapper(std::make_shared<IndirectStereoMethod>(inv_sigma2) );
  loop_detetor_ = new LoopDetector(map_);
  loop_closer_  = new LoopCloser(map_, std::make_shared<IndirectPoseTracker>(inv_sigma2) );

  mapping_thread_ = new std::thread(&Pipeline::mapping, this);
}

void Pipeline::mapping() {
  while(true){
    Frame* frame = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_new_keyframes_);
      if(exit_flag_)
        break;
      if(!new_keyframes_.empty()){
        frame = new_keyframes_.front();
        new_keyframes_.pop();
      }
    }
    if(!frame){
      sleep(1);
      continue;
    }

    if(loop_detetor_){
      if(loop_detetor_->DetectLoop(frame) ){
        auto loop_candidates = loop_detetor_->GetLoopCandidates();
        Frame* loop_frame;
        g2o::SE3Quat T_curr_loop;
        bool enough_match
          = loop_closer_->GetRelativePose(frame, loop_candidates, loop_frame, T_curr_loop);
        if(enough_match){
          std::set<Frame*> updated_keyframes
            =loop_closer_->CloseLoop(frame, loop_frame, T_curr_loop);
          for(Frame* kf : updated_keyframes)
            for(auto viewer : viewers_)
              viewer->UpdateMappoints(kf);
        }
        else{
          std::cout << "For loop candidate F# = {";
          for(Frame* kf : loop_candidates)
            std::cout << kf->GetIndex() << ", ";
          std::cout << "}, not enough match to frame F#" << frame->GetIndex() << std::endl;
        }
      }
    }
    const int n_iter = 10;
    local_mapper_->Optimize(map_, frame, n_iter);
  }
  return;
}

g2o::SE3Quat Pipeline::Track(cv::Mat im_left, cv::Mat im_right){
	auto start = std::chrono::steady_clock::now();
  StereoFrame* frame = new StereoFrame(im_left, im_right, Frame::frame_n_++,
                                 static_cast<const StereoCamera*>(camera_), extractor_);
  std::set<Frame*> neighbor_keyframes;

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Lock loop closing during pose estimation
  map_->Lock();
  if(latest_keyframe_){
    const bool already_locked = true;
    g2o::SE3Quat dT;
    StereoFrame* f0 = nullptr;
    StereoFrame* f1 = nullptr;
    g2o::SE3Quat Tcw_previous;
    if(map_->GetFramesNumber(already_locked) > 1){
      f1 = static_cast<StereoFrame*>( map_->GetFrame(frame->GetIndex()-1, already_locked) );
      f0 = static_cast<StereoFrame*>( map_->GetFrame(frame->GetIndex()-2, already_locked) );
      const g2o::SE3Quat Tc0w = f0->GetTcw();
      const g2o::SE3Quat Tc1w = f1->GetTcw();
      dT = Tc1w * Tc0w.inverse();
      Tcw_previous = Tc1w;
    }
    g2o::SE3Quat predicted_Tcw = dT * Tcw_previous;
    std::set<Mappoint*> neighbor_mappoints;
    latest_keyframe_->GetNeighbors(neighbor_keyframes, neighbor_mappoints);
    {
      // Fuse 기능.
      // TODO 스케일 고려하는 matching 필요
      auto latests = map_->GetLatestFrames(10, already_locked);
      for(Frame* nkf : latests){
        std::set<Mappoint*> mps = nkf->GetMappoints();
        neighbor_mappoints.insert(mps.begin(), mps.end() );
      }
    }
    g2o::SE3Quat Tcw;
    const int n_iter = 10;
    pose_tracker_->Track(neighbor_mappoints, (void*)&predicted_Tcw, (void*)&Tcw, n_iter, frame);
    frame->SetTcw(Tcw);
  }
  else{
    g2o::SE3Quat Tcw0;
    frame->SetTcw(Tcw0);
  }
  bool iskeyframe = IsKeyframe(frame);
  if(iskeyframe)
    frame->SetKeyframe(map_->GetVocaBulary(), Frame::keyframe_n_++);
  {
    const bool already_locked = true;
    map_->AddFrame(frame, already_locked);
  }
  map_->UnLock();
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


  if(iskeyframe){
    AddMappoints(frame, neighbor_keyframes);
    mutex_new_keyframes_.lock();
    new_keyframes_.push(frame);
    mutex_new_keyframes_.unlock();
    latest_keyframe_ = frame;
  }

  // 불필요한 frame 삭제
  map_->CullingOldFrames(5);

	auto end = std::chrono::steady_clock::now();
  FrameInfo info;
  info.elapsed_ms_
    = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  for(auto viewer : viewers_)
    viewer->OnFrame(frame, info);

  if(iskeyframe){
    for(auto viewer : viewers_){
      viewer->OnSetKeyframe(frame);
      viewer->UpdateMappoints(frame);
    }
  }

  return frame->GetTcw();
}

void Pipeline::AddViewer(PipelineViewer* viewer) {
  viewers_.push_back(viewer);
}

void Pipeline::Save() const {
  std::lock_guard<std::mutex> lock(mutex_new_keyframes_);
#if 0
  MementoPipeline memento(this);
  std::ofstream fo;
  fo.open("memento.bin", std::ios::binary);
  memento.Write(fo);
  fo.flush();
  fo.close();
#endif
}

void Pipeline::AddMappoints(Frame* frame,
                            const std::set<Frame*>& neighbor_keyframes
                            ) const {
  StereoFrame* stereo_frame = static_cast<StereoFrame*>(frame);
  stereo_frame->ExtractRightKeypoints();
  StereoMatch(stereo_frame);
  std::set<Mappoint*> new_mappoints = SupplyMappoints(frame);

  // Try match to pevious keyframes.
  double search_radius = 50.;
  for(Frame* nkf : neighbor_keyframes){
    std::map<int, Mappoint*> keypoint_matches
      = ProjectionMatch(new_mappoints, nkf->GetTcw(), nkf, search_radius);
    for(auto it_match : keypoint_matches){
      int kpt_idx = it_match.first;
      Mappoint* new_mp = it_match.second;
      nkf->SetMappoitIfEmpty(new_mp, kpt_idx);
    }
  }
}

bool DoFrameNeedsNewMappoints(const Frame* frame) {
  const int nw = 4;
  const int nh = 1;
  const int min_num_mappoitns_in_grid = 5;
  const int min_num_mappoitns = 20;
  const Camera* camera = frame->GetCamera();
  int width = camera->GetWidth() / nw;
  int height = camera->GetHeight() / nw;
  Eigen::MatrixXi number_of_mappoints(nw,nh);
  number_of_mappoints.setZero();
  std::set<Mappoint*> mappoints = frame->GetMappoints();
  if(mappoints.size() < min_num_mappoitns){
    return true;
  }
  const auto& keypoints = frame->GetKeypoints();
  auto vec_mappoitns = frame->GetVecMappoints(); // IsKeyframe
  for(size_t i = 0; i < keypoints.size(); i++){
    Mappoint* mp = vec_mappoitns.at(i);
    if(!mp)
      continue;
    const cv::KeyPoint& kpt = keypoints.at(i);
    int ix = (int) kpt.pt.x / width;
    int iy = (int) kpt.pt.y / height;
    ix = std::max(0, ix);
    iy = std::max(0, iy);
    ix = std::min<int>(number_of_mappoints.rows()-1, ix);
    iy = std::min<int>(number_of_mappoints.cols()-1, iy);
    number_of_mappoints(ix,iy)++;
  }
  for(int r = 0; r < number_of_mappoints.rows(); r++){
    for(int c = 0; c < number_of_mappoints.cols(); c++){
      if(number_of_mappoints(r,c) < min_num_mappoitns_in_grid)
        return true;
    }
  }
  return false;
}

bool Pipeline::IsKeyframe(Frame* frame) const {
  if(!latest_keyframe_)
    return true;
  return DoFrameNeedsNewMappoints(frame);
}

