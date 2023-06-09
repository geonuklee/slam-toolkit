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

#include "loopcloser.h"
#include "frame.h"
#include "mappoint.h"
#include "matcher.h"
#include "optimizer.h"
#include "pipeline_map.h"
#include "posetracker.h"

#include <queue>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

LoopCloser::LoopCloser(PipelineMap* map,
                       std::shared_ptr<IndirectPoseTracker> pose_tracker)
: map_(map),
  pose_tracker_(pose_tracker)
{

}

bool LoopCloser::GetRelativePose(const Frame* curr_frame,
                                 const std::vector<Frame*>& loop_candidates,
                                 Frame*& loop_frame,
                                 g2o::SE3Quat& T_curr_loop
                                 ) const {
  loop_frame = nullptr;
  std::map<Mappoint*,int> best_keypoint_matches;

  double search_radius = 50.; // search radius
  const double max_reprojection_error = 5.;
  g2o::SE3Quat best_T_curr_w;
  const int n_iter =10;
  ReprojectionFilter filter(10);

  for(size_t i = 0; i < 2; i++ ){
    for(Frame* candidate : loop_candidates){
      const std::set<Mappoint*> loop_mappoints = candidate->GetMappoints();
      //std::map<int, Mappoint*> keypoint_matches
      //  = ProjectionMatch(loop_mappoints, candidate->GetTcw(), curr_frame, search_radius);
      g2o::SE3Quat prediction = candidate->GetTcw();
      g2o::SE3Quat T_curr_w;
      pose_tracker_->BeforeEstimation(loop_mappoints,
                                      (void*)&prediction,
                                      curr_frame);
      pose_tracker_->EstimatePose(loop_mappoints, (void*)&prediction, (void*)&T_curr_w,
                                  n_iter, curr_frame);
      auto keypoint_matches = pose_tracker_->GetMatches();
      auto outliers = filter.GetOutlier(curr_frame, (void*) &T_curr_w);
      for(auto it : outliers)
        keypoint_matches.erase(it.second);

      if(keypoint_matches.size() > best_keypoint_matches.size() ){
        loop_frame = candidate;
        best_keypoint_matches = keypoint_matches;
        best_T_curr_w = T_curr_w;
      }
    } // for candidate

    if(best_keypoint_matches.size() > 8)
      break;
    else
      search_radius *= 2;
  }

  if(best_keypoint_matches.size() <= 8)
    return false;

  T_curr_loop = best_T_curr_w * (loop_frame->GetTcw().inverse() );
  return true;
}



std::set<Frame*> LoopCloser::CloseLoop(Frame* curr_frame,
                                       Frame* loop_frame,
                                       const g2o::SE3Quat& T_curr_loop) {
  // TODO 방향이 정확하고 (stereo를 써도, 원경에서) scale이 부정확한 vision 성질을 고려, SE3->Sim3로 변경해야함.
  // Loop closing
  map_->Lock(); // Lock map to stop adding new frame during loop closing, for consistency.
  bool already_locked = true;
  const std::map<int, Frame*> frames = map_->GetFrames(already_locked);

  Eigen::Matrix<double,6,6> wv;
  wv.setIdentity();
  wv(0,0) = wv(1,1) = wv(2,2) = 100.;
  wv(5,5) = 0.01;

  g2o::SparseOptimizer optimizer;
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver
    = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg* solver
    =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
  solver->setUserLambdaInit(1e-16);
  optimizer.setAlgorithm(solver);

  EigenMap<Frame*, g2o::SE3Quat> mTc1curr; // Relative pose of non keyframe frame
  std::map<Frame*, g2o::VertexSE3Expmap*> v_poses;
  int n_vertex = 0;
  Frame* previous_frame = nullptr;
  Frame* oldest = curr_frame;
  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    if(!frame->IsKeyframe()){
      mTc1curr[frame]
        = frame->GetTcw() * curr_frame->GetTcw().inverse();
      continue;
    }
    auto v_pose = new g2o::VertexSE3Expmap();
    v_pose->setId(n_vertex++);
    optimizer.addVertex(v_pose);
    v_poses[frame] = v_pose;
    v_pose->setEstimate(frame->GetTcw());
    if(frame->GetIndex() < oldest->GetIndex())
      oldest = frame;
    if(previous_frame){
      auto v_pose0 =  v_poses.at(previous_frame);
      auto edge = new EdgeSE3();
      //auto edge = new g2o::EdgeSE3Expmap();
      edge->setVertex(0, v_pose0);
      edge->setVertex(1, v_pose);
      g2o::SE3Quat z = v_pose->estimate() * v_pose0->estimate().inverse();
      edge->setMeasurement(z);
      edge->setInformation(wv);
      optimizer.addEdge(edge);
    }
    previous_frame = frame;
  }
  v_poses.at(oldest)->setFixed(true);

  std::vector<std::pair<Frame*,Frame*> > all_pairs;
  EigenVector<g2o::SE3Quat> realtive_poses;
  all_pairs.reserve(closed_loops_.size() + 1);
  realtive_poses.reserve(all_pairs.size());
  for(auto it : closed_loops_){
    Frame* closed_curr_frame = it.first.first;
    Frame* closed_loop_frame = it.first.second;
    all_pairs.push_back(std::make_pair(closed_curr_frame, closed_loop_frame));
    realtive_poses.push_back(it.second);
  }
  all_pairs.push_back(std::make_pair(curr_frame,loop_frame));
  realtive_poses.push_back(T_curr_loop);

  for(size_t i = 0; i < all_pairs.size(); i++){
    Frame* curr_f_each = all_pairs.at(i).first;
    Frame* loop_f_each = all_pairs.at(i).second;
    const g2o::SE3Quat& T_curr_loop_each = realtive_poses.at(i);
    auto v_pose0 = v_poses.at(loop_f_each);
    auto v_pose = v_poses.at(curr_f_each);
    auto edge = new EdgeSE3();
    edge->setVertex(0, v_pose0);
    edge->setVertex(1, v_pose);
    edge->setMeasurement(T_curr_loop_each);
    edge->setInformation(wv);
    optimizer.addEdge(edge);
  }

  size_t n_iter = 20;
  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);

  closed_loops_[std::make_pair(curr_frame,loop_frame)] = T_curr_loop;

  std::set<Frame*> updated_keyframes;

  for(auto it : frames){
    Frame* frame = it.second;
    if(v_poses.count(frame) ) {
      g2o::SE3Quat Tcw = v_poses.at(frame)->estimate();
      frame->SetTcw(Tcw);
    }
    else{
      const g2o::SE3Quat& Tc1curr = mTc1curr.at(frame);
      const g2o::SE3Quat& Tcurrw = v_poses.at(curr_frame)->estimate();
      g2o::SE3Quat Tc1w = Tc1curr * Tcurrw;
      frame->SetTcw(Tc1w);
    }
    updated_keyframes.insert(frame);
  }
  map_->UnLock();

  {
    std::cout << "Loop closing between " << loop_frame->GetIndex() << " and "
      << curr_frame->GetIndex() << " done" << std::endl;
  }

  // Merging mappoints.
  CombineNeighborMappoints(curr_frame, loop_frame);

  return updated_keyframes;
}


void LoopCloser::CombineNeighborMappoints(Frame* curr_frame, Frame* loop_frame) const {
  double search_radius = 10.;
  map_->Lock();
  std::set<Frame*> loop_frames, curr_frames;
  std::set<Mappoint*> loop_mappoints, curr_mappoints;
  loop_frame->GetNeighbors(loop_frames, loop_mappoints);
  curr_frame->GetNeighbors(curr_frames, curr_mappoints);
  curr_frames.insert(curr_frame);
  map_->UnLock();

  std::map< Mappoint*, std::set<Mappoint*> > merge_candidates;
  std::map< std::pair<Mappoint*, Mappoint*>, size_t> duplication_numbers;

  for(Frame* curr_nkf : curr_frames){
    std::map<int, Mappoint*> curr_keypoint_matches
      = ProjectionMatch(loop_mappoints, curr_nkf->GetTcw(), curr_nkf, search_radius);
    for(auto it_match : curr_keypoint_matches){
      Mappoint* mp_curr = curr_nkf->GetMappoint(it_match.first);
      if(!mp_curr){
        continue;
      }
      Mappoint* mp_loop = it_match.second;
      duplication_numbers[std::make_pair(mp_curr, mp_loop)]++;
      merge_candidates[mp_curr].insert(mp_loop);
    }
  }

  std::set<Mappoint*> closed_mp_lists;
  std::set<Mappoint*> merged_mp_lists;

  map_->Lock();
  for(auto it: merge_candidates){
    Mappoint* mp_curr = it.first;
    Mappoint* mp_champ = nullptr;
    size_t nmax = 0;
    for(Mappoint* mp_loop : it.second){
      size_t n = duplication_numbers.at(std::make_pair(mp_curr,mp_loop));
      if(n> nmax){
        mp_champ = mp_loop;
        nmax = n;
      }
    }

    if(closed_mp_lists.count(mp_champ))
      continue;
    if(merged_mp_lists.count(mp_curr))
      continue;
    if(closed_mp_lists.count(mp_curr)){
      //throw std::invalid_argument("3 Loop closing cases");
      continue;
    }

    closed_mp_lists.insert(mp_champ);
    merged_mp_lists.insert(mp_curr);

    // Change mp_champ to mp_curr.
    // Reserve mp_curr considering unprocessed frame at pose track thread.
    std::set<Frame*> frames_of_mp_delete = mp_champ->GetKeyframes();
    const bool already_locked = true;
    auto latests = map_->GetLatestFrames(5, already_locked);
    for(Frame* frame : latests) // non keyframe frame에 포함된 경우를 확인하기위해.
      frames_of_mp_delete.insert(frame);

    for(Frame* frame : frames_of_mp_delete){
      if(frame->GetIndex(mp_champ) < 0)
        continue;
      int kpt_idx = frame->EraseMappoint(mp_champ);
      if(frame->GetIndex(mp_curr) < 0)
        frame->SetMappoint(mp_curr, kpt_idx);
    }
    mp_champ->SetBad();
    //delete mp_champ;
  }
  map_->UnLock();
  std::cout << "Merging is done. n(Merge) = " << merge_candidates.size() << std::endl;
  return;
}
