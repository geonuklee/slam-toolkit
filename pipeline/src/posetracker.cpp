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

#include "posetracker.h"
#include "optimizer.h"
#include "matcher.h"
#include "frame.h"
#include "mappoint.h"
#include "camera.h"
#include "orb_extractor.h"
#include "method.h"

const float deltaMono = sqrt(5.991); // TODO Why?
const float thHuber3D = sqrt(7.815);
using BlockSolver_8_3 = g2o::BlockSolverPL<8, 3>;

BasicPoseTracker::BasicPoseTracker(StandardMethod* method)
: method_(method) {

}

void BasicPoseTracker::Track(const std::set<Mappoint*>& mappoints,
                                const void* predict,
                                void* estimation,
                                int n_iter,
                                Frame* frame
                                ){
  BeforeEstimation(mappoints, predict, frame);
  EstimatePose(mappoints, predict, estimation, n_iter,frame);
  RetriveEstimation(estimation, frame);
}

void BasicPoseTracker::EstimatePose(const std::set<Mappoint*>& mappoints,
                                       const void* predict,
                                       void* estimation,
                                       int n_iter,
                                       const Frame* frame
                                       ) {
  g2o::SparseOptimizer optimizer;
  method_->SetSolver(StandardMethod::LM, optimizer);
  g2o::OptimizableGraph::Vertex* v_pose = nullptr;

  std::map<g2o::OptimizableGraph::Edge*, std::pair<const Frame*,Mappoint*> > measurment_edges;
  InitializeGraph(mappoints, predict, frame, optimizer, v_pose, measurment_edges);

  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);

  method_->RetrivePose(v_pose, estimation);
  return;
}

void BasicPoseTracker::InitializeGraph(const std::set<Mappoint*>& mappoints,
                                          const void* predict,
                                          const Frame* frame,
                                          g2o::SparseOptimizer& optimizer,
                                          g2o::OptimizableGraph::Vertex*& v_pose,
                                          std::map<g2o::OptimizableGraph::Edge*, std::pair<const Frame*,Mappoint*> >& measurment_edges){

  v_pose = method_->CreatePoseVertex(optimizer, nullptr);
  method_->SetPose(predict, v_pose);
  method_->SetPose(predict, v_pose);

  for(Mappoint* mp : mappoints){
    if(!HasMeasurement(frame,mp))
      continue;
    auto v_mp = method_->CreateStructureVertex(optimizer, mp);
    v_mp->setMarginalized(true);
    v_mp->setFixed(true);
    g2o::OptimizableGraph::Edge* edge = method_->CreateMeasurementEdge(frame, mp, false);
    SetMeasurement(frame, mp, edge);
    edge->setVertex(0, v_mp);
    edge->setVertex(1, v_pose);
    optimizer.addEdge(edge);
    measurment_edges[edge] = std::make_pair(frame, mp);
  }

  return;
}

ReprojectionFilter::ReprojectionFilter(double max_reprojection_error)
  : max_reprojection_error_(max_reprojection_error){

}

std::list<std::pair<size_t, Mappoint*> > ReprojectionFilter::GetOutlier(const Frame* frame,
                                                                        const void* expected_pose
                                                                       )
{
  const g2o::SE3Quat& Tcw = *(g2o::SE3Quat*) expected_pose;
  const std::vector<cv::KeyPoint>& keypoints = frame->GetKeypoints();
  const std::vector<Mappoint*> mappoints = frame->GetVecMappoints(); // GetOutlier
  const Camera* camera = frame->GetCamera();
  std::list<std::pair<size_t, Mappoint*> > outliers;
  for(size_t i = 0; i < keypoints.size(); i++) {
    Mappoint* mp = mappoints.at(i);
    const cv::KeyPoint& kpt = keypoints.at(i);
    if(!mp)
      continue;
    // ProjectionMatch 참고
    // TODO 20.12.31 이유가 있어 이렇게 했을텐데, 기억이 안남. 확신이 안서니 아래 if statement삭제를 완성 후 성능 비교후로 미룸.
    if(frame->GetIndex(mp) >= 0)
      continue;
    Eigen::Vector3d Xw = mp->GetXw();
    Eigen::Vector3d Xc = Tcw * Xw;
    if(Xc.z() < 0.){
      outliers.push_back(std::make_pair(i, mp));
      continue;
    }
    Eigen::Vector2d uv0(kpt.pt.x, kpt.pt.y);
    Eigen::Vector2d uv1 = camera->Project(Xc);
    double d = (uv1-uv0).norm();
    if(d > max_reprojection_error_)
      outliers.push_back(std::make_pair(i, mp));
  }
  return outliers;
}


PhotometricErrorFilter::PhotometricErrorFilter(double max_error)
  : max_error_(max_error) {

}

std::list<std::pair<size_t, Mappoint*> > PhotometricErrorFilter::GetOutlier(const Frame* frame,
                                                                            const void* expected_pose
                                                                           ){
  const BrightenSE3& pose = *( BrightenSE3*) expected_pose;
  const std::vector<Mappoint*> mappoints = frame->GetVecMappoints(); // GetOutlier
  const Camera* camera = frame->GetCamera();
  std::list<std::pair<size_t, Mappoint*> > outliers;
  for(size_t i = 0; i < mappoints.size(); i++) {
    Mappoint* mp = mappoints.at(i);
    if(!mp)
      continue;
    Eigen::Vector3d Xw = mp->GetXw();
    Eigen::Vector3d Xc = pose.Tcw_ * Xw;
    if(Xc.z() < 0.){
      outliers.push_back(std::make_pair(i, mp));
      continue;
    }
    Frame* frame0 = mp->GetRefFrame();
    EdgeProjectBrightenXYZ::Measurement error;
    EdgeProjectBrightenXYZ::GetError(frame0->GetImage(), frame->GetImage(),
                                     frame0->GetBrightenPose(), 
                                     mp, 1.,
                                     frame->GetBrightenPose(),
                                     mp->GetXw(),
                                     error);
    if(error.lpNorm<1>() > max_error_)
      outliers.push_back(std::make_pair(i, mp));
  }
  return outliers;
}

IndirectPoseTracker::IndirectPoseTracker(const std::vector<float>& inv_scales_sigma2)
  : BasicPoseTracker(new IndirectStereoMethod(inv_scales_sigma2) )
{
}

void IndirectPoseTracker::BeforeEstimation(const std::set<Mappoint*>& mappoints,
                                           const void* predict,
                                           const Frame* frame){
  predicted_Tcw_ = *(g2o::SE3Quat*) predict;
  double search_radius = 50.;
  auto keypoint_matches= ProjectionMatch(mappoints, predicted_Tcw_, frame, search_radius);
  if(keypoint_matches.size() < 8){
    search_radius *= 2.;
    keypoint_matches = ProjectionMatch(mappoints, predicted_Tcw_, frame, search_radius);
  }

  matched_mappoints_.clear();
  for(auto it : keypoint_matches)
    matched_mappoints_[it.second] = it.first;
  keypoints_ = frame->GetKeypoints();
  return;
}

void IndirectPoseTracker::RetriveEstimation(const void* estimation, Frame* frame){ // Update frame->mappoints_
  for(auto it_match : matched_mappoints_){
    Mappoint* mp = it_match.first;
    int kpt_idx = it_match.second;
    if(frame->GetMappoint(kpt_idx)) // 이전 Pipeline에서 이미 matching이 된 keypoint
      continue;
    frame->SetMappoint(mp, kpt_idx);
  }

  ReprojectionFilter filter(10);
  auto outliers = filter.GetOutlier(frame, estimation);

  size_t n_after_remove_outliers = frame->GetMappoints().size() - outliers.size();
  if(n_after_remove_outliers > 8){
    for(auto it : outliers){
      Mappoint* mp = it.second;
      frame->EraseMappoint(mp);
    }
  }
  const g2o::SE3Quat& estimated_Tcw = *(g2o::SE3Quat*) estimation;
  frame->SetTcw(estimated_Tcw);
  return;
}

bool IndirectPoseTracker::HasMeasurement(const Frame* frame, Mappoint* mp) {
  return matched_mappoints_.count(mp);
}

void IndirectPoseTracker::SetMeasurement(const Frame* frame, Mappoint* mp, g2o::OptimizableGraph::Edge* _edge) {
  g2o::EdgeSE3ProjectXYZ* edge = static_cast<g2o::EdgeSE3ProjectXYZ*>(_edge);
  IndirectStereoMethod* method = static_cast<IndirectStereoMethod*>(method_.get());
  const auto& inv_scale_sigma2 = method->GetInvScaleSigma2();

  const cv::KeyPoint& kpt = keypoints_.at(matched_mappoints_.at(mp) );
  int kpt_idx = matched_mappoints_.at(mp);
  Eigen::Vector2d obs = frame->GetNormalizedPoint(kpt_idx);
  edge->setMeasurement(obs);
  const float invSigma2 = inv_scale_sigma2.at(kpt.octave);
  edge->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;

  const float deltaMono = sqrt(5.991); // TODO Why?
  rk->setDelta(0.0001*deltaMono);
  edge->setRobustKernel(rk);
  edge->fx = 1.;
  edge->fy = 1.;
  edge->cx = 0.;
  edge->cy = 0.;
  return;
}

BrightenDirectPoseTracker::BrightenDirectPoseTracker(double min_search_radius)
  : BasicPoseTracker(new DirectStereoMethod ),
    min_search_radius_(min_search_radius)
{
}

bool BrightenDirectPoseTracker::HasMeasurement(const Frame* frame, Mappoint* mp){
  const Camera* camera = frame->GetCamera();
  Eigen::Vector3d Xc = predicted_pose_.Tcw_ * mp->GetXw();
  Eigen::Vector2d uv = camera->Project(Xc);
  return camera->IsInImage(uv);
}

void BrightenDirectPoseTracker::SetMeasurement(const Frame* frame, Mappoint* mp, g2o::OptimizableGraph::Edge* edge){

}

void BrightenDirectPoseTracker::BeforeEstimation(const std::set<Mappoint*>& mappoints,
                                                 const void* predict,
                                                 const Frame* frame) {
  if(frame->IsKeyframe())
    throw std::invalid_argument("Input of PoseTracker::Track supposed be non keyframe.");

  mappoints_  = mappoints;
  predicted_pose_= *(BrightenSE3*) predict;
  return;
}

void BrightenDirectPoseTracker::RetriveEstimation(const void* estimation, Frame* frame) {
  // Direct method에서의 frame->mappoints 취급. descriptor와 무관하게라도, keypoint 연결해야 할것으로 보임.
  const double max_reprojection_error = 5.;
  const Camera* camera = frame->GetCamera();
  const BrightenSE3& estimated_pose = *(BrightenSE3*) estimation;

  for(Mappoint* mp : mappoints_){
    Eigen::Vector3d Xw = mp->GetXw();
    Eigen::Vector3d Xc = estimated_pose.Tcw_ * Xw;
    Eigen::Vector2d uv = camera->Project(Xc);
    if(!camera->IsInImage(uv) )
      continue;
    int kpt_idx;
    double reprj_error;
    frame->SearchNeareast(uv, kpt_idx, reprj_error);
    if(kpt_idx<0)
      continue;
    if(reprj_error > max_reprojection_error)
      continue;
    if(frame->GetMappoint(kpt_idx)) // 이전 Pipeline에서 이미 matching이 된 keypoint
      continue;

    frame->SetMappoint(mp, kpt_idx);
  }

  frame->SetTcw(estimated_pose.Tcw_);
  frame->SetBrightness(estimated_pose.brightness_);

  PhotometricErrorFilter filter(200.);
  auto outliers = filter.GetOutlier(frame, estimation);

  size_t n_after_remove_outliers = frame->GetMappoints().size() - outliers.size();
  if(n_after_remove_outliers > 8){
    for(auto it : outliers){
      Mappoint* mp = it.second;
      frame->EraseMappoint(mp);
    }
  }
  return;
}

void BrightenDirectPoseTracker::EstimatePose(const std::set<Mappoint*>& mappoints,
                                       const void* predict,
                                       void* estimation,
                                       int n_iter,
                                       const Frame* frame
                                       ){
  const Camera* camera = frame->GetCamera();
  DirectStereoMethod* method = static_cast<DirectStereoMethod*>(method_.get());
  std::shared_ptr<DirectPyramid> pyramid = method->GeyPyramid();

  int min_lv = 0;
  int max_lv = 0;
  double r = Pattern::GetRadius();
  double min_search_radius = std::min(min_search_radius_, 0.5*camera->GetWidth() );
  min_search_radius = std::min(min_search_radius, 0.5*camera->GetHeight() );
  while( r <  min_search_radius){
    r /= pyramid->GetRatio();
    max_lv += 1;
  }

  g2o::SparseOptimizer optimizer;
  method_->SetSolver(StandardMethod::LM, optimizer);
  g2o::OptimizableGraph::Vertex* v_pose = nullptr;
  std::map<g2o::OptimizableGraph::Edge*, std::pair<const Frame*,Mappoint*> > measurment_edges;
  InitializeGraph(mappoints, predict, frame, optimizer, v_pose, measurment_edges);
  optimizer.initializeOptimization();

  for(int lv = max_lv; lv >= min_lv; lv--){
    pyramid->SetLv(lv);
    optimizer.optimize(n_iter);
  }

  method_->RetrivePose(v_pose, estimation);
  return;
}

