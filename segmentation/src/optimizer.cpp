#include "optimizer.h"
#include "Eigen/src/Core/Matrix.h"
#include "common.h"
#include "g2o_types.h"
#include "segslam.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

inline float GetInvdInfo(const float& invd){
  return invd*invd*1e-2;
  //return invd*invd*1e+2; //별 차이 없었음.
}

inline float GetSwitchableInvdInfo(const float& invd){
  return invd*invd;
  //return invd*invd*1e+3;
}

namespace NEW_SEG {
PoseTracker::PoseTracker() {
}
PoseTracker::~PoseTracker() {
}
g2o::SE3Quat PoseTracker::GetTcq(const Camera* camera,
                                 Qth qth,
                                 Frame* curr_frame,
                                 bool vis_verbose
                                ) {
  Param param(camera);
  g2o::SparseOptimizer optimizer;
  // Dynamic block size due to 6dim SE3, 1dim prior vertex.
  typedef g2o::BlockSolverPL<-1,-1> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
#if 1
  g2o::OptimizationAlgorithm* solver \
    =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#else
  g2o::OptimizationAlgorithm* solver \
    = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#endif
  optimizer.setAlgorithm(solver);
  auto v_pose = new g2o::VertexSE3Expmap();
  v_pose->setId(optimizer.vertices().size() );
  optimizer.addVertex(v_pose);
  const auto& Tcq = curr_frame->GetTcq(qth);
  v_pose->setEstimate(Tcq);

  const std::vector<Mappoint*>& mappoints = curr_frame->GetMappoints();
  const std::vector<cv::KeyPoint>& keypoints = curr_frame->GetKeypoints();
  const std::vector<float>& depths = curr_frame->GetMeasuredDepths();
  const EigenVector<Eigen::Vector3d>& normalized_points = curr_frame->GetNormalizedPoints();

  const double focal = camera->GetK()(0,0);
  const double uv_info = 1.;
  const double invd_info = 1.;
  const double delta = 10./focal;
  const int n_iter = 10;

  std::map<Mappoint*, g2o::OptimizableGraph::Vertex*> v_mappoints;
  for(size_t n=0; n<mappoints.size(); n++){
    Mappoint* mp = mappoints[n];
    if(!mp) continue;
    const cv::Point2f& pt = keypoints[n].pt;
    auto v_mp = new g2o::VertexSBAPointXYZ();
    v_mp->setId(optimizer.vertices().size() );
    v_mp->setEstimate(mp->GetXq(qth));
    v_mp->setMarginalized(true);
    v_mp->setFixed(true);
    optimizer.addVertex(v_mp);
    v_mappoints[mp]  = v_mp;
    const float& z = depths[n];
    float measured_invd = 1./std::max<float>(z, MIN_NUM);
    Eigen::Vector3d uvi = normalized_points[n];
    g2o::OptimizableGraph::Edge* edge = nullptr;
    if(z >  MIN_NUM){
      auto ptr = new EdgeSE3PointXYZDepth(&param, uv_info, GetInvdInfo(measured_invd));
      uvi[2] = measured_invd;
      ptr->setVertex(0, v_mp);
      ptr->setVertex(1, v_pose);
      ptr->setMeasurement( uvi );
      edge = ptr;
    }
    else{
      auto ptr = new EdgeProjection(&param, uv_info);
      ptr->setVertex(0, v_mp);
      ptr->setVertex(1, v_pose);
      ptr->setMeasurement( uvi.head<2>() );
      edge = ptr;
    }
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(delta);
    edge->setRobustKernel(rk);
    optimizer.addEdge(edge);
  }

  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);
  return v_pose->estimate();
}

inline bool IsDepthOutlier(const double& invd1, const double& invd2){
  const double near_invd =  1 / 20.; // [m]^-1
  if(invd1 < near_invd)
    return false;
  if(invd2 < near_invd)
    return false;
  return std::abs( (1./invd1) - (1./invd2) ) > 1.;
}

std::map<Pth,float> Mapper::ComputeLBA(const Camera* camera,
                        Qth qth,
                        const std::set<Mappoint*>& neighbor_mappoints,
                        const std::map<Jth, Frame*>& neighbor_keyframes,
                        Frame* curr_frame,
                        Frame* prev_frame,
                        const std::set<Pth>& fixed_instances,
                        const cv::Mat& gradx,
                        const cv::Mat& grady,
                        const cv::Mat& valid_grad,
                        bool vis_verbose
                        ) {
  const double focal = camera->GetK()(0,0);
  const double uv_info = 1.;
  const double rprj_threshold = 5./focal; // rprj error threshold on normalized inmage plane
  const double delta = .5 * rprj_threshold;
  const int n_iter = 10;
  std::map<Jth, Frame*> frames = neighbor_keyframes;
  frames[curr_frame->GetId()] = curr_frame;
  Param param(camera);
  g2o::SparseOptimizer optimizer;
  // Dynamic block size due to 6dim SE3, 1dim prior vertex.
  typedef g2o::BlockSolverPL<-1,-1> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
#if 1
  g2o::OptimizationAlgorithm* solver \
    =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#else
  g2o::OptimizationAlgorithm* solver \
    = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#endif
  optimizer.setAlgorithm(solver);

  std::map<Jth, std::pair<g2o::VertexSE3Expmap*,size_t> > v_poses; // with n_filtered_edges
  std::map<Mappoint*, g2o::VertexSBAPointXYZ*> v_mappoints;
  std::map<Pth, VertexSwitchLinear*>  v_switches;
  std::map<Pth, size_t>               mp_counts;
  std::map<Pth, EdgeSwitchPrior*>     prior_edges;
  std::map<Pth, double>               swedges_parameter;

  Frame*const kf_latest = neighbor_keyframes.rbegin()->second;
  const int kf_id_latest = kf_latest->GetKfId(qth);
  Frame*const kf_oldest = neighbor_keyframes.begin()->second;

  for(Mappoint* mp : neighbor_mappoints){
    // n_kf는 mappoint의 keyframes가 아니라, LBA의 keyframes.
    size_t n_kf = mp->GetKeyframes(qth).size();
    if(n_kf < 2){ // Projection Edge만 1개 생기는 경우 NAN이 발생하는것을 막기위해.
      if(curr_frame->GetIndex(mp) < 0 )
        continue;
    }
    //Instance* ins = mp->GetInstance();
    Instance* ins = mp->GetLatestInstance();
    if(ins && ins->GetId() > 0 ){
      mp_counts[ins->GetId()]++;
    }
    g2o::VertexSBAPointXYZ* v_mp = new g2o::VertexSBAPointXYZ();
    v_mp->setId(optimizer.vertices().size() );
    v_mp->setEstimate(mp->GetXq(qth));
    v_mp->setMarginalized(true);
    optimizer.addVertex(v_mp);
    v_mappoints[mp] = v_mp;
  }
  for(auto it_pth : mp_counts){
    //if(it_pth.second < 5) // Tracked mappoints가 너무 적은 instance는 생략.
    //  continue;
    auto v_switch = new VertexSwitchLinear();
    v_switch->setId(optimizer.vertices().size() );
    v_switch->setEstimate(1.);
    optimizer.addVertex(v_switch);
    auto sw_prior_edge = new EdgeSwitchPrior();
    sw_prior_edge->setMeasurement(1.);
    sw_prior_edge->setVertex(0, v_switch);
    sw_prior_edge->setLevel(0);
    optimizer.addEdge(sw_prior_edge);
    v_switches[it_pth.first] = v_switch;
    prior_edges[it_pth.first] = sw_prior_edge;
    if(fixed_instances.count(it_pth.first))
      v_switch->setFixed(true);
  }
  std::map<Pth, std::list<g2o::OptimizableGraph::Edge*> > filtered_edges;
  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    Jth jth = it_frame.first;
    g2o::VertexSE3Expmap* v_pose = new g2o::VertexSE3Expmap();
    v_pose->setId(optimizer.vertices().size() );
    v_pose->setEstimate(frame->GetTcq(qth));
    v_pose->setMarginalized(false);
    optimizer.addVertex(v_pose);
    size_t n_filtered_edges = 0;
    const auto& jth_mappoints = frame->GetMappoints();
    for(size_t n=0; n<jth_mappoints.size(); n++){
      Mappoint* mp = jth_mappoints[n];
      if(!mp)
        continue;
      if(!v_mappoints.count(mp)) // ins가 exclude 된 케이스라 neighbors에서 제외됬을 수 있다.
        continue;
      auto v_mp = v_mappoints.at(mp);
      const cv::KeyPoint& kpt = frame->GetKeypoint(n);
      Eigen::Vector3d uvi = frame->GetNormalizedPoint(n);
      const float& z = frame->GetDepth(n);
      double measure_invd = 1. / std::max<float>(z, MIN_NUM); // 함수화?
      uvi[2] = measure_invd;
      g2o::Vector3 h(v_pose->estimate().map(v_mp->estimate()));
      double h_invd = GetInverse(h[2]);
      h.head<2>() *= h_invd; // [Xc, Yc] /Zc
      h[2] = h_invd;
      double greater_invd = std::max(measure_invd,h_invd);

      Instance* ins = mp->GetInstance();
      Pth pth = ins ? ins->GetId() : -1;
      bool valid_depth = z > 5.; // 너무 가까운데서 stereo disparity가 부정확.
      bool switchable = v_switches.count(pth) ;
      if(switchable ){
        VertexSwitchLinear* v_switch = v_switches.at(pth);
        swedges_parameter[pth] += 1.;
        g2o::OptimizableGraph::Edge* edge_swtichable = nullptr;
        if(valid_depth) {
          auto ptr = new EdgeSwSE3PointXYZDepth(&param, uv_info, GetSwitchableInvdInfo(greater_invd));
          ptr->setVertex(0, v_mp);
          ptr->setVertex(1, v_pose);
          ptr->setVertex(2, v_switch);
          ptr->setMeasurement( uvi );
          edge_swtichable = ptr;
        }
        else {
          auto ptr = new EdgeSwProjection(&param, uv_info);
          ptr->setVertex(0, v_mp);
          ptr->setVertex(1, v_pose);
          ptr->setVertex(2, v_switch);
          ptr->setMeasurement( uvi.head<2>() );
          edge_swtichable = ptr;
        }
        edge_swtichable->setLevel(0);
        optimizer.addEdge(edge_swtichable);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta);
        edge_swtichable->setRobustKernel(rk);
      }

      g2o::Vector2 rprj_err(h.head<2>() - uvi.head<2>() );
      if(IsDepthOutlier(h[2], uvi[2]) )
        continue;
      if( std::abs(rprj_err[0]) > rprj_threshold || std::abs(rprj_err[1]) > rprj_threshold)
        continue;
      g2o::OptimizableGraph::Edge* edge_filtered = nullptr;
      if(valid_depth) {
        float invd_info = GetInvdInfo(measure_invd); // 10.7
        auto ptr = new EdgeSE3PointXYZDepth(&param, uv_info, invd_info);
        ptr->setVertex(0, v_mp);
        ptr->setVertex(1, v_pose);
        ptr->setMeasurement( uvi );
        edge_filtered = ptr;
      }
      else {
        auto ptr = new EdgeProjection(&param, uv_info);
        ptr->setVertex(0, v_mp);
        ptr->setVertex(1, v_pose);
        ptr->setMeasurement( uvi.head<2>() );
        edge_filtered = ptr;
      }
      edge_filtered->setLevel(1);
      if(switchable)
        filtered_edges[pth].push_back(edge_filtered);
      optimizer.addEdge(edge_filtered);
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(delta);
      edge_filtered->setRobustKernel(rk);
      n_filtered_edges++;
    } // for mappoints
    v_poses[jth] = std::pair(v_pose, n_filtered_edges);
  } // for frames

  for(auto it_v_sw : prior_edges){
    const auto& n = swedges_parameter.at(it_v_sw.first);
    double info = 1e-7 * n * n;
    it_v_sw.second->SetInfomation(info);
  }

  optimizer.setVerbose(false);
#if 0
  if(!v_switches.empty()){
    for(auto it_mp : v_mappoints)
      it_mp.second->setFixed(true);
    for(auto it_pose : v_poses)
      it_pose.second.first->setFixed(true);
    optimizer.initializeOptimization(0); // Optimize switchable edges only
    optimizer.optimize(n_iter);
    for(auto it : v_switches)
      it.second->setFixed(true);
    for(auto it : filtered_edges){
      VertexSwitchLinear* v_switch  = v_switches.at(it.first);
      if(v_switch->estimate() < .3)
        for(auto edge : it.second)
          edge->setLevel(0);
    }
  }
  for(auto it_mp : v_mappoints){
    Mappoint* mp = it_mp.first;
#if 1
    int kf_id_ref = mp->GetRefFrame(qth)->GetKfId(qth);
    it_mp.second->setFixed(kf_id_latest-kf_id_ref > 5);
#else
    int ref_pth = mp->GetRefFrame(qth)->GetId();
    if(v_poses.count(ref_pth) && !v_poses.at(ref_pth).first->fixed() )
      it_mp.second->setFixed(false);
    else
      it_mp.second->setFixed(true);
#endif
  }
  for(auto it_pose : v_poses)
    it_pose.second.first->setFixed(it_pose.second.second < 20);
  if(!v_poses.empty()){
    g2o::OptimizableGraph::Vertex* v_oldest = v_poses.begin()->second.first;
    v_oldest->setFixed(true);
  }
#else

  for(auto it_mp : v_mappoints){
    Mappoint* mp = it_mp.first;
    //int kf_id_ref = mp->GetRefFrame(qth)->GetKfId(qth);
    //it_mp.second->setFixed(kf_id_latest-kf_id_ref > 5);
    int ref_pth = mp->GetRefFrame(qth)->GetId();
    if(v_poses.count(ref_pth) && !v_poses.at(ref_pth).first->fixed() )
      it_mp.second->setFixed(false);
    else
      it_mp.second->setFixed(true);
  }
  if(!v_switches.empty()){
    for(auto it_pose : v_poses)
      it_pose.second.first->setFixed(it_pose.first != curr_frame->GetId());

    optimizer.initializeOptimization(0); // Optimize switchable edges only
    optimizer.optimize(n_iter);
    for(auto it : v_switches)
      it.second->setFixed(true);
    for(auto it : filtered_edges){
      VertexSwitchLinear* v_switch  = v_switches.at(it.first);
      if(v_switch->estimate() < .3)
        for(auto edge : it.second)
          edge->setLevel(0);
    }
  }
#endif
  for(auto it_pose : v_poses)
    it_pose.second.first->setFixed(it_pose.second.second < 20);
  g2o::OptimizableGraph::Vertex* v_oldest = v_poses.at(kf_oldest->GetId()).first;
  v_oldest->setFixed(true);
  optimizer.initializeOptimization(1);  // Optimize filtered edges only
  optimizer.optimize(n_iter);
  // Retrieve
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  const Jth curr_jth = curr_frame->GetId();
  int n_pose = 0;
  int N_pose = v_poses.size();
  bool txt_verbose = false;
  if(txt_verbose)
    std::cout << "At Q#" << qth << " curr F#"<< curr_frame->GetId() <<"--------------------" << std::endl;
  for(auto it_poses : v_poses){
    g2o::VertexSE3Expmap* vertex = it_poses.second.first;
    if(txt_verbose)
      std::cout << "F#" << it_poses.first << ", " << vertex->estimate().inverse().translation().transpose() << std::endl;
    if(vertex->fixed())
      continue;
    frames[it_poses.first]->SetTcq(qth, vertex->estimate());
    n_pose++;
  }
  if(txt_verbose)
    std::cout << "---------------------" << std::endl;
  int n_structure = 0;
  int N_structure = v_mappoints.size();
  for(auto it_v_mpt : v_mappoints){
    g2o::VertexSBAPointXYZ* vertex = it_v_mpt.second;
    if(vertex->fixed())
      continue;
    n_structure++;
    Mappoint* mpt = it_v_mpt.first;
    mpt->SetXq(qth,vertex->estimate());
  }
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  std::map<Pth, float> switch_state; // posterior for inlier.
  for(auto it : v_switches)
    switch_state[it.first] = it.second->estimate();
  return switch_state;
}

} // namespace NEW_SEG

