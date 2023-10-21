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

#include <boost/math/distributions/chi_squared.hpp>

namespace NEW_SEG {

double ChiSquaredThreshold(double p, double dof){
  boost::math::chi_squared_distribution<> chi2dist( dof );
  double t = boost::math::quantile(chi2dist, p);
  return t;
}

inline Instance* GetIns(Mappoint* mp){
  //return mp->GetInstance();
  return mp->GetLatestInstance();
}

g2o::SE3Quat EstimateTcp(const std::vector<cv::Point3f>& Xp,
                         const std::vector<cv::Point3f>& vec_uvz_curr,
                         const Camera* camera, double uv_info, double invd_info, double delta,
                         std::vector<double>&vec_chi2
                         ) {
  const int n_iter = 5;
  g2o::SparseOptimizer optimizer;
  // Dynamic block size due to 6dim SE3, 1dim prior vertex.
  typedef g2o::BlockSolverPL<-1,-1> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
  g2o::OptimizationAlgorithm* solver =  new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);
  Param param(camera);

  g2o::VertexSE3Expmap* v_pose = new g2o::VertexSE3Expmap();
  v_pose->setId(optimizer.vertices().size() );
  v_pose->setMarginalized(false);
  optimizer.addVertex(v_pose);

  std::vector<g2o::OptimizableGraph::Edge*> edges;
  edges.reserve(Xp.size());
  for(size_t i = 0; i < Xp.size(); i++){
    const auto& xp = Xp.at(i);
    g2o::VertexSBAPointXYZ* v_mp = new g2o::VertexSBAPointXYZ();
    v_mp->setId(optimizer.vertices().size() );
    v_mp->setEstimate(Eigen::Vector3d(xp.x, xp.y, xp.z) );
    v_mp->setMarginalized(true);
    v_mp->setFixed(true);
    optimizer.addVertex(v_mp);
    const auto& uvz = vec_uvz_curr.at(i);
    bool valid_depth = uvz.z > 1e-5 && uvz.z < 80.; // ex) seq00 너무 먼, 건물에서 depth residual이 과하게 증가. HITNET의 오인식으로 의심.
    Eigen::Vector3d uvi(uvz.x, uvz.y, valid_depth?1./uvz.z:0.);
    g2o::OptimizableGraph::Edge* edge;
    if(valid_depth)
      edge = new EdgeSE3PointXYZDepth(&param, uv_info, invd_info, v_mp, v_pose, uvi);
    else
      edge = new EdgeProjection(&param, uv_info, v_mp, v_pose, uvi.head<2>() );
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(delta);
    edge->setRobustKernel(rk);
    edges.push_back(edge);
    optimizer.addEdge(edge);
  }

  optimizer.setVerbose(false);
  optimizer.initializeOptimization(0); // Optimize switchable edges only
  optimizer.optimize(n_iter);

  vec_chi2.reserve(Xp.size());
  for(auto edge : edges)
    vec_chi2.push_back(edge->chi2());
  return v_pose->estimate();
}

PoseTracker::PoseTracker() {
}
PoseTracker::~PoseTracker() {
}
g2o::SE3Quat PoseTracker::GetTcq(const Camera* camera,
                                 Qth qth,
                                 Frame* curr_frame,
                                 bool vis_verbose
                                ) {
  const std::vector<Mappoint*>&    _mappoints = curr_frame->GetMappoints();
  const std::vector<cv::KeyPoint>& _keypoints = curr_frame->GetKeypoints();
  const std::vector<Instance*>&    _instances = curr_frame->GetInstances();
  const std::vector<float>&        _depths    = curr_frame->GetMeasuredDepths();
  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  obj_points.reserve(_mappoints.size());
  img_points.reserve(_mappoints.size());

  for(int n =0; n < _mappoints.size(); n++){
    Mappoint* mp = _mappoints.at(n);
    if(!mp)
      continue;
    Instance* ins = GetIns(mp);
    if(!ins)
      continue;
    if(ins->GetQth() != qth)
      continue;
    const Eigen::Vector3d Xq = mp->GetXq(qth);
    obj_points.push_back( cv::Point3f(Xq[0],Xq[1],Xq[2]) );
    img_points.push_back(_keypoints.at(n).pt);
  }

  double rprj_threshold = 5.;
  int iteration = 100;
  bool use_extrinsic_guess = true;
  double confidence = .99;
  int flag = cv::SOLVEPNP_EPNP;
  cv::Mat K = cvt2cvMat(camera->GetK() );
  cv::Mat D = cvt2cvMat(camera->GetD() );
  cv::Mat rvec, tvec, inliers;
  const g2o::SE3Quat Tcq0 = curr_frame->GetTcq(qth);
  tvec = cvt2cvMat(Tcq0.translation());
  cv::Mat R = cvt2cvMat( Tcq0.rotation().toRotationMatrix() );
  cv::Rodrigues(R,rvec);
  cv::solvePnPRansac(obj_points, img_points, K, D, rvec, tvec,
                     use_extrinsic_guess, iteration, rprj_threshold, confidence, inliers, flag);
  cv::Rodrigues(rvec,R);
  g2o::SE3Quat Tcq(cvt2Eigen(R), cvt2Eigen(tvec) ); // Transform : {c}urrent camera <- {p}revious camera
  return Tcq;
}

void GetEdges(g2o::VertexSBAPointXYZ* v_mp, g2o::VertexSE3Expmap* v_pose, VertexSwitchLinear* v_switch,
              const Param& param, double uv_info, double invd_info, double delta,
              double rprj_threshold, double invd_threshold,
              Frame* frame, int kpt_idx,
              g2o::OptimizableGraph::Edge*& edge_switchable,
              g2o::OptimizableGraph::Edge*& edge_filtered
              ) {
  const cv::KeyPoint& kpt = frame->GetKeypoint(kpt_idx);
  Eigen::Vector3d uvi = frame->GetNormalizedPoint(kpt_idx);
  const float& z = frame->GetDepth(kpt_idx);
  double measure_invd = GetInverse(z);
  uvi[2] = measure_invd;
  g2o::Vector3 h(v_pose->estimate().map(v_mp->estimate()));
  double h_invd = GetInverse(h[2]);
  h.head<2>() *= h_invd; // [Xc, Yc] /Zc
  h[2] = h_invd;
  double greater_invd = std::max(measure_invd,h_invd);
  bool valid_depth = z > 1e-5 && z < 80.; // ex) seq00 너무 먼, 건물에서 depth residual이 과하게 증가. HITNET의 오인식으로 의심.
  edge_switchable = nullptr;
  edge_filtered = nullptr;
  g2o::Vector2 rprj_err(h.head<2>() - uvi.head<2>() );
  if( (std::abs(rprj_err[0]) < rprj_threshold) && (std::abs(rprj_err[1]) < rprj_threshold) && std::abs(h_invd-measure_invd) < invd_threshold ){
    if(valid_depth)
      edge_filtered = new EdgeSE3PointXYZDepth(&param, uv_info, invd_info, v_mp, v_pose, uvi);
    else
      edge_filtered = new EdgeProjection(&param, uv_info, v_mp, v_pose, uvi.head<2>() );
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(delta);
    edge_filtered->setRobustKernel(rk);
  }
  else if (v_switch){
    if(valid_depth)
      edge_switchable = new EdgeSwSE3PointXYZDepth(&param, uv_info, invd_info, v_mp, v_pose, v_switch, uvi);
    else
      edge_switchable = new EdgeSwProjection(&param, uv_info, v_mp, v_pose, v_switch, uvi.head<2>() );;
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(delta);
    edge_switchable->setRobustKernel(rk);
  }
  return;
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
  const StereoCamera* scam = dynamic_cast<const StereoCamera*>(camera);
  const double focal = scam->GetK()(0,0);
  double uv_std = 1. / focal; // normalized standard deviation
  const double uv_info = 1./uv_std/uv_std;
  const auto Trl_ = scam->GetTrl();
  const float base_line = -Trl_.translation().x();
  const double invd_info = uv_info * base_line * base_line;  // focal legnth 1.인 normalized image ponint임을 고려.
  const double rprj_threshold = 10./focal; // rprj error threshold on normalized inmage plane
  const double delta = 5./focal;
  const double invd_threshold = rprj_threshold * base_line;
  const int n_iter = 10;
  const double dynamic_eval_duration = .5; // [sec]
  const double sec_final = curr_frame->GetSec();

  std::map<Jth, Frame*> frames = neighbor_keyframes;
  frames[curr_frame->GetId()] = curr_frame;

  Param param(camera);
  g2o::SparseOptimizer optimizer;
  // Dynamic block size due to 6dim SE3, 1dim prior vertex.
  typedef g2o::BlockSolverPL<-1,-1> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
#if 1
  g2o::OptimizationAlgorithm* solver =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#else
  g2o::OptimizationAlgorithm* solver =  new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
#endif
  optimizer.setAlgorithm(solver);

  std::map<Jth, std::pair<g2o::VertexSE3Expmap*,size_t> > v_poses; // with n_filtered_edges

  std::map<Mappoint*, g2o::VertexSBAPointXYZ*> v_mappoints;
  std::map<Instance*, size_t> mp_counts;
  std::map<Instance*, VertexSwitchLinear*>  v_switches;
  std::map<Instance*, EdgeSwitchPrior*>     prior_edges;
  std::map<Instance*, double>               n_pth_measurements;
  Frame*const kf_latest = neighbor_keyframes.rbegin()->second;
  const int kf_id_latest = kf_latest->GetKfId(qth);
  Frame*const kf_oldest = neighbor_keyframes.begin()->second;
  std::map<Instance*, std::list<g2o::OptimizableGraph::Edge*> > filtered_edges;

  for(Mappoint* mp : neighbor_mappoints){
    // n_kf는 mappoint의 keyframes가 아니라, LBA의 keyframes.
    size_t n_kf = mp->GetKeyframes(qth).size();
    if(n_kf < 2 && curr_frame->GetIndex(mp) < 0 ) // Projection Edge만 1개 생기는 경우 NAN이 발생하는것을 막기위해.
      continue;
    Instance* ins = GetIns(mp);
    //assert(ins->GetQth() == qth);
    g2o::VertexSBAPointXYZ* v_mp = new g2o::VertexSBAPointXYZ();
    v_mp->setId(optimizer.vertices().size() );
    v_mp->setEstimate(mp->GetXq(qth));
    v_mp->setMarginalized(true);
    optimizer.addVertex(v_mp);
    v_mappoints[mp] = v_mp;
    mp_counts[ins]++;
  }
  for(auto it : mp_counts){
    //if(it_pth.second < 5) // Tracked mappoints가 너무 적은 instance는 생략.
    //  continue;
    if(fixed_instances.count( it.first->GetId() ))
      continue;
    auto v_switch = new VertexSwitchLinear();
    v_switch->setId(optimizer.vertices().size() );
    optimizer.addVertex(v_switch);
    auto sw_prior_edge = new EdgeSwitchPrior();
    sw_prior_edge->setMeasurement(1.);
    v_switch->setEstimate(.4);
    sw_prior_edge->setVertex(0, v_switch);
    sw_prior_edge->setLevel(0);
    optimizer.addEdge(sw_prior_edge);
    v_switches[it.first] = v_switch;
    prior_edges[it.first] = sw_prior_edge;
  }

  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    Jth jth = it_frame.first;
    g2o::VertexSE3Expmap* v_pose = new g2o::VertexSE3Expmap();
    v_pose->setId(optimizer.vertices().size() );
    v_pose->setEstimate(frame->GetTcq(qth));
    v_pose->setMarginalized(false);
    optimizer.addVertex(v_pose);
    v_poses[jth].first = v_pose;
  }
  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    Jth jth = it_frame.first;
    g2o::VertexSE3Expmap* v_pose = v_poses.at(jth).first;
    size_t& n_filtered_edges = v_poses[jth].second;
    const auto& jth_mappoints = frame->GetMappoints();
    for(size_t n=0; n<jth_mappoints.size(); n++){
      Mappoint* mp = jth_mappoints[n];
      if(!mp)
        continue;
      if(!v_mappoints.count(mp)) // ins가 exclude 된 케이스라 neighbors에서 제외됬을 수 있다.
        continue;
      auto v_mp = v_mappoints.at(mp);
      Instance* ins = GetIns(mp);
      VertexSwitchLinear* v_switch = v_switches.count(ins) ? v_switches.at(ins) : nullptr;
      g2o::OptimizableGraph::Edge *edge_switchable, *edge_filtered;

      if(sec_final - frame->GetSec() > dynamic_eval_duration)
        v_switch = nullptr;
      GetEdges(v_mp, v_pose, v_switch, param, uv_info, invd_info, delta, rprj_threshold, invd_threshold,
               frame, n, edge_switchable, edge_filtered);
      if(v_switch){
        //n_pth_measurements[ins] += edge_switchable? edge_switchable->dimension() : 2;
        n_pth_measurements[ins] += 1;
      }
      if( edge_switchable ){
        edge_switchable->setLevel(0);
        optimizer.addEdge(edge_switchable);
      }
      if(edge_filtered){
        edge_filtered->setLevel(1);
        optimizer.addEdge(edge_filtered);
        n_filtered_edges++;
        if(v_switch)
          filtered_edges[ins].push_back(edge_filtered);
      }
#if 0
      // 너무 느리다. Marginalization 준비 안한탓에..
      for(Frame* kf : mp->GetKeyframes(qth) ) {
        Jth kf_jth = kf->GetId();
        if(frames.count(kf_jth))
          continue; // optmizable poses
        if(!v_poses.count(kf_jth) ) {
          g2o::VertexSE3Expmap* v_kf = new g2o::VertexSE3Expmap();
          v_kf->setId(optimizer.vertices().size() );
          v_kf->setEstimate(kf->GetTcq(qth));
          v_kf->setMarginalized(false);
          v_kf->setFixed(true); // out of neighbors
          optimizer.addVertex(v_kf);
          v_poses[kf_jth].first = v_kf;
        }
        g2o::VertexSE3Expmap* v_kf = v_poses.at(kf_jth).first;
        int kpt_idx = kf->GetIndex(mp);
        GetEdges(v_mp, v_kf, nullptr, param, uv_info, delta, invd_info, rprj_threshold, kf, kpt_idx,
                 edge_switchable, edge_filtered); // no edge_switchable because of nullptr v_switch
        if(edge_filtered)
          optimizer.addEdge(edge_filtered);
      }
#endif
    } // for int n < jth_mappoints.size();
  } // for auto it_frame : frames
  for(auto it_v_sw : prior_edges){
    int n = std::max<int>(n_pth_measurements[it_v_sw.first], 1);
    double info = ChiSquaredThreshold(.9, (double) n);
    //double info = 1e-4 * n; // 유도
    it_v_sw.second->SetInfomation(.1*info); // TODO switchable이 아닌 값을 직접비교하는 접근.
  }
  optimizer.setVerbose(false);

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

  for(auto it_mp : v_mappoints)
    it_mp.second->setFixed(false);
  for(auto it : frames){
    auto& pair = v_poses.at(it.first);
    pair.first->setFixed(pair.second < 20);
  }
  v_poses.at(kf_oldest->GetId()).first->setFixed(true);
  optimizer.initializeOptimization(1);  // Optimize filtered edges only
  optimizer.optimize(n_iter);

  for(auto it_poses : v_poses){
    g2o::VertexSE3Expmap* vertex = it_poses.second.first;
    if(vertex->fixed())
      continue;
    frames[it_poses.first]->SetTcq(qth, vertex->estimate());
  }
  for(auto it_v_mp : v_mappoints){
    g2o::VertexSBAPointXYZ* vertex = it_v_mp.second;
    if(vertex->fixed())
      continue;
    Mappoint* mp = it_v_mp.first;
    mp->SetXq(qth,vertex->estimate());
  }
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  std::map<Pth, float> switch_state; // posterior for inlier.
  for(auto it : v_switches)
    switch_state[it.first->GetId()] = it.second->estimate();
  return switch_state;
}

} // namespace NEW_SEG

