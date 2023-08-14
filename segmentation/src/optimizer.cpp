#include "optimizer.h"
#include "Eigen/src/Core/Matrix.h"
#include "g2o_types.h"
#include "segslam.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/imgproc.hpp>


namespace seg {

Mapper::Mapper() {
}

Mapper::~Mapper() {
}

g2o::OptimizableGraph::Vertex* Mapper::CreatePoseVertex(const Qth& qth,
                                                        g2o::SparseOptimizer& optimizer,
                                                        Frame* frame) {
  auto vertex = new g2o::VertexSE3Expmap();
  vertex->setId(optimizer.vertices().size() );
  optimizer.addVertex(vertex);
  const auto& Tcq = frame->GetTcq(qth);
  vertex->setEstimate(Tcq);
  return vertex;
}

g2o::OptimizableGraph::Vertex* Mapper::CreateStructureVertex(Qth qth,
                                                     g2o::SparseOptimizer& optimizer,
                                                     Mappoint* mp){
  auto vertex = new g2o::VertexSBAPointXYZ();
  vertex->setId(optimizer.vertices().size() );
  const Eigen::Vector3d& Xq = mp->GetXq(qth);
  vertex->setEstimate(Xq);
  optimizer.addVertex(vertex);
  return vertex;
}


std::map<Pth,float> Mapper::ComputeLBA(const Camera* camera,
                        Qth qth,
                        const std::set<Mappoint*>& neighbor_mappoints,
                        const std::map<Jth, Frame*>& neighbor_keyframes,
                        Frame* curr_frame,
                        const cv::Mat& gradx,
                        const cv::Mat& grady,
                        const cv::Mat& valid_grad,
                        bool vis_verbose
                        ) {
  cv::Mat vis_rgb, dst, vis_full;
  if(vis_verbose){
    vis_rgb = curr_frame->GetRgb().clone();
    dst = cv::Mat::zeros(vis_rgb.size(), CV_8UC3);
    vis_full = cv::Mat::zeros(dst.size(), CV_8UC1);
  }
  const double focal = camera->GetK()(0,0);
  const double delta = 10./focal;
  // info : inverted covariance

  const int n_iter = 20;
  std::map<Jth, Frame*> frames = neighbor_keyframes;
  frames[curr_frame->GetId()] = curr_frame;
  /*
  argmin sig(s(p,q)^) [ Xc(j) - Tcq(j)^ Xq(i∈p)^ ]
  Xq(i∈p)^ : q th group coordinate에서 본 'i'th mappoint의 좌표.
  -> 일시적으로, 하나의 'i'th mappoint가 여러개의 3D position을 가질 수 있다.
  -> Instance 의 모양이 group마다 바뀐다는 모순이 생길순 있지만, 이런 경우가 적으므로,
  사소한 연결때문에 group을 넘나드는 SE(3)사이의 covariance를 늘릴 필요는 없다.( solver dimmension)
  SE(3) 의 dimmension을 줄일려고( n(G) dim, 1iter -> 1dim, n(G) iter ) 
  binary로 group member 가지치기 들어가기.
  */
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

  std::map<Jth, g2o::OptimizableGraph::Vertex*> v_poses;
  std::map<Mappoint*, g2o::OptimizableGraph::Vertex*> v_mappoints;
  std::map<Pth, VertexSwitchLinear*>  v_instances;
  std::map<Pth, size_t>               mp_counts;
  std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> > measurment_edges;
  std::map<Pth, EdgeSwitchPrior*> prior_edges;

  for(Mappoint* mp : neighbor_mappoints){
    Instance* ins = mp->GetInstance();
    if(ins && ins->GetId() > 0 ){
      mp_counts[ins->GetId()]++;
    }
    // Future : ins 수집해서 prior_vertex
    g2o::OptimizableGraph::Vertex* v_mp = CreateStructureVertex(qth, optimizer, mp);
    v_mp->setMarginalized(true);
    v_mappoints[mp]  = v_mp;
  }
  for(auto it_pth : mp_counts){
    if(it_pth.second < 5)
      continue;
    auto sw_vertex = new VertexSwitchLinear();
    sw_vertex->setId(optimizer.vertices().size() );
    sw_vertex->setEstimate(1.);
    optimizer.addVertex(sw_vertex);
    auto sw_prior_edge = new EdgeSwitchPrior();
    sw_prior_edge->setMeasurement(1.);
    sw_prior_edge->setVertex(0, sw_vertex);
    optimizer.addEdge(sw_prior_edge);
    v_instances[it_pth.first] = sw_vertex;
    prior_edges[it_pth.first] = sw_prior_edge;
  }

  int na = 0; int nb = 0;
  std::map<Pth, double> swedges_parameter;
  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    Jth jth = it_frame.first;
    g2o::OptimizableGraph::Vertex* v_pose = CreatePoseVertex(qth, optimizer, frame);
    v_pose->setFixed(false);
    v_poses[jth] = v_pose;
    const auto& jth_mappoints = frame->GetMappoints();
    for(size_t n=0; n<jth_mappoints.size(); n++){
      Mappoint* mp = jth_mappoints[n];
      if(!mp)
        continue;
      if(!v_mappoints.count(mp))
        continue;
      auto v_mp = v_mappoints.at(mp);
      const cv::KeyPoint& kpt = frame->GetKeypoint(n);
      Eigen::Vector3d uvi = frame->GetNormalizedPoint(n);
      const float& z = frame->GetDepth(n);
      if(z >  MIN_NUM)
        uvi[2] = 1. / z;
      else // If invalid depth = finite depth
        uvi[2] = MIN_NUM; // invd close too zero

      const float& vg = valid_grad.at<uchar>(kpt.pt);
      const float& gx = gradx.at<float>(kpt.pt);
      const float& gy = grady.at<float>(kpt.pt);
      bool valid_depth_measurment =  z > MIN_NUM && vg && std::abs(gx) < 2. ;
#if 1
      //double cinfo = 1.;
      double cinfo = 1e-2/std::max(MIN_NUM,uvi[0]);
      const double uv_info = 1.;
      const double invd_info = 100.;
      // TODO inverse depth의 fisher information
#endif

      Instance* ins = mp->GetInstance();
      Pth pth = ins ? ins->GetId() : -1;
      g2o::OptimizableGraph::Edge* edge = nullptr;
      if( valid_depth_measurment ) {
        if(v_instances.count(pth) ){
          VertexSwitchLinear* v_ins = v_instances.at(pth);
          auto ptr = new EdgeSwSE3PointXYZDepth(&param, uv_info, invd_info);
          ptr->setVertex(0,v_mp);
          ptr->setVertex(1,v_pose);
          ptr->setVertex(2,v_ins);
          ptr->setMeasurement( uvi );
          edge = ptr;
          swedges_parameter[pth] += 1.;
          optimizer.addEdge(edge);
        }
        else {
          auto ptr = new EdgeSE3PointXYZDepth(&param, uv_info, invd_info);
          ptr->setVertex(0, v_mp);
          ptr->setVertex(1, v_pose);
          ptr->setMeasurement( uvi );
          edge = ptr;
          optimizer.addEdge(edge);
        }
      }
      else {
        if(v_instances.count(pth) ){
          VertexSwitchLinear* v_ins = v_instances.at(pth);
          auto ptr = new EdgeSwProjection(&param, uv_info);
          ptr->setVertex(0,v_mp);
          ptr->setVertex(1,v_pose);
          ptr->setVertex(2,v_ins);
          ptr->setMeasurement( uvi.head<2>() );
          edge = ptr;
          swedges_parameter[pth] += 1.;
          optimizer.addEdge(edge);
        } else {
          auto ptr = new EdgeProjection(&param, uv_info);
          ptr->setVertex(0, v_mp);
          ptr->setVertex(1, v_pose);
          ptr->setMeasurement( uvi.head<2>() );
          edge = ptr;
          optimizer.addEdge(edge);
        }
      }
      const Ith& ith = mp->GetId();
      if(!dst.empty() && jth == curr_frame->GetId()){
        if(vis_full.at<uchar>(kpt.pt) > 100) // 글자가 너무 많은경우.
          continue;
        cv::circle(vis_full, kpt.pt, 10, 255, -1); // prevent overlaped texts
        std::stringstream ss;
        ss << std::hex << ith; // Convert to hexadecimal
        ss << std::setprecision(3) << cinfo;
        std::string msg = ss.str();
        cv::putText(dst, msg, kpt.pt, cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,255,255) );
        cv::circle(dst, kpt.pt, 2, CV_RGB(0,255,0),-1);
      }
    }
  }

  if(!v_poses.empty()){
    g2o::OptimizableGraph::Vertex* v_oldest = v_poses.begin()->second;
    v_oldest->setFixed(true);
  }

  for(auto it_v_sw : prior_edges){
    // Set Prior information
    //const double info = 1e-8* std::pow(swedges_parameter.at(it_v_sw.first),2);
    const double info = 1e-3;
    it_v_sw.second->SetInfomation(info);
  }

  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);

  // Retrieve
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  const Jth curr_jth = curr_frame->GetId();
  int n_pose = 0;
  int N_pose = v_poses.size();
  bool txt_verbose = true;
  if(txt_verbose)
    std::cout << "At Q#" << qth << " curr F#"<< curr_frame->GetId() <<"--------------------" << std::endl;
  for(auto it_poses : v_poses){
    g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(it_poses.second);
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
    g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(it_v_mpt.second);
    if(vertex->fixed())
      continue;
    n_structure++;
    Mappoint* mpt = it_v_mpt.first;
    mpt->SetXq(qth,vertex->estimate());
  }
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  if(!dst.empty()){
    cv::addWeighted(vis_rgb, .4, dst, 1., 1., dst);
    cv::imshow("optimizer", dst);
  }

  if(txt_verbose){
    printf("Update Pose(%d/%d), Structure(%d/%d), SwEdges(%d,%d)\n",
           n_pose, N_pose, n_structure, N_structure, na, na+nb);
  }

  std::map<Pth, float> switch_state; // posterior for inlier.
  for(auto it : v_instances)
    switch_state[it.first] = it.second->estimate();
  return switch_state;
}

} // namespace seg
