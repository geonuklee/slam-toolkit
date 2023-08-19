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


struct EdgeInfo {
  cv::KeyPoint kpt;
  Mappoint* mp;
  g2o::OptimizableGraph::Edge* edge;
  VertexSwitchLinear* vswitch;
  double uv_info;
  double invd_info;
};

std::map<Pth,float> Mapper::ComputeLBA(const Camera* camera,
                        Qth qth,
                        const std::set<Mappoint*>& neighbor_mappoints,
                        const std::map<Jth, Frame*>& neighbor_keyframes,
                        Frame* curr_frame,
                        Frame* prev_frame,
                        const cv::Mat& gradx,
                        const cv::Mat& grady,
                        const cv::Mat& valid_grad,
                        bool vis_verbose
                        ) {
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
  std::map<Pth, EdgeSwitchPrior*> prior_edges;

  for(Mappoint* mp : neighbor_mappoints){
    size_t n_kf = mp->GetKeyframes(qth).size();
    if(n_kf < 2){ // Projection Edge만 1개 생기는 경우 NAN이 발생하는것을 막기위해.
      if(curr_frame->GetIndex(mp) > -1 )
        n_kf++;
      if(n_kf < 2)
        continue;
    }
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
    //sw_vertex->setFixed(true);
  }

  int na = 0; int nb = 0;
  std::map<Pth, double> swedges_parameter;
  std::map<Jth, std::map<int, EdgeInfo> >infos;
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
      if(!v_mappoints.count(mp)) // ins가 exclude 된 케이스라 neighbors에서 제외됬을 수 있다.
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
      bool valid_depth_measurment =  z > MIN_NUM && vg;
#if 0
      double uv_info = 1.;
#else
      double uv_info = 0.; {
        double p = std::max(std::abs(uvi[0]), std::abs(uvi[1]) );
        //double p = std::abs(uvi[0]);
        double pmin = 0.3;
        double pmax = 0.4;
        uv_info = std::max(1e-10 , (pmax-p) /(pmax-pmin) );
        uv_info = std::min(1., uv_info);
        uv_info *= uv_info;
      }
#endif
#if 1
      double invd_info = 1e+2;
#else
      double invd_info = 0.;
      if(vg){
        double p = std::max(std::abs(gx), std::abs(gy) );
        //double p = std::abs(gx);
        double pmin = 1.;
        double pmax = 2.;
        invd_info = std::max(0.001 ,  (pmax-p) /(pmax-pmin) );
        invd_info = std::min(1., invd_info);
      }
      else{
        invd_info = 1e-5;
      }
#endif

      //const double invd_info = 1e+2;
      //const double invd_info =  vg ? std::max(1., .2 - std::abs(gx) ) : MIN_NUM;
      /* TODO False positive를 줄이기위해
      * [x] Huber kernel
      * [x] Fisher information
        * [x] uv info는 선형화. std::max(1e-1, tan(30deg) - normal uv) 이런거
      * [ ] Instance별 error 요인 분류. depth 오차가 안줄어드는건지 uv 오차가 안줄어드는건지.
        * ins 별 말고 point별로 표시하는것도 좋겠다. 
      */

      Instance* ins = mp->GetInstance();
      Pth pth = ins ? ins->GetId() : -1;
      g2o::OptimizableGraph::Edge* edge = nullptr;
      EdgeInfo info;
      info.kpt = kpt;
      info.mp = mp;
      info.vswitch = nullptr;
      info.uv_info = uv_info;
      info.invd_info = invd_info;
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
          info.vswitch = v_ins;
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
          info.vswitch = v_ins;
        } else {
          auto ptr = new EdgeProjection(&param, uv_info);
          ptr->setVertex(0, v_mp);
          ptr->setVertex(1, v_pose);
          ptr->setMeasurement( uvi.head<2>() );
          edge = ptr;
          optimizer.addEdge(edge);
        }
      }

      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(delta);
      edge->setRobustKernel(rk);

      info.edge = edge;
      const Ith& ith = mp->GetId();
      if(vis_verbose)
        infos[jth][n] = info;
    } // for mappoints
  } // for frames

  if(!v_poses.empty()){
    g2o::OptimizableGraph::Vertex* v_oldest = v_poses.begin()->second;
    v_oldest->setFixed(true);
  }

  for(auto it_v_sw : prior_edges){
    // Set Prior information
    double info = 1e-4 *  swedges_parameter.at(it_v_sw.first);
    //info = std::max(1e-3, info);
    //const double info = 1e-2; // Edge 숫자에 비례해야하나?
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
  if(vis_verbose){
#if 0
    cv::Mat vis_rgb = curr_frame->GetRgb().clone();
    cv::Mat vis_full = cv::Mat::zeros(vis_rgb.size(), CV_8UC1);
    cv::Mat dst_invdinfo = cv::Mat::zeros(vis_rgb.size(), CV_8UC3);
    cv::Mat dst_uvinfo = cv::Mat::zeros(vis_rgb.size(), CV_8UC3);
    std::map<Ith, double> chi2_sums;
    for(auto it_infos : infos){
      for(auto info : it_infos.second){
        double s = info.second.vswitch ? info.second.vswitch->estimate() : 1.;
        g2o::OptimizableGraph::Edge* edge = info.second.edge;
        chi2_sums[info.second.mp->GetId()] += edge->chi2() / s / s;
      }
    }

    for(auto info : infos[curr_frame->GetId()] ){
      double chi2 = chi2_sums.at(info.second.mp->GetId());
      if(chi2 < 1e-4)
        continue;
      std::stringstream ss;
      ss << std::setprecision(3) << chi2;
      std::string msg = ss.str();
      cv::putText(dst_uvinfo, msg, info.second.kpt.pt, cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,255,255) );
    }
    cv::addWeighted(vis_rgb, .4, dst_uvinfo, 1., 1., dst_uvinfo);
    cv::imshow("chi2 sum", dst_uvinfo);
#else
    for(auto it_infos : infos){
      Frame* frame = frames.at(it_infos.first);
      if(frame != curr_frame)
        continue;
      cv::Mat vis_rgb = frame->GetRgb().clone();
      cv::Mat vis_full = cv::Mat::zeros(vis_rgb.size(), CV_8UC1);
      cv::Mat dst_invdinfo = cv::Mat::zeros(vis_rgb.size(), CV_8UC3);
      cv::Mat dst_uvinfo = cv::Mat::zeros(vis_rgb.size(), CV_8UC3);
      for(auto info : it_infos.second){
        cv::circle(dst_invdinfo, info.second.kpt.pt, 2, CV_RGB(0,255,0), -1);

        {
          std::stringstream ss;
          ss << std::setprecision(2) << info.second.mp->GetXq(0)[2] << std::endl; // TODO sprintf로 정확한 자리수 표시.
          std::string msg = ss.str();
          cv::putText(dst_invdinfo, msg, info.second.kpt.pt, cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,255,255) );
        }


        double s = info.second.vswitch ? info.second.vswitch->estimate() : 1.;
        g2o::OptimizableGraph::Edge* edge = info.second.edge;
        //Eigen::VectorXd err(edge->errorData(), edge->dimension());
        int dim = edge->dimension();
        g2o::MatrixX::MapType information(edge->informationData(), dim, dim);
        Eigen::VectorXd err(dim);
        for(int i=0; i<dim; i++){
          const double e = edge->errorData()[i]/s;
          err(i) = std::abs(e) * std::sqrt(information(i,i));
        }
        double uv_err = err.head<2>().norm();
        if(uv_err > 1e-2){
          std::stringstream ss;
          //ss << std::setprecision(3) << information(0,0);
          ss << std::setprecision(3) << uv_err;
          std::string msg = ss.str();
          cv::putText(dst_uvinfo, msg, info.second.kpt.pt, cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,255,255) );
        }
        if(dim < 3)
          continue;
        double invd_err = std::abs( err(2) );
        /*if(invd_err > 1e-2){
          std::stringstream ss;
          ss << std::setprecision(1) << info.second.invd_info;
          std::string msg = ss.str();
          cv::putText(dst_invdinfo, msg, info.second.kpt.pt, cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,255,255) );
        }*/
      }
      cv::addWeighted(vis_rgb, .4, dst_invdinfo, 1., 1., dst_invdinfo);
      //cv::imshow("invd info"+std::to_string(it_infos.first), dst_invdinfo);
      cv::imshow("invd info", dst_invdinfo);
      cv::addWeighted(vis_rgb, .4, dst_uvinfo, 1., 1., dst_uvinfo);
      //cv::imshow("uv info"+std::to_string(it_infos.first), dst_uvinfo);
      //cv::imshow("uv info", dst_uvinfo);
      static bool stop = false;
      char c = cv::waitKey(stop?0:1);
      if(c == 'q')
        exit(1);
      else if (c == 's')
        stop = !stop;

    }
#endif
  }

  {
    const auto& T0q = prev_frame->GetTcq(0);
    const auto& T1q = curr_frame->GetTcq(0);
    const auto T01 = T0q * T1q.inverse();
    std::cout << "dT = " << T01.translation().transpose() << std::endl;
    //if(T01.translation().z() < 0.)
    //  throw -2;
    //if(T01.translation().norm() < 1e-2)
    //  throw -1;  // 버그 감지용.정지장면이 있는 seq에서는 오인식함.
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
