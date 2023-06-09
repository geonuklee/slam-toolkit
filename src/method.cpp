#include "method.h"
#include "frame.h"
#include "mappoint.h"
#include "camera.h"
#include "orb_extractor.h"
#include "optimizer.h"

#include <g2o/types/sba/types_six_dof_expmap.h>

#include <g2o/core/block_solver.h>

#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>

IndirectStereoMethod::IndirectStereoMethod(const std::vector<float>& inv_scales_sigma2)
  : inv_scales_sigma2_(inv_scales_sigma2)
{
}

void IndirectStereoMethod::SetSolver(OptimizationMethod optimization_method, g2o::SparseOptimizer& optimizer) {
  typedef g2o::BlockSolverPL<6,3> BlockSolver;

  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver
    = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();

  g2o::OptimizationAlgorithm* solver = nullptr;
  switch(optimization_method){
    case LM:
      solver =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
      break;
    case GN:
      solver =  new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
      break;
  }
  optimizer.setAlgorithm(solver);

  return;
}

g2o::OptimizableGraph::Edge* IndirectStereoMethod::CreateAnchorMeasurementEdge(const Frame* ref_frame, Mappoint* mappoint){
  auto edge = new g2o::EdgeStereoSE3ProjectXYZ();
  Eigen::Vector3d obs;
  int kpt_idx = ref_frame->GetIndex(mappoint);
  obs.head<2>() = ref_frame->GetNormalizedPoint(kpt_idx);
  const StereoFrame* stereo_frame = static_cast<const StereoFrame*>(ref_frame);
  cv::KeyPoint rkpt =stereo_frame->GetRightKeypoint(mappoint);

  const Camera* cam = ref_frame->GetCamera();
  const double& fx = cam->GetK()(0,0);
  const double& cx = cam->GetK()(0,2);
  obs[2] = (rkpt.pt.x-cx)/fx;
  edge->setMeasurement(obs);
  return edge;
}

g2o::OptimizableGraph::Edge* IndirectStereoMethod::CreateMeasurementEdge(const Frame* frame, Mappoint* mappoint, bool set_measurement_from_frame){
  auto edge = new g2o::EdgeSE3ProjectXYZ();

  if(set_measurement_from_frame){
    // Called from LBA, GBA
    const cv::KeyPoint& kpt = frame->GetKeypoint(mappoint);
    int kpt_idx = frame->GetIndex(mappoint);
    Eigen::Vector2d obs = frame->GetNormalizedPoint(kpt_idx);
    edge->setMeasurement(obs);
    const float invSigma2 = inv_scales_sigma2_.at(kpt.octave);
    edge->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    const float deltaMono = sqrt(5.991); // TODO Why?
    rk->setDelta(0.0001*deltaMono);
    edge->setRobustKernel(rk);
    edge->fx = 1.;
    edge->fy = 1.;
    edge->cx = 0.;
    edge->cy = 0.;
  }
  return edge;
}

g2o::OptimizableGraph::Vertex* IndirectStereoMethod::CreatePoseVertex(g2o::SparseOptimizer& optimizer, Frame* frame){
  auto vertex = new g2o::VertexSE3Expmap();
  vertex->setId(optimizer.vertices().size() );
  optimizer.addVertex(vertex);
  if(frame)
    vertex->setEstimate(frame->GetTcw());
  return vertex;
}

g2o::OptimizableGraph::Vertex* IndirectStereoMethod::CreateStructureVertex(g2o::SparseOptimizer& optimizer, Mappoint* mappoint){
  auto vertex = new g2o::VertexSBAPointXYZ();
  vertex->setId(optimizer.vertices().size() );
  vertex->setMarginalized(true);
  vertex->setEstimate(mappoint->GetXw());
  optimizer.addVertex(vertex);
  return vertex;
}

void IndirectStereoMethod::RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, Frame* frame){
  auto ptr = static_cast<const g2o::VertexSE3Expmap*>(vertex);
  frame->SetTcw(ptr->estimate());
  return;
}

void IndirectStereoMethod::RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, void* estimation){
  auto vertex_ptr = static_cast<const g2o::VertexSE3Expmap*>(vertex);
  g2o::SE3Quat* pose_ptr = static_cast<g2o::SE3Quat*>(estimation);
  *pose_ptr = vertex_ptr->estimate();
}

void IndirectStereoMethod::SetPose(const void* predict, g2o::OptimizableGraph::Vertex* vertex) {
  auto pose_ptr = static_cast<const g2o::SE3Quat*>(predict);
  auto vertex_ptr = static_cast<g2o::VertexSE3Expmap*>(vertex);
  vertex_ptr->setEstimate(*pose_ptr);
}

void IndirectStereoMethod::RetriveStructure(const g2o::OptimizableGraph::Vertex* vertex, Mappoint* mappoint){
  Frame* rkf = mappoint->GetRefFrame();
  const g2o::SE3Quat& Tcw = rkf->GetTcw();
  auto ptr = static_cast<const g2o::VertexSBAPointXYZ*>(vertex);
  const Eigen::Vector3d& Xw = ptr->estimate();
  Eigen::Vector3d Xc = Tcw*Xw;
  mappoint->SetInvD(1./Xc[2]);
  return;
}

DirectStereoMethod::DirectStereoMethod()
  : pyramid_(new DirectPyramid),
  huber_delta_( std::sqrt(std::pow(50., 2.)*EdgeProjectBrightenXYZ::Dimension) )
{
}

std::shared_ptr<DirectPyramid> DirectStereoMethod::GeyPyramid( ) const {
  return pyramid_;
}

void DirectStereoMethod::SetSolver(OptimizationMethod optimization_method, g2o::SparseOptimizer& optimizer){
  typedef g2o::BlockSolverPL<8,3> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver
    = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();

  g2o::OptimizationAlgorithm* solver = nullptr;
  switch(optimization_method){
    case LM:
      solver =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
      break;
    case GN:
      solver =  new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolver>(std::move(linear_solver)));
      break;
  }
  optimizer.setAlgorithm(solver);
  return;
}

g2o::OptimizableGraph::Edge* DirectStereoMethod::CreateAnchorMeasurementEdge(const Frame* ref_frame, Mappoint* mappoint){
  auto edge = new g2o::EdgeStereoSE3ProjectXYZ();
  Eigen::Vector3d obs;
  int kpt_idx = ref_frame->GetIndex(mappoint);
  obs.head<2>() = ref_frame->GetNormalizedPoint(kpt_idx);
  const StereoFrame* stereo_frame = static_cast<const StereoFrame*>(ref_frame);
  cv::KeyPoint rkpt =stereo_frame->GetRightKeypoint(mappoint);
  const Camera* cam = ref_frame->GetCamera();
  const double& fx = cam->GetK()(0,0);
  const double& cx = cam->GetK()(0,2);
  obs[2] = (rkpt.pt.x-cx)/fx;
  edge->setMeasurement(obs);
  return edge;
}

g2o::OptimizableGraph::Edge* DirectStereoMethod::CreateMeasurementEdge(const Frame* frame, Mappoint* mappoint, bool set_measurement_from_frame){
  auto edge = new EdgeProjectBrightenXYZ(pyramid_.get(), frame, mappoint);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
  rk->setDelta( huber_delta_ );
  edge->setRobustKernel(rk);
  //std::cout << "Create edge" << std::endl;
  return edge;
}

g2o::OptimizableGraph::Vertex* DirectStereoMethod::CreatePoseVertex(g2o::SparseOptimizer& optimizer, Frame* frame) {
  auto v_pose = new VertexBrightenSE3();
  v_pose->setId(optimizer.vertices().size() );
  optimizer.addVertex(v_pose);
  if(frame)
    v_pose->setEstimate(frame->GetBrightenPose());
  auto prior = new EdgeBrightenessPrior(10., 10.);
  prior->setVertex(0, v_pose);
  optimizer.addEdge(prior);
  //std::cout << "Create Pose" << std::endl;
  return v_pose;
}

g2o::OptimizableGraph::Vertex* DirectStereoMethod::CreateStructureVertex(g2o::SparseOptimizer& optimizer, Mappoint* mappoint) {
  auto vertex = new g2o::VertexSBAPointXYZ();
  vertex->setMarginalized(true);
  vertex->setEstimate(mappoint->GetXw());
  vertex->setId(optimizer.vertices().size() );
  optimizer.addVertex(vertex);
  return vertex;
}

void DirectStereoMethod::RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, Frame* frame){
  auto vertex_ptr = static_cast<const VertexBrightenSE3*>(vertex);
  frame->SetTcw(vertex_ptr->estimate().Tcw_);
  frame->SetBrightness(vertex_ptr->estimate().brightness_);
  return;
}

void DirectStereoMethod::RetrivePose(const g2o::OptimizableGraph::Vertex* vertex, void* estimation){
  auto vertex_ptr = static_cast<const VertexBrightenSE3*>(vertex);
  auto pose_ptr   = static_cast<BrightenSE3*>(estimation);
  *pose_ptr = vertex_ptr->estimate();
  return;
}

void DirectStereoMethod::SetPose(const void* predict, g2o::OptimizableGraph::Vertex* vertex) {
  auto pose_ptr = static_cast<const BrightenSE3*>(predict);
  auto vertex_ptr = static_cast<VertexBrightenSE3*>(vertex);
  vertex_ptr->setEstimate(*pose_ptr);
}

void DirectStereoMethod::RetriveStructure(const g2o::OptimizableGraph::Vertex* vertex, Mappoint* mappoint){
  Frame* rkf = mappoint->GetRefFrame();
  const g2o::SE3Quat& Tcw = rkf->GetTcw();
  auto ptr = static_cast<const g2o::VertexSBAPointXYZ*>(vertex);
  const Eigen::Vector3d& Xw = ptr->estimate();
  Eigen::Vector3d Xc = Tcw*Xw;
  mappoint->SetInvD(1./Xc[2]);
}

