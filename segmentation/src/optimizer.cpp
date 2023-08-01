#include "optimizer.h"
#include "segslam.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/se3quat.h>

#define CHECK_NAN

namespace seg {

const double MIN_NUM = 1e-5;

Mapper::Mapper() {
}

Mapper::~Mapper() {
}

g2o::OptimizableGraph::Vertex* Mapper::CreatePoseVertex(g2o::SparseOptimizer& optimizer,
                                                Frame* frame,
                                                const g2o::SE3Quat& Tcq) {
  auto vertex = new g2o::VertexSE3Expmap();
  vertex->setId(optimizer.vertices().size() );
  optimizer.addVertex(vertex);
  if(frame)
    vertex->setEstimate(Tcq);
  return vertex;
}

g2o::OptimizableGraph::Vertex* Mapper::CreateStructureVertex(Qth qth,
                                                     g2o::SparseOptimizer& optimizer,
                                                     Mappoint* mpt){
  auto vertex = new g2o::VertexSBAPointXYZ();
  vertex->setId(optimizer.vertices().size() );
  const Eigen::Vector3d& Xq = mpt->GetXq(qth);
  vertex->setEstimate(Xq);
  optimizer.addVertex(vertex);
  return vertex;
}

/*
  1) slam3d/edge_se3_pointxyz_depth.h, g2o::EdgeSE3PointXYZDepth,
  2) sba/types_six_dof_expmap.h, g2o::EdgeStereoSE3ProjectXYZ 를 참고한, Edge함수.
  Error정의를 기존 [normalized uv, depth] 대신
  nuv, invd depth
  Structure param은 그대로 XYZ
*/
class EdgeSE3PointXYZDepth
  : public g2o::BaseBinaryEdge<3, g2o::Vector3, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3PointXYZDepth(const Param* param)
    :param_(param),
    g2o::BaseBinaryEdge<3, g2o::Vector3, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
    {
      information().setIdentity(3,3);
      //information()(0,0) = information()(1,1) = 1.; // covariance for normalized u,v
      information()(2,2) = 1e-2; // inverted covariance for inverse depth
    }
    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }
    virtual int measurementDimension() const {return 3;}
    void computeError() {
      const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
      const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
      g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
      g2o::Vector3 h(vj->estimate().map(vi->estimate()));
      auto invd = std::max(MIN_NUM, 1./h[2] );
      h[0] *= invd; // Xc/Zc
      h[1] *= invd; // Yc/Zc
      h[2] = invd;
      _error = h-obs;
#ifdef CHECK_NAN
      if(_error.hasNaN()){
        std::cout << "vi = " << vi->estimate().transpose() << std::endl;
        std::cout << "vj = " << vj->estimate().to_homogeneous_matrix() << std::endl;
        throw -1;
      }
      return;
    }
#endif
    virtual void linearizeOplus() {
      const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
      const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
      const g2o::SE3Quat& Tcw = vj->estimate();
      const g2o::Vector3& Xw  = vi->estimate();
      g2o::Vector3 Xc = Tcw.map(Xw);
      Xc[2] = std::max(MIN_NUM, Xc[2]); // Ignore negative depth
      double inv_Zc = 1 / Xc[2];
      double inv_Zc2 = inv_Zc* inv_Zc;
      Eigen::Matrix<double,3,3> dhdXc;
      dhdXc << inv_Zc, 0., -Xc[0]*inv_Zc2,
               0., inv_Zc, -Xc[1]*inv_Zc2,
               0., 0.,     -inv_Zc2;

      Eigen::Matrix<double,3,9> jacobian;
      jacobian.block<3,3>(0,0) = Tcw.rotation().toRotationMatrix(); // jac Xi
      jacobian.block<3,3>(0,3) = -g2o::skew(Xc); // jac Xj for omega in g2o::SE3Quat::exp [omega; upsilon]
      jacobian.block<3,3>(0,6).setIdentity();    // jac Xj for upsilon
      jacobian = dhdXc * jacobian;
#ifdef CHECK_NAN
      if(jacobian.hasNaN())
        throw -1;
      _jacobianOplusXi = jacobian.block<3,3>(0,0);
      _jacobianOplusXj = jacobian.block<3,6>(0,3);
#endif
      return;
    }

    /*
    virtual bool setMeasurementData(const number_t* d) {
      Eigen::Map<const g2o::Vector3> v(d);
      _measurement = v;
      return true;
    }
    virtual bool getMeasurementData(number_t* d) const {
      Eigen::Map<g2o::Vector3> v(d);
      v=_measurement;
      return true;
    }
    */
    
private:
    const Param*const param_;
    Eigen::Matrix<number_t,3,9,Eigen::ColMajor> J; // jacobian before projection
};

void Mapper::ComputeLBA(const Camera* camera,
                        Qth qth,
                        const std::set<Mappoint*>& neighbor_mappoints,
                        const std::map<Jth, Frame*>& neighbor_keyframes,
                        Frame* curr_frame,
                        EigenMap<Jth, g2o::SE3Quat> & _kf_Tcqs,
                        g2o::SE3Quat& curr_Tcq,
                        bool verbose
                       ) {
  const int n_iter = 10;
  std::map<Jth, Frame*> frames = neighbor_keyframes;
  if(curr_frame)
    frames[curr_frame->GetId()] = curr_frame;
  EigenMap<Jth, g2o::SE3Quat> Tcqs;
  for(auto it_nkf : neighbor_keyframes) // Copy Tcq for nkf only
    Tcqs[it_nkf.first] = _kf_Tcqs.at(it_nkf.first);
  if(curr_frame)
    Tcqs[curr_frame->GetId()] = curr_Tcq;

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
  typedef g2o::BlockSolverPL<6,3> BlockSolver;
  std::unique_ptr<BlockSolver::LinearSolverType> linear_solver
    = g2o::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
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
  std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> > measurment_edges;
  for(Mappoint* mpt : neighbor_mappoints){
    // Future : ins 수집해서 prior_vertex
    g2o::OptimizableGraph::Vertex* v_mpt = CreateStructureVertex(qth, optimizer, mpt);
    v_mpt->setMarginalized(true);
    v_mpt->setFixed(true);
    v_mappoints[mpt]  = v_mpt;
  }

  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    Jth jth = it_frame.first;
    g2o::OptimizableGraph::Vertex* v_pose = CreatePoseVertex(optimizer,
                                                             it_frame.second,
                                                             Tcqs.at(jth) );
    v_pose->setFixed(false);
    v_poses[jth] = v_pose;
    const auto& jth_mappoints = frame->GetMappoints();
    for(size_t n=0; n<jth_mappoints.size(); n++){
      Mappoint* mpt = jth_mappoints[n];
      if(!mpt)
        continue;
      if(!v_mappoints.count(mpt))
        continue;
      auto v_mpt = v_mappoints.at(mpt);
      const cv::KeyPoint& kpt = frame->GetKeypoint(n);
      Eigen::Vector3d uvi = frame->GetNormalizedPoint(n);
      const float& z = frame->GetDepth(n);
      if(z >  MIN_NUM)
        uvi[2] = 1. / z;
      else // If invalid depth = finite depth
        uvi[2] = MIN_NUM; // invd close too zero
      auto edge = new EdgeSE3PointXYZDepth(&param);
      edge->setMeasurement( uvi );

      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      const float deltaMono = sqrt(5.991); // TODO Why? ORB_SLAM2
      rk->setDelta(deltaMono);
      edge->setRobustKernel(rk);

      edge->setVertex(0, v_mpt);
      edge->setVertex(1, v_pose);
      optimizer.addEdge(edge);
    }
  }
  if(!v_poses.empty()){
    g2o::OptimizableGraph::Vertex* v_oldest = v_poses.begin()->second;
    v_oldest->setFixed(true);
  }

  /*
  TODO
  * [ ] Measumenet Edge
    - 먼저 switch edge 넣기전에 depth sensor를 고려한 measreument edge부터 넣어서
      최소한 world tracking은 되는지부터 확인 먼저
  * [ ] Instance Switch edge
  */
  optimizer.setVerbose(verbose);
  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);

  // Retrieve
   //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  const Jth curr_jth = curr_frame->GetId();
  int n_pose = 0;
  int N_pose = v_poses.size();
  if(verbose)
    std::cout << "At Q#" << qth << " curr F#"<< curr_frame->GetId() <<"--------------------" << std::endl;
  for(auto it_poses : v_poses){
    g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(it_poses.second);
    if(verbose)
      std::cout << "F#" << it_poses.first << ", " << vertex->estimate().inverse().translation().transpose() << std::endl;
    if(vertex->fixed())
      continue;
    if(it_poses.first==curr_jth)
      curr_Tcq = vertex->estimate();
    else
      _kf_Tcqs[it_poses.first] = vertex->estimate();
    n_pose++;
  }
  if(verbose)
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
  if(verbose)
    printf("Update Pose(%d/%d), Structure(%d/%d)", n_pose, N_pose, n_structure, N_structure);
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  return;
}

} // namespace seg
