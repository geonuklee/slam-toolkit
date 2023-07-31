#include "optimizer.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace seg {
Mapper::Mapper() {
}

Mapper::~Mapper() {
}

g2o::OptimizableGraph::Vertex* CreatePoseVertex(g2o::SparseOptimizer& optimizer,
                                                Frame* frame,
                                                const g2o::SE3Quat& Tcq) {
  auto vertex = new g2o::VertexSE3Expmap();
  vertex->setId(optimizer.vertices().size() );
  optimizer.addVertex(vertex);
  if(frame)
    vertex->setEstimate(Tcq);
  return vertex;
}

g2o::OptimizableGraph::Vertex* CreateStructureVertex(g2o::SparseOptimizer& optimizer,
                                                     Mappoint* mpt){
  auto vertex = new g2o::VertexSBAPointXYZ();
  vertex->setId(optimizer.vertices().size() );
  vertex->setMarginalized(true);
  /*
    Xr 장점 : Tcq에 독립된 변수

  */
  Eigen::Vector3d Xr = mpt->GetXr();
  /*
  TODO Xq를 구해서 맵핑해야한다.
  -> 그럴바에 그냥 argument를 Xq가 아니라 Xr로 정의해버릴까?
  */
  vertex->setEstimate(Xr);
  optimizer.addVertex(vertex);
  return vertex;
}


void Mapper::ComputeLBA(const std::set<Mappoint*>& mappoints,
                const std::map<Jth, Frame*>& frames,
                const Frame* curr_frame,
                EigenMap<Jth, g2o::SE3Quat> & kf_Tcqs,
                g2o::SE3Quat& Tcq
               ) {
  /*
  argmin sig(s(p,q)^) [ Xc(j) - Tcq(j)^ Xq(i∈p)^ ]
  Xq(i∈p)^ : q th group coordinate에서 본 'i'th mappoint의 좌표.
  -> 일시적으로, 하나의 'i'th mappoint가 여러개의 3D position을 가질 수 있다.
  -> Instance 의 모양이 group마다 바뀐다는 모순이 생길순 있지만, 이런 경우가 적으므로,
  사소한 연결때문에 group을 넘나드는 SE(3)사이의 covariance를 늘릴 필요는 없다.( solver dimmension)
  SE(3) 의 dimmension을 줄일려고( n(G) dim, 1iter -> 1dim, n(G) iter ) 
  binary로 group member 가지치기 들어가기.
  */

  g2o::SparseOptimizer optimizer;
  std::map<Jth, g2o::OptimizableGraph::Vertex*> v_poses;
  std::map<Ith, g2o::OptimizableGraph::Vertex*> v_mappoints;
  std::map<g2o::OptimizableGraph::Edge*, std::pair<Frame*,Mappoint*> > measurment_edges;

  for(auto it_frame : frames)
    v_poses[it_frame.first] = CreatePoseVertex(optimizer,
                                              it_frame.second,
                                              kf_Tcqs.at(it_frame.first) );
  if(!v_poses.empty()){
    g2o::OptimizableGraph::Vertex* v_oldest = v_poses.begin()->second;
    v_oldest->setFixed(true);
  }


























  return;
}

} // namespace seg
