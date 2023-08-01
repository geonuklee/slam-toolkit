#include "g2o_types.h"
#include <g2o/config.h>
#define CHECK_NAN

namespace seg {

const double MIN_NUM = 1e-10;

EdgeSE3PointXYZDepth::EdgeSE3PointXYZDepth(const Param* param)
  :param_(param),
  g2o::BaseBinaryEdge<3, g2o::Vector3, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
{
  information().setIdentity(3,3);
  information()(2,2) = 1e-4; // inverted covariance for inverse depth
}

void EdgeSE3PointXYZDepth::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 h(vj->estimate().map(vi->estimate()));
  auto invd = std::max(MIN_NUM, 1./std::max(MIN_NUM,h[2]) );
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
#endif
  return;
}

void EdgeSE3PointXYZDepth::linearizeOplus() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const g2o::SE3Quat& Tcw = vj->estimate();
  const g2o::Vector3& Xw  = vi->estimate();
  g2o::Vector3 Xc = Tcw.map(Xw);
  auto invd = std::max(MIN_NUM, 1./std::max(MIN_NUM,Xc[2]) );
  const double inv_Zc2 = invd* invd;
  Eigen::Matrix<double,3,3> dhdXc;
  dhdXc << invd, 0., -Xc[0]*inv_Zc2,
        0., invd, -Xc[1]*inv_Zc2,
        0., 0.,     -inv_Zc2;

  Eigen::Matrix<double,3,9> jacobian;
  jacobian.block<3,3>(0,0) = Tcw.rotation().toRotationMatrix(); // jac Xi
  jacobian.block<3,3>(0,3) = -g2o::skew(Xc); // jac Xj for omega in g2o::SE3Quat::exp [omega; upsilon]
  jacobian.block<3,3>(0,6).setIdentity();    // jac Xj for upsilon
  jacobian = dhdXc * jacobian;
#ifdef CHECK_NAN
  if(jacobian.hasNaN())
    throw -1;
#endif
  _jacobianOplusXi = jacobian.block<3,3>(0,0);
  _jacobianOplusXj = jacobian.block<3,6>(0,3);
  return;
}

void VertexSwitchLinear::setEstimate(const number_t &et) {
  _x=et;
  _estimate=_x;
}

void VertexSwitchLinear::oplusImpl(const number_t* update) {
  _x += update[0];
  if (_x<0.) _x=0.;
  if (_x>1.) _x=1.;
  _estimate=_x;
}

void EdgeSwitchPrior::linearizeOplus() {
  _jacobianOplusXi[0]=-1.0;
}

void EdgeSwitchPrior::computeError() {
  const VertexSwitchLinear* s = static_cast<const VertexSwitchLinear*>(_vertices[0]);
  _error[0] = measurement() - s->x();
}

EdgeSwSE3PointXYZDepth::EdgeSwSE3PointXYZDepth(const Param* param)
: param_(param) {
#if 1
  resize(3);
#else
  resize(3); // 3 node
  _jacobianOplus[0].resize(3,6); 
  _jacobianOplus[1].resize(3,3);
  _jacobianOplus[2].resize(3,1);
#endif

  information().setIdentity(3,3);
  information()(2,2) = 1e-4; // inverted covariance for inverse depth
  //information() *= 1e+4;
}

void EdgeSwSE3PointXYZDepth::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const VertexSwitchLinear*     vp = static_cast<const VertexSwitchLinear*>(_vertices[2]);
  g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 h(vj->estimate().map(vi->estimate()));
  auto invd = std::max(MIN_NUM, 1./std::max(MIN_NUM,h[2]) );
  h[0] *= invd; // Xc/Zc
  h[1] *= invd; // Yc/Zc
  h[2] = invd;
  _error = vp->estimate() * (h-obs);
}

void EdgeSwSE3PointXYZDepth::linearizeOplus() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const VertexSwitchLinear*     vp = static_cast<const VertexSwitchLinear*>(_vertices[2]);
  const g2o::SE3Quat& Tcw = vj->estimate();
  const g2o::Vector3& Xw  = vi->estimate();
  g2o::Vector3 Xc = Tcw.map(Xw);
  auto invd = std::max(MIN_NUM, 1./std::max(MIN_NUM,Xc[2]) );
  const double inv_Zc2 = invd* invd;
  Eigen::Matrix<double,3,3> dhdXc;
  dhdXc << invd, 0., -Xc[0]*inv_Zc2,
           0., invd, -Xc[1]*inv_Zc2,
           0., 0.,     -inv_Zc2;
  // swtich error
  dhdXc *= vp->estimate();

  // Param 10 = 3 + 6 + 1
  Eigen::Matrix<double,3,9> jacobian;
  jacobian.block<3,3>(0,0) = Tcw.rotation().toRotationMatrix(); // jac Xi
  jacobian.block<3,3>(0,3) = -g2o::skew(Xc); // jac Xj for omega in g2o::SE3Quat::exp [omega; upsilon]
  jacobian.block<3,3>(0,6).setIdentity();    // jac Xj for upsilon
  jacobian = dhdXc * jacobian;
#ifdef CHECK_NAN
  if(jacobian.hasNaN())
    throw -1;
#endif

  _jacobianOplus[0] = jacobian.block<3,3>(0,0);
  _jacobianOplus[1] = jacobian.block<3,6>(0,3);

  g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 h;
  h.head<2>() = invd*Xc.head<2>();
  h[2] = invd;
  _jacobianOplus[2] = h-obs;
  return;
}


} // namespace seg
