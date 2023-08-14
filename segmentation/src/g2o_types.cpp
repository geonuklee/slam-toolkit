#include "g2o_types.h"
#include <g2o/config.h>
#define CHECK_NAN

namespace seg {

const double MIN_NUM = 1e-10;

EdgeSE3PointXYZDepth::EdgeSE3PointXYZDepth(const Param* param,
                                           const double& uv_info,
                                           const double& invd_info
                                           )
  :param_(param),
  g2o::BaseBinaryEdge<3, g2o::Vector3, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
{
  auto& info = information();
  info.setZero();
  info(0,0) = info(1,1) = uv_info;
  info(2,2)             = invd_info;
}

inline double GetInverse(const double& z) {
  return std::max(MIN_NUM, 1./std::max<double>(MIN_NUM,z) );
}

void EdgeSE3PointXYZDepth::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 h(vj->estimate().map(vi->estimate()));
  auto invd = GetInverse(h[2]);
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
  auto invd = GetInverse(Xc[2]);
  const double inv_Zc2 = invd* invd;
  Eigen::Matrix<double,3,3> dhdXc;
  dhdXc << invd, 0., -Xc[0]*inv_Zc2,
        0., invd, -Xc[1]*inv_Zc2,
        0., 0.,   -inv_Zc2;

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

EdgeProjection::EdgeProjection(const Param* param,
                               const double& uv_info)
  : param_(param),
  g2o::BaseBinaryEdge<2, g2o::Vector2, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
{
  auto& info = information();
  info.setZero();
  info(0,0) = info(1,1) = uv_info;
}

void EdgeProjection::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  g2o::Vector2 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 Xc(vj->estimate().map(vi->estimate()));
  auto invd = GetInverse(Xc[2]);
  g2o::Vector2 h = Xc.head<2>();
  h[0] *= invd; // Xc/Zc
  h[1] *= invd; // Yc/Zc
  _error = h-obs;
}

void EdgeProjection::linearizeOplus() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const g2o::SE3Quat& Tcw = vj->estimate();
  const g2o::Vector3& Xw  = vi->estimate();
  g2o::Vector3 Xc = Tcw.map(Xw);
  auto invd = GetInverse(Xc[2]);
  const double inv_Zc2 = invd* invd;
  Eigen::Matrix<double,2,3> dhdXc;
  dhdXc << invd, 0., -Xc[0]*inv_Zc2,
        0., invd, -Xc[1]*inv_Zc2;
  Eigen::Matrix<double,3,9> dXcdP;
  dXcdP.block<3,3>(0,0) = Tcw.rotation().toRotationMatrix(); // jac Xi
  dXcdP.block<3,3>(0,3) = -g2o::skew(Xc); // jac Xj for omega in g2o::SE3Quat::exp [omega; upsilon]
  dXcdP.block<3,3>(0,6).setIdentity();    // jac Xj for upsilon
#ifdef CHECK_NAN
  if(dXcdP.hasNaN())
    throw -1;
#endif
  _jacobianOplusXi = dhdXc* dXcdP.block<3,3>(0,0);
  _jacobianOplusXj = dhdXc* dXcdP.block<3,6>(0,3);
  return;
}

EdgeSwProjection::EdgeSwProjection(const Param* param,
                                   const double& uv_info)
  : param_(param),
  g2o::BaseMultiEdge<2, g2o::Vector2>()
{
  resize(3);
  auto& info = information();
  info.setZero();
  info(0,0) = info(1,1) = uv_info;
}

void EdgeSwProjection::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const VertexSwitchLinear*     vp = static_cast<const VertexSwitchLinear*>(_vertices[2]);
  g2o::Vector2 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 Xc(vj->estimate().map(vi->estimate()));
  auto invd = GetInverse(Xc[2]);
  g2o::Vector2 h = Xc.head<2>();
  h[0] *= invd; // Xc/Zc
  h[1] *= invd; // Yc/Zc
  _error = vp->estimate() * (h-obs);
}

void EdgeSwProjection::linearizeOplus() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const VertexSwitchLinear*     vp = static_cast<const VertexSwitchLinear*>(_vertices[2]);
  const g2o::SE3Quat& Tcw = vj->estimate();
  const g2o::Vector3& Xw  = vi->estimate();
  g2o::Vector3 Xc = Tcw.map(Xw);
  auto invd = GetInverse(Xc[2]);
  const double inv_Zc2 = invd* invd;
  Eigen::Matrix<double,2,3> dhdXc;
  dhdXc << invd, 0., -Xc[0]*inv_Zc2,
        0., invd, -Xc[1]*inv_Zc2;
  Eigen::Matrix<double,3,9> dXcdP;
  dXcdP.block<3,3>(0,0) = Tcw.rotation().toRotationMatrix(); // jac Xi
  dXcdP.block<3,3>(0,3) = -g2o::skew(Xc); // jac Xj for omega in g2o::SE3Quat::exp [omega; upsilon]
  dXcdP.block<3,3>(0,6).setIdentity();    // jac Xj for upsilon

  _jacobianOplus[0] = dhdXc * dXcdP.block<3,3>(0,0);
  _jacobianOplus[1] = dhdXc * dXcdP.block<3,6>(0,3);

  g2o::Vector2 h = Xc.head<2>();
  h[0] *= invd; // Xc/Zc
  h[1] *= invd; // Yc/Zc
  g2o::Vector2 obs(_measurement); // [nu, nv, 1./Zc]
  _jacobianOplus[2] = h-obs;
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

EdgeSwSE3PointXYZDepth::EdgeSwSE3PointXYZDepth(const Param* param,
                                               const double& uv_info,
                                               const double& invd_info)
: param_(param) {
#if 1
  resize(3);
#else
  resize(3); // 3 node
  _jacobianOplus[0].resize(3,6); 
  _jacobianOplus[1].resize(3,3);
  _jacobianOplus[2].resize(3,1);
#endif
  auto& info = information();
  info.setZero();
  info(0,0) = info(1,1) = uv_info;
  info(2,2)             = invd_info;
}

void EdgeSwSE3PointXYZDepth::computeError() {
  const g2o::VertexSBAPointXYZ* vi = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap*   vj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const VertexSwitchLinear*     vp = static_cast<const VertexSwitchLinear*>(_vertices[2]);
  g2o::Vector3 obs(_measurement); // [nu, nv, 1./Zc]
  g2o::Vector3 h(vj->estimate().map(vi->estimate()));
  auto invd = GetInverse(h[2]);
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
  auto invd = GetInverse(Xc[2]);
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
