#ifndef SEG_G2OTYPES_
#define SEG_G2OTYPES_
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/base_multi_edge.h>

/*
1) slam3d/edge_se3_pointxyz_depth.h, g2o::EdgeSE3PointXYZDepth,
2) sba/types_six_dof_expmap.h, g2o::EdgeStereoSE3ProjectXYZ 를 참고한, Edge함수.
Error정의를 기존 [normalized uv, depth] 대신
nuv, invd depth
Structure param은 그대로 XYZ
*/
namespace seg {

class Param;
extern const double MIN_NUM ;

// D=3 : Measurement dimmension
// E=g2o::Vector3 : Measurement type
// VertexXi=g2o::VertexSBAPointXYZ : First vertex node
// VertexXj=g2o::VertexSE3Expmap : second vertex node
class EdgeSE3PointXYZDepth
  : public g2o::BaseBinaryEdge<3, g2o::Vector3, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3PointXYZDepth(const Param* param);
  void computeError();
  virtual void linearizeOplus();
  virtual bool read(std::istream& is) { return false; }
  virtual bool write(std::ostream& os) const { return false; }
  virtual int measurementDimension() const {return 3;}

private:
  const Param*const param_;
  Eigen::Matrix<number_t,3,9,Eigen::ColMajor> J; // jacobian before projection
};


/*
 * vertex_switchLinear.h
 *
 *  Created on: 17.10.2011
 *      Author: niko
 *  Revised for latest g2o on : 5. 8. 2023
 *      Author: Geonuk Lee
 */
class VertexSwitchLinear : public g2o::BaseVertex<1, number_t> {
public:
  VertexSwitchLinear() { setToOrigin(); };
  virtual bool read(std::istream& is) { return false; }
  virtual bool write(std::ostream& os) const { return false; }
  virtual void setToOriginImpl()  {
    _x=1.;
    _estimate=_x;
  }
  virtual void setEstimate(const number_t &et);
  virtual void oplusImpl(const number_t* update_);
  double x() const { return _x; };
  //! The gradient at the current estimate is always 1;
  double gradient() const { return 1; } ;
private:
  double _x;
};

class EdgeSwitchPrior : public g2o::BaseUnaryEdge<1, double, VertexSwitchLinear> {
public:
  // Measurement가 double인데 EIGEN_MAKE_ALIGNED_OPERATOR_NEW가 필요한가 의문이지만,..
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSwitchPrior(const number_t& inform) {
    information()(0,0) = inform;
  };
  void linearizeOplus();
  void computeError();
  virtual bool read(std::istream& is) { return false; }
  virtual bool write(std::ostream& os) const { return false; }
};

// D=3 : Dimenison
// E=g2o::Vector3 : Measurement
class EdgeSwSE3PointXYZDepth : public g2o::BaseMultiEdge<3, g2o::Vector3> {
public:
  EdgeSwSE3PointXYZDepth(const Param* param);
  virtual bool read(std::istream& is) { return false; }
  virtual bool write(std::ostream& os) const { return false; }
  void computeError();
  void linearizeOplus();
private:
  const Param*const param_;
};


} // namespace seg


#endif