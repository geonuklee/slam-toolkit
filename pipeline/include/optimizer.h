/*
Copyright (c) 2020 Geonuk Lee

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "stdafx.h"
#include "common.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

class Mappoint;
class Frame;

class EdgeSE3 : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3(){
    }

    bool read(std::istream& is){ return false; }
    bool write(std::ostream& os) const{ return false; }
    void computeError();
};

class DirectPyramid {
public:

  DirectPyramid();

  void SetLv(int lv);
  void PutFrame(const Frame* frame);

  int GetLv() const { return lv_;}
  double GetScale() const { return scale_; }
  const cv::Mat& GetLvGray(const Frame* frame) const { return lv_gray_.at(frame); }

  double GetRatio() const { return ratio_; }

private:
  const double ratio_;
  std::map<const Frame*, cv::Mat> org_gray_;
  std::map<const Frame*, cv::Mat> lv_gray_;
  int lv_;
  double scale_;
};

class Pattern {
public:
  static void GetPattern(int i, int& dx, int& dy);
  static double GetRadius();
};

class VertexBrightenSE3 : public g2o::BaseVertex<8, BrightenSE3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexBrightenSE3() { }
  bool read(std::istream& is) { return false; }
  bool write(std::ostream& os) const { return false; }
  void oplusImpl(const number_t* update_);
  void setToOriginImpl();
};

class EdgeBrightenessPrior : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexBrightenSE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeBrightenessPrior(double info_a, double info_b);
  bool read(std::istream& is) { return false; }
  bool write(std::ostream& os) const { return false; }
  void computeError();
  void linearizeOplus();
};

class EdgeProjectBrightenXYZ : public g2o::BaseBinaryEdge<8, Eigen::Matrix<double,8,1>,
                                                          g2o::VertexSBAPointXYZ,
                                                          VertexBrightenSE3 > {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectBrightenXYZ(DirectPyramid* pyramid, const Frame* frame, const Mappoint* mp);
  bool read(std::istream& is) { return false; }
  bool write(std::ostream& os) const { return false; }
  void computeError();
  void linearizeOplus();

  static bool GetError(const cv::Mat& scaled_gray0,
                       const cv::Mat& scaled_gray,
                       const BrightenSE3& brighten_pose0,
                       const Mappoint* mp,
                       double image_scale,
                       const BrightenSE3& brighten_pose,
                       const Eigen::Vector3d Xw,
                       Measurement& error);

private:
  const DirectPyramid*const pyramid_;
  const Frame*const frame_;
  const Mappoint*const mp_;
};

g2o::SE3Quat AlignTrajectory(const EigenMap<int, g2o::SE3Quat>& gtTcws,
                             const std::map<int, Frame*>& frames,
                             const g2o::SE3Quat Tgt_est_prior
                             );

#endif
