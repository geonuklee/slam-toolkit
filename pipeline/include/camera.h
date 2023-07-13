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

#ifndef CAMERA_H_
#define CAMERA_H_
#include "stdafx.h"

double GetInetrpolatedIntensity(cv::Mat gray, const Eigen::Vector2d& uv);

class Camera {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual ~Camera() {}
  Camera(const Eigen::Matrix<double,3,3>& K, const Eigen::VectorXd& D, int width, int height);

  Eigen::Vector2d Project(const Eigen::Vector3d& Xc) const;
  Eigen::Vector3d NormalizedUndistort(const Eigen::Vector2d& uv) const;

  const Eigen::Matrix<double,3,3>& GetK() const { return K_; }
  const Eigen::Matrix<double,3,3>& GetInvK() const { return invK_; }
  const Eigen::VectorXd& GetD() const { return D_; }


  bool IsInImage(const Eigen::Vector2d& uv) const;

  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }

  virtual std::string GetType() const { return "mono"; }

protected:

  const int width_;
  const int height_;

  const Eigen::Matrix<double,3,3> K_;
  Eigen::Matrix<double,3,3> invK_;
  const Eigen::VectorXd D_;
};

class StereoCamera : public Camera {
public:
  virtual ~StereoCamera() {}
  StereoCamera(const Eigen::Matrix<double,3,3>& K1, const Eigen::VectorXd& D1, 
               const Eigen::Matrix<double,3,3>& K2, const Eigen::VectorXd& D2,
               const g2o::SE3Quat& Trl,
               int width, int height
               );
  const g2o::SE3Quat& GetTrl() const { return Trl_; }
  virtual std::string GetType() const { return "stereo"; }

protected:
  const g2o::SE3Quat Trl_;
  const Eigen::Matrix<double,3,3> Kr_;
  const Eigen::VectorXd Dr_;
};

class DepthCamera : public Camera {
public:
  virtual ~DepthCamera() {}
  DepthCamera(const Eigen::Matrix<double,3,3>& K, const Eigen::VectorXd& D, int width, int height);
  virtual std::string GetType() const { return "depth"; }
};


#endif
