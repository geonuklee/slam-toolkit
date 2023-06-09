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

#include "camera.h"

bool Camera::IsInImage(const Eigen::Vector2d& uv) const {
  if(uv.x() < 0.)
    return false;
  if(uv.y() < 0.)
    return false;
  if(uv.x() > width_)
    return false;
  if(uv.y() > height_)
    return false;
  return true;
}

Camera::Camera(const Eigen::Matrix<double,3,3>& K,
               const Eigen::VectorXd& D,
               int width,
               int height
               )
  : width_(width), height_(height), K_(K), D_(D)
{
  if(D_.rows() < 4)
    throw std::invalid_argument("Need 4 distortion param");
  invK_ = K.inverse();
}

Eigen::Vector2d Distort(const Eigen::VectorXd& D,
                        const Eigen::Vector2d& normalized_x){
  // ref) modules/calib3d/src/calibration.cpp
  const double& k0 = D(0);
  const double& k1 = D(1);
  const double& k2 = D(2);
  const double& k3 = D(3);
  double x = normalized_x.x();
  double y = normalized_x.y();
  double r2 = x*x + y*y;
  double r4 = r2*r2;
  double a1 = 2.*x*y;
  double a2 = r2 + 2.*x*x;
  double a3 = r2 + 2.*y*y;
  double cdist = 1. + k0*r2 + k1*r4; // + k[4]*r6;
  double xd0 = x*cdist + k2*a1 + k3*a2;
  double yd0 = y*cdist + k2*a3 + k3*a1;
  return Eigen::Vector2d(xd0, yd0);
}

Eigen::Vector2d Camera::Project(const Eigen::Vector3d& Xc) const {
#if 1
  // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html , 'description'
  Eigen::Vector2d x = Distort(D_, Xc.head<2>()/Xc.z() );
  const double& fx = K_(0,0);
  const double& fy = K_(1,1);
  const double& cx = K_(0,2);
  const double& cy = K_(1,2);
  return Eigen::Vector2d(fx*x.x()+cx, fy*x.y()+cy);
#else
  cv::Mat cvK = cvt2cvMat(*(kf->K_));
  cv::Mat cvD = cvt2cvMat(*(kf->D_));
  cvK.convertTo(cvK, CV_32F);
  cvD.convertTo(cvD, CV_32F);

  cv::Mat rvec = cv::Mat::zeros(3,1, CV_32F);
  cv::Mat tvec = cv::Mat::zeros(3,1, CV_32F);

  std::vector<cv::Point3f> input = { cv::Point3f(Xc.x(), Xc.y(), Xc.z() ) };
  std::vector<cv::Point2f> output;
  cv::projectPoints(input, rvec, tvec, cvK, cvD, output);
  return Eigen::Vector2d( output.at(0).x, output.at(0).y);
#endif
}

Eigen::Vector3d Camera::NormalizedUndistort(const Eigen::Vector2d& uv) const {
#if 1
  // See document of cv::undistortPoints()
  // https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga55c716492470bfe86b0ee9bf3a1f0f7e
  const double& fx = K_(0,0);
  const double& fy = K_(1,1);
  const double& cx = K_(0,2);
  const double& cy = K_(1,2);
  Eigen::Vector2d normalized_uv( (uv.x() - cx)/fx, (uv.y() - cy)/fy );
  Eigen::Vector2d normalized_x = normalized_uv;
  for(size_t i = 0; i < 5; i++){
    Eigen::Vector2d err = normalized_uv - Distort(D_, normalized_x);
    normalized_x += err;
  }
  return Eigen::Vector3d(normalized_x.x(), normalized_x.y(), 1.);
#else
  cv::Mat cvK = cvt2cvMat(*K_);
  cv::Mat cvD = cvt2cvMat(*D_);
  std::vector<cv::Point2f> input = { pt};
  std::vector<cv::Point2f> output;

  cvK.convertTo(cvK, CV_32F);
  cvD.convertTo(cvD, CV_32F);
  cv::undistortPoints(input, output, cvK, cvD);
  const cv::Point2d& nuv = output.at(0); // undistorted, noramlized uv.
  return Eigen::Vector3d(nuv.x, nuv.y, 1.);
#endif
}

StereoCamera::StereoCamera(const Eigen::Matrix<double,3,3>& Kl, const Eigen::VectorXd& Dl,
               const Eigen::Matrix<double,3,3>& Kr, const Eigen::VectorXd& Dr,
               const g2o::SE3Quat& Tlr,
               int width, int height
               ) 
: Camera(Kl, Dl, width, height),
  Tlr_(Tlr), Kr_(Kr), Dr_(Dr)
{

}


double GetInetrpolatedIntensity(cv::Mat gray, const Eigen::Vector2d& uv) {
  // ref) https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
  int x = (int)uv.x();
  int y = (int)uv.y();

  int x0 = cv::borderInterpolate(x,   gray.cols, cv::BORDER_REFLECT_101);
  int x1 = cv::borderInterpolate(x+1, gray.cols, cv::BORDER_REFLECT_101);
  int y0 = cv::borderInterpolate(y,   gray.rows, cv::BORDER_REFLECT_101);
  int y1 = cv::borderInterpolate(y+1, gray.rows, cv::BORDER_REFLECT_101);

  float a = uv.x() - (float)x;
  float c = uv.y() - (float)y;

  float i00 = gray.at<unsigned char>(y0, x0);
  float i10 = gray.at<unsigned char>(y0, x1);
  float i01 = gray.at<unsigned char>(y1, x0);
  float i11 = gray.at<unsigned char>(y1, x1);
  float i = ((i00 * (1.f - a) + i10 * a) * (1.f - c) + (i01 * (1.f - a) + i11 * a) * c);


  return i;
}
