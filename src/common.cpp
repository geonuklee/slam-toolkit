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

#include "common.h"

std::string GetPackageDir(){
  std::string dir = PACKAGE_DIR;
  return dir;
}

cv::Mat cvt2cvMat(const Eigen::Matrix<double,-1,-1>& eigen_mat){
  int r = eigen_mat.rows();
  int c = eigen_mat.cols();
  cv::Mat cv_mat = cv::Mat::zeros(r, c, CV_64F);
  for(int i = 0; i < r; i++){
    for(int j= 0; j < c; j++){
      cv_mat.at<double>(i,j) = eigen_mat(i,j);
    }
  }
  return cv_mat;
}

Eigen::MatrixXd cvt2Eigen(cv::Mat cm) {
  Eigen::MatrixXd em(cm.rows, cm.cols);
  if(cm.type() == CV_64F){
  for(int i = 0; i < cm.rows; i++)
    for(int j= 0; j < cm.cols; j++)
      em(i,j) = cm.at<double>(i,j);
  }
  else if(cm.type() == CV_32F){
  for(int i = 0; i < cm.rows; i++)
    for(int j= 0; j < cm.cols; j++)
      em(i,j) = cm.at<float>(i,j);
  }
  else{
    std::cerr << "Unexpected type" << std::endl;
    throw 1;
  }
  return em;
}

BrightenSE3::BrightenSE3(){
  Tcw_ = g2o::SE3Quat();
  brightness_.setZero();
}

BrightenSE3::BrightenSE3(const g2o::SE3Quat& Tcw, const Eigen::Vector2d& brightness) {
  Tcw_ = Tcw;
  brightness_ = brightness;
}

