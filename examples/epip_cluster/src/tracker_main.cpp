#include "stdafx.h"
#include "../include/tracker.h"

int main(int argc, char** argv){
  //suggest "17", "13";
  std::string seq(argv[1]);
  std::string dir_im = "/home/geo/dataset/kitti_odometry_dataset/sequences/"+seq+"/image_0/";
  std::string dir_im_r = "/home/geo/dataset/kitti_odometry_dataset/sequences/"+seq+"/image_1/";
  Eigen::Matrix<double,3,3> K;
  K << 707.1, 0., 601.89, 0., 707.1, 183.1, 0., 0., 1.;
  float fx = K(0,0);
  float base = 3.861448000000e+02/fx;
  char buff[100];
  bool stop = true;

  DenseTracker dense_tracker(K, base);
  for(int i = 0; ; i+=1){
    snprintf(buff, sizeof(buff), "%06d.png", i);
    std::string fn_im = dir_im + std::string(buff);
    std::string fn_im_r = dir_im_r + std::string(buff);
    cv::Mat gray = cv::imread(fn_im, cv::IMREAD_GRAYSCALE);
    cv::Mat gray_r = cv::imread(fn_im_r, cv::IMREAD_GRAYSCALE);
    if(gray.empty())
      break;
    dense_tracker.Track(gray, gray_r);

    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  return 0;
}

