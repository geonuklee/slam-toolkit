#ifndef COMMON_H_
#define COMMON_H_

#include "stdafx.h"

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
const std::string RESET       = "\033[0m";
const std::string BLACK       = "\033[30m";      
const std::string RED         = "\033[31m";      
const std::string GREEN       = "\033[32m";      
const std::string YELLOW      = "\033[33m";      
const std::string BLUE        = "\033[34m";      
const std::string MAGENTA     = "\033[35m";      
const std::string CYAN        = "\033[36m";      
const std::string WHITE       = "\033[37m";      
const std::string BOLDBLACK   = "\033[1m\033[30m";
const std::string BOLDRED     = "\033[1m\033[31m";
const std::string BOLDGREEN   = "\033[1m\033[32m";
const std::string BOLDYELLOW  = "\033[1m\033[33m";
const std::string BOLDBLUE    = "\033[1m\033[34m";
const std::string BOLDMAGENTA = "\033[1m\033[35m";
const std::string BOLDCYAN    = "\033[1m\033[36m";
const std::string BOLDWHITE   = "\033[1m\033[37m";

cv::Mat cvt2cvMat(const Eigen::Matrix<double,-1,-1>& eigen_mat);

Eigen::MatrixXd cvt2Eigen(cv::Mat cm);

template <typename K, typename T>
using EigenMap = std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<const K, T> > >;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T> >;


std::string GetPackageDir(); 

template<typename T>
T Get(const cv::FileStorage& fs, const std::string& name) {
  T val;
  fs[name] >> val;
  return val;
}

/*
template<typename T>
T GetRequiredParam(ros::NodeHandle nh, const std::string name){
  T val;
  if(!nh.getParam(name, val)){
    std::cerr << RED << "Failure" << RESET << " to get param : " << RED << name << RESET << std::endl;
    throw 1;
  }
  static std::set<std::string> closed;
  if(!closed.count(name)){
    std::cout << BLUE << name << RESET << ":" << val << std::endl;
    closed.insert(name);
  }
  return val;
}

template<typename T>
std::vector<T> GetRequiredParamVector(ros::NodeHandle nh, const std::string name){
  std::vector<T> val;
  if(!nh.getParam(name, val)){
    std::cerr << RED << "Failure" << RESET << " to get param : " << RED << name << RESET << std::endl;
    throw 1;
  }

  static std::set<std::string> closed;
  if(!closed.count(name)){
    std::cout << BLUE << name << RESET << ":";
    for(const T& each : val)
      std::cout << each << ",";
    std::cout << std::endl;
    closed.insert(name);
  }
  return val;
}

template<typename T, int r, int c>
Eigen::Matrix<T,r,c> GetRequiredMatrix(const ros::NodeHandle& nh, std::string name){
  std::vector<T> vec;
  if(!nh.getParam(name, vec)){
    std::cerr << RED << "Failure" << RESET << " to get param : " << RED << name << RESET << std::endl;
    throw 1;
  }

  Eigen::Matrix<T,r,c> mat;
  for(int i=0; i<r; i++)
    for(int j=0; j<c; j++)
      mat(i,j) = vec.at(i*c + j);


  static std::set<std::string> closed;
  if(!closed.count(name)){
    std::cout <<BLUE << name << RESET;
    if(c==1)
      std::cout << ":" << mat.transpose() << std::endl;
    else
      std::cout << std::endl << ":" << mat << std::endl;
    closed.insert(name);
  }
  return mat;
}

template<typename T, int r, int c>
Eigen::Matrix<T,r,c> GetParam(const ros::NodeHandle& nh, std::string key){
  std::vector<T> vec;
  nh.getParam(key, vec);
  Eigen::Matrix<T,r,c> mat;
  for(int i=0; i<r; i++)
    for(int j=0; j<c; j++)
      mat(i,j) = vec.at(i*c + j);
  return mat;
}

*/

template<typename T>
T GetFromFileStorage(const cv::FileStorage& fs, const std::string& name) {
  T val;
  fs[name] >> val;
  return val;
}

struct BrightenSE3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BrightenSE3();
  BrightenSE3(const g2o::SE3Quat& Tcw, const Eigen::Vector2d& brightness);

  g2o::SE3Quat Tcw_;
  Eigen::Vector2d brightness_;
};

#endif
