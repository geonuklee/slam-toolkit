/*
#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include <g2o/types/slam3d/se3quat.h>
*/
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <iostream>
#include "simd.h"
#include "dataset.h"
#include "camera.h"

// https://stackoverflow.com/questions/17735863/opencv-save-cv-32fc1-images
bool writeRawImage(const cv::Mat& image, const std::string& filename) {
  std::ofstream file;
    file.open (filename, std::ios::out|std::ios::binary);
    if (!file.is_open())
        return false;
    file.write(reinterpret_cast<const char *>(&image.rows), sizeof(int));
    file.write(reinterpret_cast<const char *>(&image.cols), sizeof(int));
    const int depth = image.depth();
    const int type  = image.type();
    const int channels = image.channels();
    file.write(reinterpret_cast<const char *>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char *>(&type), sizeof(type));
    file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    int sizeInBytes = image.step[0] * image.rows;
    file.write(reinterpret_cast<const char *>(&sizeInBytes), sizeof(int));
    file.write(reinterpret_cast<const char *>(image.data), sizeInBytes);
    file.close();
    return true;
}

bool readRawImage(cv::Mat& image, const std::string& filename) {
    int rows, cols, data, depth, type, channels;
    std::ifstream file (filename, std::ios::in|std::ios::binary);
    if (!file.is_open())
        return false;
    try {
        file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        file.read(reinterpret_cast<char *>(&depth), sizeof(depth));
        file.read(reinterpret_cast<char *>(&type), sizeof(type));
        file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
        file.read(reinterpret_cast<char *>(&data), sizeof(data));
        image = cv::Mat(rows, cols, type);
        file.read(reinterpret_cast<char *>(image.data), data);
    } catch (...) {
        file.close();
        return false;
    }

    file.close();
    return true;
}


int TestFromDepthImageFile(int argc, char** argv) {
  cv::Mat depth;
  readRawImage(depth, "depth.raw");
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .4;

  bool stop_flag = false;
  int n = 0;
  while(true){
    cv::imshow("depth", 0.01*depth);
    float fx = 512.;
    float fy = fx;
    int sample_offset = 4;

    for(bool with_simd : {false, true}){
      cv::Mat dd_edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
      cv::Mat gradx = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
      cv::Mat grady = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
      cv::Mat valid_grad = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
      cv::Mat concave_edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
      cv::Mat valid_mask = depth > 0.;

      {
        auto start = std::chrono::steady_clock::now();
        GetDDEdges(depth,dd_edges,true); 
        auto stop = std::chrono::steady_clock::now();
        std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
        cv::Mat dst;
        cv::cvtColor(255*dd_edges, dst, cv::COLOR_GRAY2BGR);
        cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
        cv::imshow( with_simd?"avx edge":"old edge", dst);
      }
      {
        auto start = std::chrono::steady_clock::now();
        GetGrad(depth, fx, fy, valid_mask, sample_offset, gradx, grady, valid_grad, with_simd);
        auto stop = std::chrono::steady_clock::now();
        std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
        cv::Mat dst = VisualizeGrad(gradx, grady);
        cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
        cv::imshow( with_simd?"avx grad":"old grad", dst);
      }
      {
        auto start = std::chrono::steady_clock::now();
        GetConcaveEdges(gradx,grady,depth,valid_mask,sample_offset, fx,fy, -100., concave_edges, with_simd);
        auto stop = std::chrono::steady_clock::now();
        std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
        cv::Mat dst;
        cv::cvtColor(255*concave_edges, dst, cv::COLOR_GRAY2BGR);
        cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
        cv::imshow( with_simd?"avx concave":"old concave", dst);
      }
    }

    n = (n+1) % 5; // cpu 혹사 방지...
    if(n == 0)
      stop_flag = true;
    char c= cv::waitKey(stop_flag?0:1);
    if(c=='q')
      break;
    else if(c=='s')
      stop_flag = !stop_flag;
  }

  return 1;
}

int TestWaymodataset(int argc, char** argv) {
  // 주어진 RGB+Dense Depthmap에서 
  const std::string dataset_path = GetPackageDir()+ "/../thirdparty/waymo-dataset/output/";
  const std::string seq(argv[1]);

  KittiDataset dataset(seq,dataset_path);
  const DepthCamera* camera = dynamic_cast<const DepthCamera*>(dataset.GetCamera());
  assert(camera);
  float fx = camera->GetK()(0,0);
  float fy = camera->GetK()(1,1);
  int sample_offset = 6;

  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .4;
  bool stop_flag = true;
  for(int i=0; i<dataset.Size(); i+=1){
    const cv::Mat rgb   = dataset.GetImage(i,cv::IMREAD_UNCHANGED);
    const cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat dd_edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::Mat gradx = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
    cv::Mat grady = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
    cv::Mat valid_grad = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::Mat concave_edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::Mat valid_mask = depth > 0.;

    bool with_simd = true;
    {
      auto start = std::chrono::steady_clock::now();
      GetDDEdges(depth,dd_edges,true); 
      auto stop = std::chrono::steady_clock::now();
      std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
      cv::Mat dst;
      cv::cvtColor(255*dd_edges, dst, cv::COLOR_GRAY2BGR);
      cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
      cv::imshow( with_simd?"avx edge":"old edge", dst);
    }
    {
      auto start = std::chrono::steady_clock::now();
      GetGrad(depth, fx, fy, valid_mask, sample_offset, gradx, grady, valid_grad, with_simd);
      auto stop = std::chrono::steady_clock::now();
      std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
      cv::Mat dst = VisualizeGrad(gradx, grady);
      cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
      cv::imshow( with_simd?"avx grad":"old grad", dst);
      cv::imshow( "valid_grad", 255*valid_grad);
    }
    {
      auto start = std::chrono::steady_clock::now();
      GetConcaveEdges(gradx,grady,depth,valid_mask,sample_offset, fx,fy, -100., concave_edges, with_simd);
      auto stop = std::chrono::steady_clock::now();
      std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
      cv::Mat dst;
      cv::cvtColor(255*concave_edges, dst, cv::COLOR_GRAY2BGR);
      cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
      cv::imshow( with_simd?"avx concave":"old concave", dst);
    }
    cv::imshow("rgb", rgb);
    cv::imshow("depth", .01*depth);
    char c= cv::waitKey(stop_flag?0:1);
    if(c=='q')
      break;
    else if(c=='s')
      stop_flag = !stop_flag;

  }
  std::cout << "Done. The end of the dataset." << std::endl;
  return 1;
}


int TestFromOutlineImageFile(int argc, char** argv) {
  cv::Mat outline_edges = cv::imread("..//outline.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat rgb = cv::imread("../rgb.bmp", cv::IMREAD_COLOR);
  cv::Mat valid_mask = cv::Mat::ones(outline_edges.size(), CV_8UC1);

  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .8;


  bool stop_flag = false;
  int n = 0;
  while(true){
    bool limit_expand_range = false;
    //cv::Mat rgb4vis = rgb.clone();
    cv::Mat rgb4vis;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat mask_old = OLD::Segment(outline_edges, valid_mask, limit_expand_range, rgb4vis);
    auto t1 = std::chrono::steady_clock::now();
    cv::Mat dst_old = GetColoredLabel(mask_old); {
      std::ostringstream os;
      os << "old : " << std::setprecision(3) <<
        std::chrono::duration<float, std::milli>(t1-t0).count() << "[milli sec]";
      cv::putText(dst_old, os.str(), cv::Point(10,20), font_face, font_scale, CV_RGB(255,0,0),2);
      std::cout << os.str() << std::endl;
    }
    cv::imshow("mask_old", dst_old);

    auto t2 = std::chrono::steady_clock::now();
    cv::Mat mask_new;
    int n_octave = 6;
    int n_downsample = 2; 
    NEW::Segment(outline_edges, n_octave, n_downsample, mask_new);
    auto t3 = std::chrono::steady_clock::now();

    cv::Mat dst_new = GetColoredLabel(mask_new,true); 
    if(true){
      std::ostringstream os;
      os << "new : " << std::setprecision(3) <<
        std::chrono::duration<float, std::milli>(t3-t2).count() << "[milli sec]";
      cv::putText(dst_new, os.str(), cv::Point(10,20), font_face, font_scale, CV_RGB(255,0,0),2);
      std::cout << os.str() << std::endl;
    }
    cv::imshow("mask_new", dst_new);

    cv::imshow("outline", 255*outline_edges);
    n = (n+1) % 5; // cpu 혹사 방지...
    if(n == 0)
      stop_flag = true;

    char c= cv::waitKey(stop_flag?0:30);
    if(c=='q')
      break;
    else if(c=='s')
      stop_flag = !stop_flag;

  }

  return 1;
}


int main(int argc, char** argv){
  //return TestWaymodataset(argc, argv);
  //return TestFromDepthImageFile(argc, argv);
  return TestFromOutlineImageFile(argc, argv);
}
