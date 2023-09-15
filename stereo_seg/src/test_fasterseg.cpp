/*
#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include <g2o/types/slam3d/se3quat.h>
*/
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <iostream>
#include "simd.h"

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


int TestFasterSeg(int argc, char** argv) {
  cv::Mat depth;
  readRawImage(depth, "depth.raw");
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .4;

  bool stop_flag = false;
  while(true){
    cv::imshow("depth", 0.01*depth);
    {
      auto start = std::chrono::steady_clock::now();
      cv::Mat dd_edges =GetDDEdges(depth,false); 
      auto stop = std::chrono::steady_clock::now();
      std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
      std::cout << etime_msg << std::endl;
      cv::Mat dst;
      cv::cvtColor(255*dd_edges, dst, cv::COLOR_GRAY2BGR);
      cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
      cv::imshow("Old edge", dst);
    }
    {
      auto start = std::chrono::steady_clock::now();
      cv::Mat dd_edges =GetDDEdges(depth,true); 
      auto stop = std::chrono::steady_clock::now();
      std::string etime_msg = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())+ "[micro sec]";
      std::cout << etime_msg << std::endl;
      cv::Mat dst;
      cv::cvtColor(255*dd_edges, dst, cv::COLOR_GRAY2BGR);
      cv::putText(dst, etime_msg, cv::Point(40,15), font_face, font_scale, CV_RGB(255,0,0) );
      cv::imshow("avx edge", dst);
    }

    char c= cv::waitKey(stop_flag?0:1);
    if(c=='q')
      break;
    else if(c=='s')
      stop_flag = !stop_flag;
  }

  return 1;
}

int main(int argc, char** argv){
  return TestFasterSeg(argc, argv);
}
