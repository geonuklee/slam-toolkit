#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>

#include "hitnet.h"
#include <pybind11/embed.h>

// https://stackoverflow.com/questions/17735863/opencv-save-cv-32fc1-images
bool writeRawImage(const cv::Mat& image, const std::string& filename) {
    ofstream file;
    file.open (filename, ios::out|ios::binary);
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
    ifstream file (filename, ios::in|ios::binary);
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

int TestHitNet(int argc, char** argv) {
  if(argc < 1){
    std::cout << "No arguments for dataset name." << std::endl;
    exit(-1);
  }
  std::string seq(argv[1]);
  KittiDataset dataset(seq);
  const auto& Tcws = dataset.GetTcws();
  if(Tcws.empty()){
    std::cout << "Seq" << seq << " with no ground truth trajectory." << std::endl;
  }
  const auto& D = dataset.GetCamera()->GetD();
  std::cout << "Distortion = " << D.transpose() << std::endl;
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);
  // Parameters for goodFeaturesToTrack
  int maxCorners = 100;  // Maximum number of corners to detect
  double qualityLevel = 0.01;  // Quality level threshold
  double minDistance = 10.0;  // Minimum distance between detected corners
  pybind11::scoped_interpreter python; // 이 인스턴스가 파괴되면 인터프리터 종료.
  HITNetStereoMatching hitnet;
  bool stop = true;
  for(int i=0; i<dataset.Size(); i+=1){
    cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);
    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);
    // small input to reserve time.
    cv::Mat disp = hitnet.GetDisparity(gray, gray_r);
    cv::Mat dst;
    cv::vconcat(rgb, rgb_r, dst);
    for(auto pt : corners){
      cv::circle(dst, pt, 5, CV_RGB(0,0,255),1);
      float dp = disp.at<float>(pt);
      cv::Point2f pt2(pt.x-dp, pt.y+rgb.rows);
      cv::line(dst, pt, pt2, CV_RGB(0,255,0),1 );
      cv::circle(dst, pt2, 5, CV_RGB(0,0,255),1);
    }

    cv::Mat depth = Disparit2Depth(disp,base_line,fx,1.);
    writeRawImage(depth, "depth.raw");
    /*
    cv::Mat fdepth;
    readRawImage(fdepth, "depth.raw");
    cv::imshow("depth_from_file", 0.01*fdepth);
    */

    cv::waitKey();

    cv::imshow("stereo", dst);
    cv::Mat ndisp;
    cv::normalize(disp, ndisp, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("ndisp", ndisp);
    cv::imshow("rgb", rgb);
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }
  std::cout << "Done. The end of the sequence" << std::endl;
  return 1;
}

int main(int argc, char** argv){
  return TestHitNet(argc, argv);
}
