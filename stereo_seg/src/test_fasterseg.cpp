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
  std::shared_ptr<OutlineEdgeDetector> edge_detector(new OutlineEdgeDetectorWithSIMD);

  bool stop_flag = false;
  int n = 0;
  while(true){
    cv::imshow("depth", 0.01*depth);
    float fx = 512.;
    float fy = fx;
    int sample_offset = 4;

    edge_detector->PutDepth(depth,fx, fy);
    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();

    cv::imshow("outline", 255*outline_edges);

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

  std::shared_ptr<OutlineEdgeDetector> edge_detector(new OutlineEdgeDetectorWithSIMD);
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .4;
  bool stop_flag = true;
  for(int i=0; i<dataset.Size(); i+=1){
    const cv::Mat rgb   = dataset.GetImage(i,cv::IMREAD_UNCHANGED);
    const cv::Mat depth = dataset.GetDepthImage(i);
    cv::Mat valid_mask = depth > 0.;
    edge_detector->PutDepth(depth,fx, fy);
    cv::Mat gradx = edge_detector->GetGradx();
    cv::Mat grady = edge_detector->GetGrady();
    cv::Mat outline_edges = edge_detector->GetOutline();

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
  /*
  cv::Mat outline_edges = cv::imread("..//outline.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat rgb = cv::imread("../rgb.bmp", cv::IMREAD_COLOR);
  cv::Mat valid_mask = cv::Mat::ones(outline_edges.size(), CV_8UC1);
  */

  ImageTrackerNew img_tracker;
  std::shared_ptr<SegmentorNew> segmentor(new SegmentorNew);


  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = .8;
  int n = 0;
  std::string output_images = "../../segmentation/output_images/";
  bool stop_flag = true;

  cv::Mat marker0_, gray0_, outline0_;
  while(true){
    cv::Mat gray = cv::imread(output_images+"gray"+std::to_string(n)+".png", cv::IMREAD_GRAYSCALE);
    if(gray.empty()){
      std::cout << "end of image files" << std::endl;
      break;
    }
    cv::Mat outline_edges = cv::imread(output_images+"outline"+std::to_string(n)+".png", cv::IMREAD_GRAYSCALE);
    cv::Mat depth;
    readRawImage(depth, output_images+"depth"+std::to_string(n)+".raw");
    cv::Mat valid_mask = depth > 0.1;

    segmentor->Put(outline_edges, valid_mask);
    cv::Mat unsync_marker = segmentor->GetMarker();

    float sync_min_iou = .3;
    auto t0_track = std::chrono::steady_clock::now();
    img_tracker.Put(gray, unsync_marker, sync_min_iou);
    auto t1_track = std::chrono::steady_clock::now();
    std::cout <<"etime of img_track for " << " : " << std::setprecision(3) <<
      std::chrono::duration<float, std::milli>(t1_track-t0_track).count() << "[milli sec]" << std::endl;
    cv::Mat synced_marker = img_tracker.GetSyncedMarked();
    const std::vector<cv::Mat>& flow = img_tracker.GetFlow();
    cv::imshow("gray", gray);
    cv::imshow("depth", 0.01*depth);
    cv::imshow("outline", 255*outline_edges);
    cv::imshow("synced marker", GetColoredLabel(synced_marker, true));
    gray0_ = gray;
    outline0_ = outline_edges;
    marker0_ = unsync_marker;
    char c= cv::waitKey(stop_flag?0:30);
    if(c=='q')
      break;
    else if(c=='s')
      stop_flag = !stop_flag;
    n+=1;
  }
  return 1;
}


int main(int argc, char** argv){
  //return TestWaymodataset(argc, argv);
  //return TestFromDepthImageFile(argc, argv);
  return TestFromOutlineImageFile(argc, argv);
}
