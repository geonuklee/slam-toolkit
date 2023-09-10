#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>

#include "hitnet.h"
#include <pybind11/embed.h>

#if 0
cv::Mat GetDisparity(cv::cuda::GpuMat g_gray, cv::cuda::GpuMat g_gray_r){
  cv::cuda::GpuMat  g_disp;
  cv::Mat disparity;
  /*
  {
    static auto bm = cv::cuda::createStereoBM();
    bm->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disp); // 8UC
  }
  */
  if(false){
    static auto sbp = cv::cuda::createStereoBeliefPropagation();
    sbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  else{
    int ndisp=128;
    int iters=4;
    int levels=4;
    int nr_plane=4;
    static auto csbp = cv::cuda::createStereoConstantSpaceBP(ndisp,iters,levels,nr_plane);
    csbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  disparity.convertTo(disparity, CV_32FC1);
  return disparity;
}
#endif


int TestKitti(int argc, char** argv) {
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
  if(argc < 1){
    std::cout << "No arguments for dataset name." << std::endl;
    exit(-1);
  }
  return TestKitti(argc, argv);
}
