#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafilters.hpp>

#include "../include/seg.h"
#include "../include/util.h"

Seg::Seg(){
  //optical_flow_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, true, 30); // 30
  optical_flow_ = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 5, 150, 10); // 가장좋은결과.
  //opticalflow_   = cv::cuda::OpticalFlowDual_TVL1::create();
  //opticalflow_ = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13,13),3);
}


bool Seg::IsKeyframe(cv::Mat flow, cv::Mat rgb) {
  const float flow_threshold = 20.;
  bool is_keyframe = false;
  // Keyframe 판정을 위한 'Median flow' 획득.
  std::vector<float> norm_flows;
  norm_flows.reserve(corners0_.size());
  for(const cv::Point2f& pt0 : corners0_){
    const cv::Point2f& dpt = flow.at<cv::Point2f>(pt0);
    if(dpt.x==0. && dpt.y==0.)
      continue;
    norm_flows.push_back(cv::norm(dpt));
  }
  std::sort(norm_flows.begin(),norm_flows.end(), std::greater<int>());
  int n = .2 * norm_flows.size();
  float median = norm_flows.at(n);
  is_keyframe = median > flow_threshold;
  if(!rgb.empty()){
    cv::Mat dst = rgb.clone();
    const auto& color = is_keyframe?CV_RGB(255,0,0):CV_RGB(0,0,255);
    for(const cv::Point2f& pt0 : corners0_){
      const cv::Point2f& dpt = flow.at<cv::Point2f>(pt0);
      cv::arrowedLine(dst, pt0, pt0+dpt, color, 1); // verbose
    }
    cv::imshow("FeatureTrack", dst);//verbose
  }
  return is_keyframe;
}

void Seg::PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray, const g2o::SE3Quat& Tcw){
  {
    // ref: https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
    int maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack(gray, corners0_, maxCorners, qualityLevel, minDistance, cv::Mat(),
                            blockSize, gradientSize, useHarrisDetector,k);
  }
  if(corners0_.empty())
    return;
  gray0_ = gray;
  g_gray0_ = g_gray;
  Tc0w_ = Tcw;
  return;
}

void Seg::Put(cv::Mat gray, cv::Mat gray_r, const g2o::SE3Quat& Tcw, const StereoCamera& camera) {
  if(camera.GetD().norm() > 1e-5){
    std::cerr << "Not support distorted image. Put rectified image" << std::endl;
    exit(1);
  }
  auto start = std::chrono::steady_clock::now();

  int search_range = 0;
  int block_size = 21;

  cv::Mat rgb;
  cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);

  cv::cuda::GpuMat g_gray;
  g_gray.upload(gray);
  if(gray0_.empty()){
    PutKeyframe(gray, g_gray, Tcw);
    return;
  }

  cv::Mat flow;
  cv::cuda::GpuMat g_flow;
  // ref) https://android.googlesource.com/platform/external/opencv3/+/master/samples/gpu/optical_flow.cpp#189
  if(optical_flow_->getDefaultName() == "DenseOpticalFlow.BroxOpticalFlow"){
    cv::cuda::GpuMat g_f_gray,g_f_gray0;
    g_gray0_.convertTo(g_f_gray0,CV_32FC1, 1./255.);
    g_gray.convertTo(g_f_gray,CV_32FC1, 1./255.);
    //optical_flow_->calc(g_f_gray0, g_f_gray, g_flow);
    optical_flow_->calc(g_f_gray, g_f_gray0, g_flow);
  }
  else{
    //optical_flow_->calc(g_gray0_, g_gray, g_flow);
    optical_flow_->calc(g_gray, g_gray0_, g_flow);
  }
  g_flow.download(flow); // flow : 2ch float image
  flow = -flow;

#if 1
  const bool is_keyframe = true;
#else
  // Worse result with keyframe
  const bool is_keyframe = IsKeyframe(flow,rgb);
  if(!is_keyframe){
    cv::waitKey(1);
    return;
  }
#endif

  cv::cuda::GpuMat g_gray_r;
  g_gray_r.upload(gray_r);
  cv::Mat disparity = GetDisparity(g_gray,g_gray_r);
  g2o::SE3Quat Tc0c1 = Tc0w_ * Tcw.inverse();
  cv::Mat exp_flow = GetExpectedFlow(camera, Tc0c1, disparity);
  cv::Mat flow_scale; {
    std::vector<cv::Mat> tmp;
    cv::split(exp_flow,tmp);
    flow_scale = cv::abs(tmp[0]) + cv::abs(tmp[1]);
  }

  cv::Mat flow_errors = GetFlowError(flow, exp_flow);
  //cv::Mat divergence = GetDivergence(flow);
  cv::Mat flow_difference = GetDifference(flow, disparity);

  for(int r=0; r<flow_difference.rows; r++) {
    for(int c=0; c<flow_difference.cols; c++) {
      float& diff = flow_difference.at<float>(r,c);
      const float& s = flow_scale.at<float>(r,c);
      const float S = std::min<float>(10.f,s);
      //const float S = 5.;
      const float& disp = disparity.at<float>(r,c);
      float& err = flow_errors.at<float>(r,c);
      if(s > 1. && disp > 1.){
        diff /= S;
        err /= S;
      }
      else{
        diff = 0.;
        err = 0.;
      }
    }
  }

  cv::Mat texture_edge; {
#if 1
    cv::Canny(gray, texture_edge, 100, 200);
#else
    cv::cuda::GpuMat g1,g2,g3, g_all;
    static auto filter1 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,1,1);
    static auto filter2 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,1,0);
    static auto filter3 = cv::cuda::createSobelFilter(CV_8UC1,CV_32FC1,0,1);
    filter1->apply(g_gray, g1);
    filter2->apply(g_gray, g2);
    filter3->apply(g_gray, g3);
    cv::Mat s1,s2,s3;
    g1.download(s1);
    g2.download(s2);
    g3.download(s3);
    /*
    sobel = cv::abs(s1)+cv::abs(s2)+cv::abs(s3);
    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(sobel, &minValue, &maxValue, &minLoc, &maxLoc);
    sobel = sobel > .1*maxValue;
    */
    float threshold = 100.;
    cv::bitwise_or(cv::abs(s1)>threshold, cv::abs(s2)>threshold, texture_edge);
    cv::bitwise_or(texture_edge, cv::abs(s3)>threshold, texture_edge);
#endif
  }


  {
    cv::Mat flow_dist;
    cv::distanceTransform(flow_difference < .5, flow_dist, cv::DIST_L2, cv::DIST_MASK_3);

    cv::Mat output_edges = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    for(int r=0; r<output_edges.rows; r++){
      for(int c=0; c<output_edges.cols;c++){
        const uchar& te = texture_edge.at<uchar>(r,c);
        if(!te)
          continue;
        float R = 1.*disparity.at<float>(r,c);
        if(flow_dist.at<float>(r,c) > R)
          continue;
        output_edges.at<uchar>(r,c) = 1;
      }
    }
    cv::imshow("output", 255*output_edges);
    //cv::imshow("flow_dist", 255*(flow_dist<5.));
    //cv::Mat texture_dist;
    //cv::distanceTransform(texture_edge < 1, texture_dist, cv::DIST_L2, cv::DIST_MASK_3);
    //cv::imshow("texture_dist", 255*(texture_dist<5.) );
    cv::Mat zeros = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {zeros,flow_dist<2.,texture_edge>0};
    cv::Mat dst;
    cv::merge(vec, dst);
    dst *= 255;
    cv::imshow("thick_edges", dst);
  }

  //{
  //  cv::Mat edges = Score2Binary(flow_difference);
  //  cv::imshow("edges", edges);
  //}


  {
    cv::Mat n_sobel;
    cv::normalize(texture_edge, n_sobel, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("sobel", n_sobel);

    cv::Mat div_norm = convertMat(flow_difference, 0., 1.);
    //cv::Mat div_norm; cv::normalize(divergence, div_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("div_dst", div_norm);
    cv::Mat zeros = cv::Mat::zeros(texture_edge.rows, texture_edge.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {n_sobel,div_norm, zeros};
    cv::Mat dst;
    cv::merge(vec, dst);
    cv::imshow("scores", dst);
  }
  if(true){
    //cv::Mat dst;
    //cv::Mat tmp = VisualizeFlow(exp_flow - flow);
    //cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    //cv::pyrDown(dst,dst);
    //cv::imshow("flow_errors", dst);

    cv::Mat err_norm = convertMat(flow_errors, 0., 1.);
    cv::pyrDown(err_norm, err_norm);
    cv::imshow("nflow_errors", err_norm);
  }
  if(false){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(exp_flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    //cv::pyrDown(dst,dst);
    cv::imshow("exp_flow", dst);
  }

  if(true){
    cv::Mat dst;
    disparity.convertTo(dst, CV_8UC1);
    cv::imshow("disp",  dst);
    /*
    cv::Mat ndisp;
    cv::normalize(disparity, ndisp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::pyrDown(ndisp,ndisp);
    cv::imshow("disp", ndisp);
    */
  }
  if(true){
    cv::Mat dst;
    cv::Mat tmp = VisualizeFlow(flow);
    cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
    //cv::pyrDown(dst,dst);
    cv::imshow("flow", dst);
  }

  if(is_keyframe)
    PutKeyframe(gray, g_gray, Tcw);

  auto end = std::chrono::steady_clock::now();
  auto etime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "etime = " << etime_ms << std::endl;
  return;
}
