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

#include "stdafx.h"
#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
//#include "pipeline.h"
//#include "qmap_viewer.h"
//#include "common.h"
//#include <QApplication>
//#include <QWidget>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>

cv::Mat VisualizeFlow(cv::Mat flow, cv::Mat bgr=cv::Mat() ) {
  // Visualization part
#if 1
  cv::Mat dst = bgr.empty()? cv::Mat::zeros(flow.rows,flow.cols,CV_8UC3) : bgr.clone();
  const int step = 20;
  for(int r=step; r+step < flow.rows; r+=step){
    for(int c=step; c+step < flow.cols; c+=step){
      cv::Point2f pt0(c,r);
      const cv::Point2f& F = flow.at<cv::Point2f>(r,c);
      float d = cv::norm(F);
      if( d > step)
        d = step;
      cv::Point2f pt1 = pt0+d/cv::norm(F) * F;
      cv::arrowedLine(dst,pt0,pt1,CV_RGB(0,0,255),1,cv::LINE_4,0,.5);
    }
  }
  return dst;
#else
  cv::Mat flow_parts[2];
  cv::split(flow, flow_parts);
  // Convert the algorithm's output into Polar coordinates
  cv::Mat magnitude, angle, magn_norm;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
  angle *= ((1.f / 360.f) * (180.f / 255.f));

  // Build hsv image
  cv::Mat _hsv[3], hsv, hsv8, bgr;
  //_hsv[0] = angle;
  _hsv[0] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[1] = angle;
  //_hsv[2] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magn_norm;
  merge(_hsv, 3, hsv);
  hsv.convertTo(hsv8, CV_8U, 255.0);
  cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
  return bgr;
#endif
}

cv::Mat GetDivergence(cv::Mat flow){
#if 1
  int dl = 5; // Half of sample window for differenctiation
  std::vector<cv::Point2i> samples = {
    cv::Point2i(4,0),
    cv::Point2i(4,1),
    cv::Point2i(1,4),
    cv::Point2i(0,4),

    cv::Point2i(3,0),
    cv::Point2i(3,1),
    cv::Point2i(2,2),
    cv::Point2i(1,3),
    cv::Point2i(0,3),

    cv::Point2i(2,0),
    cv::Point2i(2,1),
    cv::Point2i(1,2),
    cv::Point2i(0,2),

    cv::Point2i(1,0),
    cv::Point2i(1,1),
    cv::Point2i(0,1),
  };
  //for(auto& dpt : samples)
  //  dpt *= 2;
  //dl *= 2;
  const float max_d = 50.;

  cv::Mat div = cv::Mat::zeros(flow.rows,flow.cols,CV_32F);
  std::vector<cv::Point2f> vec_dpt2;
  std::vector<float> vec_dpt2norm;
  vec_dpt2.reserve(samples.size());
  vec_dpt2norm.reserve(samples.size());
  for(const auto& dpt : samples){
    cv::Point2f dpt2 = 2.*dpt;
    vec_dpt2.push_back(dpt2);
    vec_dpt2norm.push_back(cv::norm(dpt2));
  }

  for(int r = dl; r+dl < flow.rows; r++){
    for(int c = dl; c+dl < flow.cols; c++){
      //for(const auto& dpt : samples){
      float& div_pixel = div.at<float>(r,c);
      float n = 0.;
      for(size_t i = 0; i < samples.size(); i++){
        const auto& dpt = samples.at(i);
        cv::Point2i pt0 = cv::Point2i(c,r)-dpt;
        cv::Point2i pt1 = cv::Point2i(c,r)+dpt;
        const cv::Point2f& F0 = flow.at<cv::Point2f>(pt0);
        if(F0.x == 0 && F0.y == 0)
          continue;
        const cv::Point2f& F1 = flow.at<cv::Point2f>(pt1);
        if(F1.x == 0 && F1.y == 0)
          continue;
        const auto& dpt2 = vec_dpt2.at(i);
        const auto& dpt2norm = vec_dpt2norm.at(i);
        float d =  (F1-F0).dot(dpt2) / dpt2norm;
        div_pixel += d;
        n += 1.;
      } // for i
      // TODO Median? Mean?
      div_pixel /= n;
      if(div_pixel > max_d)
        div_pixel = max_d;
      else if(div_pixel < -max_d)
        div_pixel = -max_d;
      div_pixel = std::abs(div_pixel);
    }
  }

#else
  cv::Mat div = cv::Mat::zeros(flow.rows,flow.cols,CV_32F);
  const int dl = 5; // Half of sample window for differenctiation
  const float fdl2 = 2.* (float)dl;
  const float dV = dl*dl-1.;
  const float max_d = 10.;
  for(int r0 = dl; r0+dl < flow.rows; r0++){
    for(int c0 = dl; c0+dl < flow.cols; c0++){
      // Get 'F'lux
      const float& Fx0 = flow.at<cv::Point2f>(r0-dl,c0).x;
      const float& Fx1 = flow.at<cv::Point2f>(r0+dl,c0).x;
      const float& Fy0 = flow.at<cv::Point2f>(r0,c0-dl).y;
      const float& Fy1 = flow.at<cv::Point2f>(r0,c0+dl).y;
      if(Fx0 == 0. || Fy0 == 0.)
        continue;;
      if(Fx1 == 0. || Fy1 == 0.)
        continue;;
      // Gauss' divergence theorem
      //float d = (std::abs(Fx1-Fx0) + std::abs(Fy1-Fy0))/fdl2;
      float d = (Fx1-Fx0 + Fy1-Fy0)/fdl2;
      if( d > max_d)
        d = max_d;
      else if ( d< -max_d)
        d = - max_d;
      div.at<float>(r0,c0) = std::abs(d);
    }
  }
#endif
  return div;
}

class Seg{
public:
  Seg(){
    //opticalflow_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, true, 20); // 30
    opticalflow_ = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 5, 150, 10); // 가장좋은결과.
    //opticalflow_   = cv::cuda::OpticalFlowDual_TVL1::create();
    //opticalflow_ = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13,13),3);
  }

  bool IsKeyframe(cv::Mat flow, cv::Mat rgb = cv::Mat()){
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

  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray){
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
    return;
  }

  void Put(cv::Mat gray){
    cv::Mat rgb;
    cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);

    cv::cuda::GpuMat g_gray;
    g_gray.upload(gray);
    if(gray0_.empty()){
      PutKeyframe(gray, g_gray);
      return;
    }

    cv::Mat flow;
    cv::cuda::GpuMat g_flow;
    // ref) https://android.googlesource.com/platform/external/opencv3/+/master/samples/gpu/optical_flow.cpp#189
    if(opticalflow_->getDefaultName() == "DenseOpticalFlow.BroxOpticalFlow"){
      cv::cuda::GpuMat g_f_gray,g_f_gray0;
      g_gray0_.convertTo(g_f_gray0,CV_32FC1, 1./255.);
      g_gray.convertTo(g_f_gray,CV_32FC1, 1./255.);
      opticalflow_->calc(g_f_gray0, g_f_gray, g_flow);
    }
    else
      opticalflow_->calc(g_gray0_, g_gray, g_flow);
    g_flow.download(flow); // flow : 2ch float image

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

    cv::Mat div = GetDivergence(flow);
    cv::Mat div_norm;
    cv::normalize(div, div_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat binary_div;
    cv::threshold(div, binary_div, 3., 255, cv::THRESH_BINARY);
    binary_div.convertTo(binary_div,CV_8UC1);

    if(true){
      cv::Mat zero = cv::Mat::zeros(gray.rows,gray.cols,CV_8UC1);
      std::vector<cv::Mat> vec = {zero,div_norm,div_norm};
      cv::Mat tmp,dst;
      cv::merge(vec,tmp);
      cv::addWeighted(rgb,1.,tmp,1.,1.,dst);
      cv::imshow("div_dst", dst);
    }

    if(true){
      cv::Mat zero = cv::Mat::zeros(gray.rows,gray.cols,CV_8UC1);
      std::vector<cv::Mat> vec = {zero,zero,binary_div};
      cv::Mat tmp,dst;
      cv::merge(vec,tmp);
      cv::addWeighted(rgb,1.,tmp,1.,1.,dst);
      cv::imshow("edge_dst", dst);
    }

    //cv::imshow("src", gray0_);
    if(true){
      cv::Mat dst;
      //cv::Mat tmp = VisualizeFlow(flow, rgb);
      cv::Mat tmp = VisualizeFlow(flow);
      cv::addWeighted(rgb,.2,tmp,1.,1.,dst);
      cv::imshow("flow", dst);
    }

    if(is_keyframe)
      PutKeyframe(gray, g_gray);
    return;
  }

private:
  cv::Ptr<cv::cuda::DenseOpticalFlow> opticalflow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;
  std::vector<cv::Point2f> corners0_;

};


int main(int argc, char** argv){
 //suggest "13" "20";
  std::string seq(argv[1]);
  std::string dir_im = "/home/geo/dataset/kitti_odometry_dataset/sequences/"+seq+"/image_0/";
  std::string dir_im_r = "/home/geo/dataset/kitti_odometry_dataset/sequences/"+seq+"/image_1/";
  Eigen::Matrix<double,3,3> K;
  K << 707.1, 0., 601.89, 0., 707.1, 183.1, 0., 0., 1.;
  char buff[100];
  bool stop = true;

  Seg seg;

  for(int i = 0; ; i+=1){
    snprintf(buff, sizeof(buff), "%06d.png", i);
    std::string fn_im = dir_im + std::string(buff);
    std::string fn_im_r = dir_im_r + std::string(buff);
    cv::Mat gray = cv::imread(fn_im, cv::IMREAD_GRAYSCALE);
    //cv::Mat gray_r = cv::imread(fn_im_r, cv::IMREAD_GRAYSCALE);
    if(gray.empty())
      break;
    seg.Put(gray);

    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }

  return 1;
}
