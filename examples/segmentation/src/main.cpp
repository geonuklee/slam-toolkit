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
  cv::Mat dst = bgr.empty()? cv::Mat::zeros(flow.rows,flow.cols,CV_8UC3) : bgr.clone();
  const int step = 20;
  for(int r=step; r+step < flow.rows; r+=step){
    for(int c=step; c+step < flow.cols; c+=step){
      cv::Point2f pt0(c,r);
      const cv::Point2f F = flow.at<cv::Point2f>(r,c);
      float d = cv::norm(F);
      if( d > step)
        d = step;
      else if(d < 1.){
        cv::circle(dst, pt0, 3, CV_RGB(255,0,0) );
        continue;
      }
      cv::Point2f pt1 = pt0+d/cv::norm(F) * F;
      cv::arrowedLine(dst,pt0,pt1,CV_RGB(0,0,255),1,cv::LINE_4,0,.5);
    }
  }
  return dst;
}

cv::Mat GetDivergence(cv::Mat flow){
  int dl = 5; // Half of sample window for differenctiation
  std::vector<cv::Point2i> samples = {
    //cv::Point2i(4,0),
    //cv::Point2i(4,1),
    //cv::Point2i(1,4),
    //cv::Point2i(0,4),

    cv::Point2i(3,0),
    cv::Point2i(3,1),
    cv::Point2i(2,2),
    cv::Point2i(1,3),
    cv::Point2i(0,3),

    //cv::Point2i(2,0),
    //cv::Point2i(2,1),
    //cv::Point2i(1,2),
    //cv::Point2i(0,2),

    //cv::Point2i(1,0),
    //cv::Point2i(1,1),
    //cv::Point2i(0,1),
  };
  //for(auto& dpt : samples)
  //  dpt *= 2;
  //dl *= 2;
  // const float max_d = 50.;

  cv::Mat divergency = cv::Mat::zeros(flow.rows,flow.cols,CV_32F);
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
      float& div = divergency.at<float>(r,c);
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
        div += d;
        n += 1.;
      } // for i
      // TODO Median? Mean?
      div /= n;
      //if(div_pixel > max_d)
      //  div_pixel = max_d;
      //else if(div_pixel < -max_d)
      //  div_pixel = -max_d;
      div = std::abs(div);
    }
  }

  return divergency;
}

cv::Mat convertMat(cv::Mat input, float lower_bound, float upper_bound) {
  cv::Mat output(input.rows, input.cols, CV_8UC1);
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      float value = input.at<float>(i, j);
      if (value < lower_bound)
        output.at<uchar>(i, j) = 0; // Set element as 0
      else if (value > upper_bound)
        output.at<uchar>(i, j) = 255; // Set element as 255
      else
        // Normalize the value to the range [0, 255]
        output.at<uchar>(i, j) = static_cast<uchar>((value - lower_bound) * (255.0 / (upper_bound - lower_bound)));
    }
  }
  return output;
}

cv::Mat GetExpectedFlow(const StereoCamera& camera,
                     const g2o::SE3Quat& Tc0c1,
                     cv::Mat disp1){
  const Eigen::Matrix<double,3,3> K = camera.GetK();
  const float baseline = - camera.GetTrl().translation().x();
  assert(baseline>0.);

  cv::Mat exp_flow = cv::Mat::zeros(disp1.rows, disp1.cols, CV_32FC2);
  for(int r=0;  r<exp_flow.rows; r++){
    for(int c=0; c<exp_flow.cols; c++){
      const double du = disp1.at<float>(r,c);
      if(du < 1.)
        continue;
      Eigen::Vector2d uv1( (double)c, (double)r );
      double z1 = K(0,0) * baseline / du;
      Eigen::Vector3d X1( z1 * (uv1[0]-K(0,2))/K(0,0),
                          z1 * (uv1[1]-K(1,2))/K(1,1),
                          z1);
      Eigen::Vector3d uv0 = K*  (Tc0c1 * X1);
      uv0.head<2>() /= uv0[2];
      Eigen::Vector2d f = uv1 - uv0.head<2>();
      exp_flow.at<cv::Vec2f>(r,c) = cv::Vec2f(f[0],f[1]);
    }
  }
  return exp_flow;
}

cv::Mat GetFlowError(const cv::Mat flow,
                     const cv::Mat exp_flow){
  std::vector<cv::Mat> f0, f1;
  cv::split(flow,f0);
  cv::split(exp_flow,f1);
  return cv::abs( f1[0]-f0[0] ) + cv::abs( f1[1]-f0[1] );
}

cv::Mat GetDisparity(cv::cuda::GpuMat g_gray, cv::cuda::GpuMat g_gray_r){
  cv::cuda::GpuMat  g_disp;
  cv::Mat disparity;
  /*
  {
    static auto bm = cv::cuda::createStereoBM();
    bm->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disp); // 8UC
  }
  if(false){
    static auto sbp = cv::cuda::createStereoBeliefPropagation();
    sbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  */
  if(true){
    static auto csbp = cv::cuda::createStereoConstantSpaceBP();
    csbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  disparity.convertTo(disparity, CV_32FC1);
  return disparity;
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

  void PutKeyframe(cv::Mat gray, cv::cuda::GpuMat g_gray, const g2o::SE3Quat& Tcw){
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

  void Put(cv::Mat gray, cv::Mat gray_r, const g2o::SE3Quat& Tcw, const StereoCamera& camera){
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

    cv::cuda::GpuMat g_gray_r;
    g_gray_r.upload(gray_r);
    cv::Mat disparity = GetDisparity(g_gray,g_gray_r);
    g2o::SE3Quat Tc0c1 = Tc0w_ * Tcw.inverse();
    cv::Mat exp_flow = GetExpectedFlow(camera, Tc0c1, disparity);
    cv::Mat flow_scale; {
      // TODO Flow Scale은 exp_flow 뿐만 아니라, flow가 0인 경우도 제외하고, disp가 0인 경우도 제외해서 정해야함.
      std::vector<cv::Mat> tmp;
      cv::split(exp_flow,tmp);
      flow_scale = cv::abs(tmp[0]) + cv::abs(tmp[1]);
    }

    cv::Mat flow_errors = GetFlowError(flow, exp_flow);
    cv::Mat divergence = GetDivergence(flow);
    for(int r=0; r<divergence.rows; r++) {
      for(int c=0; c<divergence.cols; c++) {
        float& div = divergence.at<float>(r,c);
        const float& s = flow_scale.at<float>(r,c);
        const float S = s*s;
        const float& disp = disparity.at<float>(r,c);
        float& err = flow_errors.at<float>(r,c);
        if(s > 1. && disp > 1.){
          div /= S;
          err /= S;
        }
        else{
          div = 0.;
          err = 0.;
        }
      }
    }


    /*
    cv::Mat binary_div;
    cv::threshold(div, binary_div, 3., 255, cv::THRESH_BINARY);
    binary_div.convertTo(binary_div,CV_8UC1);

    if(true){
      cv::Mat zero = cv::Mat::zeros(gray.rows,gray.cols,CV_8UC1);
      std::vector<cv::Mat> vec = {zero,zero,binary_div};
      cv::Mat tmp,dst;
      cv::merge(vec,tmp);
      cv::addWeighted(rgb,1.,tmp,1.,1.,dst);
      cv::imshow("edge_dst", dst);
    }
    */

    {
      cv::Mat div_norm = convertMat(divergence, 0., .05);
      cv::imshow("div_dst", div_norm);
    }
    if(true){
      cv::Mat dst;
      cv::Mat tmp = VisualizeFlow(exp_flow - flow);
      cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
      cv::pyrDown(dst,dst);
      cv::imshow("flow_errors", dst);

      cv::Mat err_norm = convertMat(flow_errors, 0., 1.);
      cv::pyrDown(err_norm, err_norm);
      cv::imshow("nflow_errors", err_norm);
    }
    if(true){
      cv::Mat dst;
      cv::Mat tmp = VisualizeFlow(exp_flow);
      cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
      cv::pyrDown(dst,dst);
      cv::imshow("exp_flow", dst);
    }

    if(true){
      cv::Mat ndisp;
      cv::normalize(disparity, ndisp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      cv::pyrDown(ndisp,ndisp);
      cv::imshow("disp", ndisp);
    }
    if(true){
      cv::Mat dst;
      cv::Mat tmp = VisualizeFlow(flow);
      cv::addWeighted(rgb,.5,tmp,1.,1.,dst);
      cv::pyrDown(dst,dst);
      cv::imshow("flow", dst);
    }

    if(is_keyframe)
      PutKeyframe(gray, g_gray, Tcw);

    auto end = std::chrono::steady_clock::now();
    auto etime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "etime = " << etime_ms << std::endl;
    return;
  }

private:
  cv::Ptr<cv::cuda::DenseOpticalFlow> opticalflow_;
  cv::Mat gray0_;
  cv::cuda::GpuMat g_gray0_;
  g2o::SE3Quat Tc0w_;
  std::vector<cv::Point2f> corners0_;

};


int main(int argc, char** argv){
  //Seq :"02",("13" "20");
  std::string fn_config = GetPackageDir()+"/config/kitti.yaml";
  cv::FileStorage config(fn_config, cv::FileStorage::READ);
  std::string seq(argv[1]);

  KittiDataset dataset(seq);
  const auto& Tcws = dataset.GetTcws();
  if(Tcws.empty()){
    std::cout << "Seq" << seq << " with no ground truth trajectory." << std::endl;
    exit(1);
  }
  const auto& D = dataset.GetCamera()->GetD();
  std::cout << "Distortion = " << D.transpose() << std::endl;
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  Seg seg;

  bool stop = true;
  for(int i=0; i<dataset.Size(); i+=1){
    const cv::Mat gray   = dataset.GetImage(i);
    const cv::Mat gray_r = dataset.GetRightImage(i);
    const g2o::SE3Quat Tcw = Tcws.at(i);
    seg.Put(gray, gray_r, Tcw, *camera );
    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }

  return 1;
}
