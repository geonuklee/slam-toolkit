#include "../include/util.h"

cv::Mat VisualizeFlow(cv::Mat flow, cv::Mat bgr) {
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

cv::Mat convertDiscrete(cv::Mat input, float istep, int ostep) {
  cv::Mat output(input.rows, input.cols, CV_8UC1);
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      const float& vi = input.at<float>(i, j);
      uchar vo = std::min<float>(254., ostep*(vi/istep) );
      output.at<uchar>(i,j) = vo;
    }
  }
  return output;
}

cv::Mat convertMat(cv::Mat input, float lower_bound, float upper_bound) {
  cv::Mat output(input.rows, input.cols, CV_8UC1);
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      const float& value = input.at<float>(i, j);
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


cv::Mat GetDifference(cv::Mat flow, cv::Mat disparity) {
  cv::Mat divergence = cv::Mat::zeros(flow.rows,flow.cols,CV_32F);
  int dl = 50;
  float dth = 20.* M_PI/180.;
  for(int r=dl; r+dl<divergence.rows; r++) {
    for(int c=dl; c+dl<divergence.cols; c++) {
      const float& disp = disparity.at<float>(r,c);
      float& div = divergence.at<float>(r,c);
#if 0
      float R = std::min<float>(dl, .5*disp );
      R       = std::max<float>(R, 5.);
#else
      const float R = 5.;
#endif
      cv::Point2i dpt0(r,r);
      float n = 0.;
      cv::Point2i pt0(c,r);
      for(float th=0.; th<2.*M_PI-dth; th+=dth){
        cv::Point2i dpt(R*std::cos(th), R*std::sin(th));
        if(dpt == dpt0)
          continue;
        dpt0 = dpt;
        cv::Point2i pt1 = pt0+dpt;
        const cv::Point2f& F0 = flow.at<cv::Point2f>(pt0);
        const cv::Point2f& F1 = flow.at<cv::Point2f>(pt1);
        const float d =  cv::norm(F1-F0);
        div += d;
        n += 1.;
      }
      div /= n;
      div = std::abs(div);
    }
  }
  return divergence;
}

cv::Mat GetDivergence(cv::Mat flow) {
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

cv::Mat GetFlowError(const cv::Mat flow, const cv::Mat exp_flow) {
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

