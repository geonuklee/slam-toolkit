#include "../include/util.h"

std::vector<cv::Scalar> colors = {
  CV_RGB(0,180,0),
  CV_RGB(0,100,0),
  CV_RGB(255,0,255),
  CV_RGB(100,0,255),
  CV_RGB(100,0,100),
  CV_RGB(0,0,180),
  CV_RGB(0,0,100),
  CV_RGB(255,255,0),
  CV_RGB(100,255,0),
  CV_RGB(100,100,0),
  CV_RGB(100,0,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};

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
  int dl = 5;
  float dth = 20.* M_PI/180.;
  for(int r=dl; r+dl<divergence.rows; r++) {
    for(int c=dl; c+dl<divergence.cols; c++) {
      const float& disp = disparity.at<float>(r,c);
      float& div = divergence.at<float>(r,c);
#if 1
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

void GetExpectedFlow(const StereoCamera& camera, const g2o::SE3Quat& Tc0c1, const cv::Mat disp1,
                     cv::Mat& exp_flow, cv::Mat& valid_mask) {
  const Eigen::Matrix<double,3,3> K = camera.GetK();
  const float baseline = - camera.GetTrl().translation().x();
  assert(baseline>0.);

  if(valid_mask.empty())
    valid_mask = cv::Mat::ones(disp1.rows, disp1.cols, CV_8UC1);

  exp_flow = cv::Mat::zeros(disp1.rows, disp1.cols, CV_32FC2);
  for(int r=0;  r<exp_flow.rows; r++){
    for(int c=0; c<exp_flow.cols; c++){
      const double du = disp1.at<float>(r,c);
      if(du < 1.){
        valid_mask.at<uchar>(r,c) = false;
        continue;
      }
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
  return;
}

cv::Mat GetFlowError(const cv::Mat flow, const cv::Mat exp_flow, cv::Mat& valid_mask) {
#if 1
  // min pooling Error
  cv::Mat output = cv::Mat::zeros(flow.rows,flow.cols, CV_32FC2);
  const int R = 10;
  const int step=2;
  const float min_flow = 1.;
  for(int r0=0; r0<flow.rows; r0++){
    for(int c0=0; c0<flow.cols; c0++){
      if( !valid_mask.at<uchar>(r0,c0) )
        continue;
      const auto& F0 = flow.at<cv::Point2f>(r0,c0);
      cv::Point2f& vec = output.at<cv::Point2f>(r0,c0);
      bool valid = false;
      float min_err = 100.;
      for(int r1=std::max(0,r0-R); r1<std::min(flow.rows,r0+1+R); r1+=step){
        for(int c1=std::max(0,c0-R); c1<std::min(flow.cols,c0+1+R); c1+=step){
          if( !valid_mask.at<uchar>(r1,c1) )
            continue;
          const auto& F1 = exp_flow.at<cv::Point2f>(r1,c1);
          float err = std::abs(F1.x-F0.x) + std::abs(F1.y-F0.y);
          if(err > min_err)
            continue;
          valid = true;
          min_err = err;
          vec = F0-F1;
        }
      }
      valid_mask.at<uchar>(r0,c0) = true;
    }
  }
  return output;
#else
  std::vector<cv::Mat> f0, f1;
  cv::split(flow,f0);
  cv::split(exp_flow,f1);
  return cv::abs( f1[0]-f0[0] ) + cv::abs( f1[1]-f0[1] );
#endif
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
  */
  if(false){
    static auto sbp = cv::cuda::createStereoBeliefPropagation();
    sbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  else{
    int ndisp=128;
    int iters=8;
    int levels=4;
    int nr_plane=4;
    static auto csbp = cv::cuda::createStereoConstantSpaceBP(ndisp,iters,levels,nr_plane);
    csbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  disparity.convertTo(disparity, CV_32FC1);
  return disparity;
}

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text){
  cv::Mat dst = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);
  std::map<int, cv::Point> annotated_lists;

  cv::Mat connected_labels, stats, centroids;
  cv::Mat binary = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(idx > 1)
        binary.at<unsigned char>(i,j) = 1;
    }
  }
  cv::connectedComponentsWithStats(binary, connected_labels, stats, centroids, 4);

  for(int i=0; i<stats.rows; i++) {
    int x = centroids.at<double>(i, 0);
    int y = centroids.at<double>(i, 1);
    if(x < 0 or y < 0 or x >= mask.cols or y >= mask.cols)
      continue;

    int idx;
    if(mask.type() == CV_8UC1)
      idx = mask.at<unsigned char>(y,x);
    else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
      idx = mask.at<int>(y,x);

    if(idx > 1 && !annotated_lists.count(idx) ){
      bool overlaped=false;
      cv::Point pt(x,y);
      for(auto it : annotated_lists){
        cv::Point e(pt - it.second);
        if(std::abs(e.x)+std::abs(e.y) < 20){
          overlaped = true;
          break;
        }
      }
      if(!overlaped)
        annotated_lists[idx] = pt;
    }
  }

  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(mask.type() == CV_8UC1 && idx == 0)
        continue;
      else if(mask.type() == CV_32S && idx < 0)
        continue;

      cv::Scalar bgr;
      if( idx == 0)
        bgr = CV_RGB(100,100,100);
      //else if (idx == 1)
      //  bgr = CV_RGB(255,255,255);
      else
        bgr = colors.at( idx % colors.size() );

      dst.at<cv::Vec3b>(i,j)[0] = bgr[0];
      dst.at<cv::Vec3b>(i,j)[1] = bgr[1];
      dst.at<cv::Vec3b>(i,j)[2] = bgr[2];

      if(idx > 1 && !annotated_lists.count(idx) ){
        bool overlaped=false;
        cv::Point pt(j,i+10);
        for(auto it : annotated_lists){
          cv::Point e(pt - it.second);
          if(std::abs(e.x)+std::abs(e.y) < 20){
            overlaped = true;
            break;
          }
        }
        if(!overlaped)
          annotated_lists[idx] = pt;
      }
    }
  }

  if(put_text){
    for(auto it : annotated_lists){
      //cv::rectangle(dst, it.second+cv::Point(0,-10), it.second+cv::Point(20,0), CV_RGB(255,255,255), -1);
      const auto& c0 = colors.at( it.first % colors.size() );
      //const auto color = c0;
      const auto color = (c0[0]+c0[1]+c0[2] > 255*2) ? CV_RGB(0,0,0) : CV_RGB(255,255,255);
      cv::putText(dst, std::to_string(it.first), it.second, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
  }
  return dst;
}
