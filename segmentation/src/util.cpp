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

void GetExpectedFlow(const Camera& camera, const g2o::SE3Quat& Tc0c1, const cv::Mat depth,
                     cv::Mat& exp_flow, cv::Mat& valid_mask) {
  const Eigen::Matrix<double,3,3> K = camera.GetK();
  exp_flow = cv::Mat::zeros(valid_mask.rows, valid_mask.cols, CV_32FC2);
  for(int r=0;  r<exp_flow.rows; r++){
    for(int c=0; c<exp_flow.cols; c++){
      Eigen::Vector2d uv1( (double)c, (double)r );
      double z1 = depth.at<float>(r,c);
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

  std::set<int> labels;

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

      labels.insert(idx);
    }
  }

  if(put_text){
    for(int l : labels){
      cv::Mat fg = mask == l;
      cv::rectangle(fg, cv::Rect(cv::Point(0,0), cv::Point(fg.cols-1,fg.rows-1)), false, 2);

      cv::Mat dist;
      cv::distanceTransform(fg, dist, cv::DIST_L2, cv::DIST_MASK_3);
      double minv, maxv;
      cv::Point minloc, maxloc;
      cv::minMaxLoc(dist, &minv, &maxv, &minloc, &maxloc);
      const auto& c0 = colors.at( l % colors.size() );
      const auto color = (c0[0]+c0[1]+c0[2] > 255*2) ? CV_RGB(0,0,0) : CV_RGB(255,255,255);
      cv::putText(dst, std::to_string(l), maxloc, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
  }
  return dst;
}

cv::Mat GetBoundary(const cv::Mat marker, int w){
  cv::Mat boundarymap = cv::Mat::zeros(marker.rows,marker.cols, CV_8UC1);
  for(int r0 = 0; r0 < marker.rows; r0++){
    for(int c0 = 0; c0 < marker.cols; c0++){
      const int& i0 = marker.at<int>(r0,c0);
      bool b = false;
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,marker.cols); c1++){
          const int& i1 = marker.at<int>(r1,c1);
          b = i0 != i1;
          if(b)
            break;
        }
        if(b)
          break;
      }
      if(!b)
        continue;
      boundarymap.at<unsigned char>(r0,c0) = true;
    }
  }
  return boundarymap;
}

