#include "../include/seg.h"
#include "util.h"
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static bool SameSign(const float& v1, const float& v2){
  if(v1 > 0.)
    return v2 > 0.;
  else if(v1 < 0.)
    return v2 < 0.;
  return (v1 == 0.) && (v2 == 0.);
}

namespace WithoutSIMD{
void GetGrad(const cv::Mat depth , const cv::Mat valid_mask,
             float fx, float fy,
             float meter_sample_offset,
             cv::Mat& gradx,
             cv::Mat& grady,
             cv::Mat& valid_grad
             ) {
  gradx = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
  grady = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
  valid_grad = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  // sample offset, [meter]
  const float sfx = meter_sample_offset*fx;
  const float sfy = meter_sample_offset*fy;
#if 1
  cv::Mat dst;
#else
  cv::Mat dst = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC3);
#endif
  for(int r = 0; r < depth.rows; r++) {
    for(int c = 0; c < depth.cols; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& z = depth.at<float>(pt);
      if(z < 1e-3)
        continue;
      const int du = std::max<int>( (int) sfx/z, 1);
      const int dv = std::max<int>( (int) sfy/z, 1);
      cv::Point2i ptx0(c-du,r);
      cv::Point2i ptx1(c+du,r);
      cv::Point2i pty0(c,r-dv);
      cv::Point2i pty1(c,r+dv);
      if(ptx0.x < 0 || !valid_mask.at<uchar>(ptx0) )
        continue;
      if(ptx1.x > depth.cols || !valid_mask.at<uchar>(ptx1) )
        continue;
      if(pty0.y < 0 || !valid_mask.at<uchar>(pty0) )
        continue;
      if(pty1.y > depth.rows || !valid_mask.at<uchar>(pty1) )
        continue;
      const float zx0 = depth.at<float>(ptx0);
      const float zx1 = depth.at<float>(ptx1);
      const float zy0 = depth.at<float>(pty0);
      const float zy1 = depth.at<float>(pty1);
      if(zx0 < 1e-3 || zx1 < 1e-3 || zy0 < 1e-3 || zy1 < 1e-3)
        continue;
      valid_grad.at<uchar>(pt) = true;
      float& gx = gradx.at<float>(pt);
      float& gy = grady.at<float>(pt);
      gx = (zx1 - zx0) / meter_sample_offset;
      gy = (zy1 - zy0) / meter_sample_offset;
      if(dst.empty())
        continue;
      auto& color = dst.at<cv::Vec3b>(pt);
      int val = std::min<int>(255, std::max<int>(0, std::abs(10.*gy) ) );
      if(gy > 0)
        color[0] = val;
      else
        color[1] = val;
      val = std::min<int>(255, std::max<int>(0, std::abs(10.*gx) ) );
      color[2] = val;
    }
  }
  if(!dst.empty())
    cv::imshow("grad", dst);
  return;
}

cv::Mat GetConcaveEdges(const cv::Mat gradx,
                        const cv::Mat grady,
                        const cv::Mat depth,
                        const cv::Mat valid_mask,
                        float fx, float fy,
                        float neg_hessian_threshold
                       ){
  cv::Mat hessian = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);

  // TODO Hessian 의 dl도 gradients 처럼 [meter] unit으로 변경
#if 1
  const float s = .1; // sample offset. [meter]
  const float R = .5; // NMAS range. [meter]
#else
  const int dl = 5; // "Shallow groove"를 찾을것도 아닌데, 간격이 좁을필요가..
  const int R = 10; // NMAS range
  const float fx = camera.GetK()(0,0);
  const float fy = camera.GetK()(1,1);
#endif
  const float sfx = s*fx;
  const float sfy = s*fy;
  const float Rfx = R*fx;
  //const float Rfy = R*camera.GetK()(1,1);

  for(int r=0; r < depth.rows-0; r++) {
    for(int c=0; c < depth.cols-0; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& z = depth.at<float>(pt);
      if(z < 1e-3)
        continue;
      const int du = std::max<int>( (int) sfx/z, 1);
      const int dv = std::max<int>( (int) sfy/z, 1);

      cv::Point2i ptx0(c-du,r);
      cv::Point2i ptx1(c+du,r);
      cv::Point2i pty0(c,r-dv);
      cv::Point2i pty1(c,r+dv);
      if(ptx0.x < 0 || !valid_mask.at<uchar>(ptx0) )
        continue;
      if(ptx1.x > depth.cols || !valid_mask.at<uchar>(ptx1) )
        continue;
      if(pty0.y < 0 || !valid_mask.at<uchar>(pty0) )
        continue;
      if(pty1.y > depth.rows || !valid_mask.at<uchar>(pty1) )
        continue;
      const float zx0 = depth.at<float>(ptx0);
      const float zx1 = depth.at<float>(ptx1);
      const float zy0 = depth.at<float>(pty0);
      const float zy1 = depth.at<float>(pty1);
      if(zx0 < 1e-3 || zx1 < 1e-3 || zy0 < 1e-3 || zy1 < 1e-3)
        continue;

      float hxx = (gradx.at<float>(ptx1) - gradx.at<float>(ptx0) ) / s;
      float hyy = (grady.at<float>(pty1) - grady.at<float>(pty0) ) / s;
      hessian.at<float>(pt) = hxx + hyy;
    }
  }
  std::vector<cv::Point2i> samples = {
    cv::Point2i(1,0),
    cv::Point2i(-1,0),
    cv::Point2i(0,1),
    cv::Point2i(0,-1),
  };

  cv::Mat edges = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  const int ioffset = 5;
  for(int r=ioffset; r < depth.rows-ioffset; r++) {
    for(int c=ioffset; c < depth.cols-ioffset; c++) {
      const cv::Point2i pt(c,r);
      if(!valid_mask.at<uchar>(pt))
        continue;
      const float& z = depth.at<float>(pt);
      if(z < 1e-3)
        continue;
      const float& h = hessian.at<float>(pt);
      int dL = std::max<int>( Rfx/z, 2);
      bool ismax = true;
      for(int dl=dL-1; dl>0; dl--){
        for(const auto& dpt : samples){
          const cv::Point2i pt1 = pt + dl * dpt;
          if(pt1.x < ioffset || pt1.y < ioffset || pt1.x >= depth.cols-ioffset || pt1.y >= depth.rows-ioffset ){
            ismax = false; // 범위 초과할경우 그만둠.
            break;
          }
          const float& h1 = hessian.at<float>(pt1);
          if(std::abs(h1) < std::abs(h))
            continue;
          if(SameSign(h1,h))
            continue;
          ismax = false;
          break;
        }
        if(!ismax)
          break;
      }
      if(!ismax) // NMAS
        continue;
      // Hessian을 고려한 edge 판단
      if( h < neg_hessian_threshold)
        edges.at<uchar>(pt) = 1;
    }
  }
  return edges;
}

cv::Mat FilterThinNoise(const cv::Mat edges){
#if 1
  cv::Mat ones = cv::Mat::ones(7,7,edges.type());
  cv::Mat expanded_outline;
  cv::dilate(255*edges, expanded_outline, ones);
  cv::threshold(expanded_outline,expanded_outline, 200., 255, cv::THRESH_BINARY);
  //cv::imshow("Before filter", 255*edges);
  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(expanded_outline,labels,stats,centroids);
  std::set<int> inliers;
  for(int i = 0; i < stats.rows; i++){
    const int max_wh = std::max(stats.at<int>(i,cv::CC_STAT_WIDTH),
                                stats.at<int>(i,cv::CC_STAT_HEIGHT));
    if(max_wh < 20)
      continue;
    if(stats.at<int>(i,cv::CC_STAT_AREA) < 400)
      continue;
    inliers.insert(i);
  }
  cv::Mat output = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
  for(int r = 0; r < output.rows; r++){
    for(int c = 0; c < output.cols; c++){
      if(edges.at<unsigned char>(r,c) < 1)
        continue;
      const int& l = labels.at<int>(r,c);
      if(inliers.count(l))
        output.at<uchar>(r,c) = 1;
    }
  }
#else
  cv::Mat kernel = cv::Mat::ones(3,3,edges.type());
  cv::Mat output;
  cv::morphologyEx(edges, output, cv::MORPH_OPEN, kernel );
#endif
  return output;
}

cv::Mat GetDDEdges(const cv::Mat depth, const cv::Mat valid_mask,
                   float fx, float fy) {
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  std::vector<cv::Point2i> samples = {
    cv::Point2i(1,0),
    cv::Point2i(-1,0),
    cv::Point2i(0,1),
    cv::Point2i(0,-1),
  };

  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt0(c0,r0);
      if(!valid_mask.at<unsigned char>(pt0))
        continue;
      const float& z0 = depth.at<float>(pt0);
      unsigned char& e = edge.at<unsigned char>(pt0);
      for(int l=1; l < 4; l++){
        for(const auto& dpt : samples){
          const cv::Point2i pt1 = pt0+l*dpt;
          if(pt1.x < 0 || pt1.x >= depth.cols || pt1.y < 0 || pt1.y >= depth.rows)
            continue;
          if(!valid_mask.at<unsigned char>(pt1))
            continue;
          const float& z1 = depth.at<float>(pt1);
          float th = std::max<float>(0.05*z0, 2.);
          //float th = 0.1;
          if(std::abs(z1-z0) < th)
            continue;
          e = true;
          break;
        }
        if(e)
          break;
      }
    }
  }
  return edge;
}

} // namepsace WithoutSIMD

#include <immintrin.h> // sse or avx
namespace WithSIMD {
/*
 * Referenced tutorials, codes 
 *  for SIMD programming : http://supercomputingblog.com/windows/image-processing-with-sse/
 *  for image process    : https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/imgproc/src/segmentation.cpp#L88
 */

void GetDDEdges(const cv::Mat _depth, cv::Mat& _edge){
  int sample_offset = 2;
  if(_edge.empty())
    _edge = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);

  cv::Size size = _depth.size();
  typedef float Real;
  typedef __m256 T;
  const int vstep = sizeof(T) / sizeof(Real);
  const Real* depth = _depth.ptr<Real>();
  uchar* edge = _edge.ptr<uchar>();

  const int dstep = int(_depth.step/sizeof(depth[0]));
  const int estep = int(_edge.step/sizeof(edge[0]));

  T m_ratio = _mm256_set1_ps(0.01);
  T m_d_min = _mm256_set1_ps(0.001); // invalid depth
  T m_th_min = _mm256_set1_ps(2.);
  T m_ps_zero = _mm256_set1_ps(-0.0f); // 주의 :  -0.0f != 0.0f
  T m_i_zero  = _mm256_set1_ps(0.f);
  T m_dcp, m_ath, m_n1, m_n2, m_n3, m_n4, m_adiff, m_edges, m_results;
  long bit_mask;
  int i, j, k;
  for( i = 0; i < sample_offset; i++ ) {
    depth += dstep;
    edge += estep;
  }
  for( i = sample_offset; i < size.height-sample_offset; i++ ) {
    for( j = sample_offset; j < size.width-1-vstep; j+=vstep ) {
      const Real* d_cp = depth + j;
      uchar* e = edge + j;
      m_dcp     = _mm256_loadu_ps(d_cp);
      m_ath     = _mm256_mul_ps(m_dcp, m_ratio);
      m_ath     = _mm256_blendv_ps(m_ath, m_th_min,  _mm256_cmp_ps(m_th_min, m_ath, _CMP_GE_OS) );

      m_n1      = _mm256_loadu_ps(d_cp-sample_offset);
      m_n3      = _mm256_loadu_ps(d_cp-sample_offset*dstep);

      m_adiff   = _mm256_sub_ps(m_dcp, m_n1);
      m_adiff   = _mm256_andnot_ps(m_ps_zero, m_adiff); // 절대값
      m_edges   = _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS);
      m_results = _mm256_blendv_ps(m_edges, m_i_zero, _mm256_cmp_ps(m_n1, m_d_min, _CMP_LE_OS) ); // ignore results from invalid n1

      m_adiff   = _mm256_sub_ps(m_dcp, m_n3);
      m_adiff   = _mm256_andnot_ps(m_ps_zero, m_adiff);
      m_edges   = _mm256_or_ps(m_edges, _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS));
      m_edges   = _mm256_blendv_ps(m_edges, m_i_zero, _mm256_cmp_ps(m_n3, m_d_min, _CMP_LE_OS) ); // ignore results from invalid n3

      m_results = _mm256_or_ps(m_results, m_edges);
      m_results = _mm256_blendv_ps(m_results, m_i_zero, _mm256_cmp_ps(m_dcp, m_d_min, _CMP_LE_OS) ); // ignore results from invalid cp

      bit_mask      = _mm256_movemask_ps(m_results);
      for (k = 0; k < vstep; k++) 
        e[k] = (bit_mask >> k) & 1;
    }
    depth += dstep;
    edge += estep;
  }
  return;
}

void GetGrad(const cv::Mat _depth, float fx, float fy, const int offset,
             cv::Mat& _gradx, cv::Mat& _grady) {
  if(_gradx.empty())
    _gradx = cv::Mat::zeros(_depth.rows, _depth.cols, CV_32FC1);
  if(_grady.empty())
    _grady = cv::Mat::zeros(_depth.rows, _depth.cols, CV_32FC1);

  cv::Size size = _depth.size();
  typedef float Real; // Type of depth
  typedef __m256 T;   // Type of vector
  const int vstep = sizeof(T) / sizeof(Real);
  Real* gradx = _gradx.ptr<Real>();
  Real* grady = _grady.ptr<Real>();
  const Real* depth = _depth.ptr<Real>();
  const int dstep = int(_depth.step/sizeof(depth[0]));
  const int gstep = int(_gradx.step/sizeof(gradx[0]));
  long bit_mask;
  int v, u,k ;
  T m_cp, m_dx0, m_dx1, m_dy0, m_dy1, m_gx, m_gy;

  const int offset_step = offset*dstep;
  T m_hfx = _mm256_set1_ps(.5 * fx / (float) offset);
  T m_hfy = _mm256_set1_ps(.5 * fy / (float) offset);

  for(v = 0; v < offset; v++){
    depth += dstep;
    gradx += gstep;
    grady += gstep;
  }
  for( v = offset; v < size.height-offset; v++ ) {
    for( u = offset; u < size.width-offset-vstep; u+=vstep ) {
      const Real* cp = depth + u;
      m_cp      = _mm256_loadu_ps(cp);
      m_dx0     = _mm256_loadu_ps(cp-offset);
      m_dx1     = _mm256_loadu_ps(cp+offset);
      m_dy0     = _mm256_loadu_ps(cp-offset_step);
      m_dy1     = _mm256_loadu_ps(cp+offset_step);
      /*
      gx = [ z(u+1,v) - z(u-1,v) ] / dx,
      dx = 2. * offset * z(u,v) / fx 
      -> gx =  .5 * inv_offset * fx * [ z(u+1,v) - z(u-1,v) ] / z(u,v)
      */
      m_gx      = _mm256_sub_ps(m_dx1, m_dx0);
      m_gx      = _mm256_div_ps(_mm256_mul_ps(m_hfx,m_gx), m_cp );

      m_gy      = _mm256_sub_ps(m_dy1, m_dy0);
      m_gy      = _mm256_div_ps(_mm256_mul_ps(m_hfy,m_gy), m_cp );

      _mm256_storeu_ps(gradx+u, m_gx);
      _mm256_storeu_ps(grady+u, m_gy);
    }
    depth += dstep;
    gradx += gstep;
    grady += gstep;
  }
  return;
}

void GetConcaveEdges(const cv::Mat& _gradx,
                     const cv::Mat& _grady,
                     const cv::Mat _depth,
                     float fx, float fy,
                     const int hoffset,
                     const int voffset,
                     float _neg_hessian_threshold, // -100.f
                     cv::Mat& _edge) {
  if(_edge.empty())
    _edge = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);

  cv::Size size = _depth.size();
  typedef float Real; // Type of depth
  typedef __m256 T;   // Type of vector
  const int vstep = sizeof(T) / sizeof(Real);
  const Real* gradx = _gradx.ptr<Real>();
  const Real* grady = _grady.ptr<Real>();
  const Real* depth = _depth.ptr<Real>();

  uchar* edge = _edge.ptr<uchar>();

  const int dstep = int(_depth.step/sizeof(depth[0]));
  const int gstep = int(_gradx.step/sizeof(gradx[0]));
  const int estep = int(_edge.step/sizeof(edge[0]));

  T m_hfx = _mm256_set1_ps(.5*fx);
  T m_hfy = _mm256_set1_ps(.5*fy);

  long bit_mask;
  int u, v, k;
  T m_cp, m_gx0, m_gx1, m_gy0, m_gy1, m_hxx, m_hyy,  m_edges, m_trace; //, m_vx0, m_vx1, m_vy0, m_vy1;

  T m_neg_h_th = _mm256_set1_ps(_neg_hessian_threshold);
  T m_w_min   = _mm256_set1_ps(1.);
  T m_w_depth = _mm256_set1_ps(20.); // [meter]
  T m_min_depth = _mm256_set1_ps(1e-5);
  T m_i_zero  = _mm256_set1_ps(0.f);
  T m_weighted_h_th;

  const int e_offset_step = voffset*estep;
  const int g_offset_step = voffset*gstep;
  for(v = 0; v < voffset; v++){
    depth += dstep;
    gradx += gstep;
    grady += gstep;
    edge  += estep;
  }
  for( ; v < size.height-voffset; v++ ) {
    for( u = hoffset; u < size.width-hoffset-vstep; u+=vstep ) {
      const Real* d_cp = depth + u;
      const Real* gx_cp = gradx + u;
      const Real* gy_cp = grady + u;
      uchar* e = edge + u;
      m_cp       = _mm256_loadu_ps(d_cp);
      m_gx0      = _mm256_loadu_ps(gx_cp-hoffset);
      m_gx1      = _mm256_loadu_ps(gx_cp+hoffset);
      m_gy0      = _mm256_loadu_ps(gy_cp-g_offset_step);
      m_gy1      = _mm256_loadu_ps(gy_cp+g_offset_step);

      /*
      Hxx = [ gx(u+1,v) - gx(u-1,v) ] / dx,
      dx = 2. * offset * z(u,v) / fx 
      -> 
      Hxx =  .5 * offset^-1 * fx * [ gx(u+1,v) - gx(u-1,v) ] / z(u,v)
      */
      m_hxx      = _mm256_sub_ps(m_gx1, m_gx0);
      m_hxx      = _mm256_div_ps(_mm256_mul_ps(m_hfx, m_hxx), m_cp);

      m_hyy      = _mm256_sub_ps(m_gy1, m_gy0);
      m_hyy      = _mm256_div_ps(_mm256_mul_ps(m_hfy, m_hyy), m_cp);

      m_weighted_h_th = _mm256_max_ps(m_w_min, _mm256_div_ps(m_w_depth,m_cp) );
      m_weighted_h_th = _mm256_mul_ps(m_weighted_h_th, m_neg_h_th);

      m_trace  = _mm256_add_ps(m_hxx, m_hyy);
      m_edges    = _mm256_cmp_ps(m_trace, m_weighted_h_th, _CMP_LE_OS);

      bit_mask      = _mm256_movemask_ps(m_edges);
      for (k = 0; k < vstep; k++) 
        e[k] = (bit_mask >> k) & 1;
    }
    depth += dstep;
    gradx += gstep;
    grady += gstep;
    edge  += estep;
  }

  return;
}


} // namespace WithSIMD

void OutlineEdgeDetectorWithSIMD::PutDepth(cv::Mat depth, float fx, float fy) {
  const int grad_sample_offset = 3;
  const int concave_hsample_offset = 3;
  const int concave_vsample_offset = 6;
  const float neg_hessian_threshold = -50.;

  valid_mask_ = depth > 0.;// 너무 가까운데는 depth 추정이 제대로 안됨.
  cv::erode(valid_mask_,valid_mask_,cv::Mat::ones(10,10,CV_8UC1) ); 
  /*
  SIMD에서는 valid depth인지 아닌지 비교하고 0으로 맵핑하는게 오히려 더 오래걸린다.
  grad, hessian 다 구하고서 일괄적으로 valid_mask 0을 0으로 지워버리는게 더 낫다.
  */

  //cv::Mat filtered_depth;
  //cv::GaussianBlur(depth,filtered_depth, cv::Size(7,7), 0., 0.); // Shallow groove가 필요없어서 그냥 Gaussian
  WithSIMD::GetDDEdges(depth,dd_edges_); // concave_edge를 보완해주는 positive detection이 없음.
  WithSIMD::GetGrad(depth, fx, fy, grad_sample_offset, gradx_, grady_);
  WithSIMD::GetConcaveEdges(gradx_,grady_,depth,fx,fy,concave_hsample_offset,concave_vsample_offset,neg_hessian_threshold, concave_edges_);
  cv::bitwise_or(concave_edges_, dd_edges_, outline_edges_);
  cv::bitwise_and(outline_edges_, valid_mask_, outline_edges_);
  cv::erode(outline_edges_,outline_edges_,cv::Mat::ones(3,3,CV_8UC1) ); 
  return;
}

void OutlineEdgeDetectorWithoutSIMD::PutDepth(cv::Mat depth, float fx, float fy) {
  float meter_sample_offset = .1;
  float neg_hessian_threshold = -15.;
  bool use_filtered_depth = true;

  cv::Mat valid_grad;
  if(use_filtered_depth){
    valid_mask_  = depth > 0.;
    cv::Mat filtered_depth;
    cv::erode(valid_mask_,valid_mask_,cv::Mat::ones(13,13,CV_8UC1) );
    cv::GaussianBlur(depth,filtered_depth, cv::Size(7,7), 0., 0.); // Shallow groove가 필요없어서 그냥 Gaussian

    WithoutSIMD::GetGrad(filtered_depth, valid_mask_, fx, fy, meter_sample_offset, gradx_, grady_, valid_grad);
    cv::bitwise_and(valid_mask_, valid_grad, valid_mask_);
    cv::Mat concave_edges = WithoutSIMD::GetConcaveEdges(gradx_,grady_,depth,valid_mask_,fx,fy,neg_hessian_threshold);
    cv::Mat dd_edges = WithoutSIMD::GetDDEdges(filtered_depth, valid_mask_,fx, fy); // concave_edge를 보완해주는 positive detection이 없음.
    cv::bitwise_or(concave_edges, dd_edges, outline_edges_);
    outline_edges_ = WithoutSIMD::FilterThinNoise(outline_edges_);
  }
  return;
}

cv::Mat OutlineEdgeDetectorWithSizelimit::ComputeDDEdges(const cv::Mat depth) const {
  // 원경에서는 FP가 안생기게, invd(disparity)를 기준으로 찾아냄.
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  float l = 2;
  std::vector<cv::Point2i> samples = {
    cv::Point2i(l,0),
    cv::Point2i(0,l),
  };

  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt(c0,r0);
      for(const auto& dpt : samples){
        const cv::Point2i pt0 = pt-dpt;
        if(pt0.x < 0 || pt0.x >= depth.cols || pt0.y < 0 || pt0.y >= depth.rows)
          continue;
        const float invz0 = 1. / depth.at<float>(pt0);

        const cv::Point2i pt1 = pt0+dpt;
        if(pt1.x < 0 || pt1.x >= depth.cols || pt1.y < 0 || pt1.y >= depth.rows)
          continue;
        const float invz1 = 1. / depth.at<float>(pt1);
        float th = invz0 * 0.1;
        float diff = invz1 - invz0;
        if(std::abs(diff) < th)
          continue;
        edge.at<unsigned char>(pt0) = diff < 0 ? DdType::FG : DdType::BG;
        edge.at<unsigned char>(pt1) = diff > 0 ? DdType::FG : DdType::BG;
      } // samples
    }
  }

  return edge;
}


cv::Mat OutlineEdgeDetectorWithSizelimit::ComputeConcaveEdges(const cv::Mat depth, const cv::Mat dd_edges, float fx, float fy) const {
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  const float sample_pixelwidth = 10.; // half height
  const float min_obj_width = 1.; // [meter]

  for(int r0 = sample_pixelwidth; r0 < depth.rows-sample_pixelwidth; r0++) {
    for(int c0 = sample_pixelwidth; c0 < depth.cols-sample_pixelwidth; c0++) {
      const cv::Point2i pt(c0,r0);
      const float& z0 = depth.at<float>(pt);
      uchar& e = edge.at<uchar>(pt);
      if( dd_edges.at<uchar>(pt) )
        continue;
#if 1
      bool left_dd = false;
      float hw = fx * min_obj_width / z0;
      for(int l = 1; l < hw; l++) {
        const cv::Point2i pt_l(pt.x-l, pt.y);
        if(pt_l.x < 0)
          break;
        if(z0 - depth.at<float>(pt_l) > min_obj_width)
          break;
        if(dd_edges.at<uchar>(pt_l) < 1)
          continue;
        left_dd = true;
        break;
      }
      bool right_dd = false;
      for(int l = 1; l < hw; l++) {
        const cv::Point2i pt_r(pt.x + l, pt.y);
        if(pt_r.x >= depth.rows)
          break;
        if(z0 - depth.at<float>(pt_r) > min_obj_width)
          break;
        if(dd_edges.at<uchar>(pt_r) < 1)
          continue;
        right_dd = true;
        break;
      }
      if(left_dd || right_dd)
        continue;
#endif
      // Convexity 계산
#if 1
      float hessian_th = -20.;
      {
        const cv::Point2i pt_y0(pt.x, pt.y-sample_pixelwidth);
        const cv::Point2i pt_y1(pt.x, pt.y+sample_pixelwidth);
        if( pt_y0.y > 0 && pt_y1.y < depth.rows){
          const float dy = sample_pixelwidth/fy*z0;
          const float gy0 = (z0 - depth.at<float>(pt_y0)) / dy;
          const float gy1 = (depth.at<float>(pt_y1) - z0) / dy;
          // atan2(dz0, dy) , atan2(dz1,dy)으로 내각을 구해야... 하지만..
          if( (gy1-gy0)/dy < hessian_th)
            e |= VERTICAL;
        }
      }
      {
        const cv::Point2i pt_x0(pt.x-sample_pixelwidth, pt.y);
        const cv::Point2i pt_x1(pt.x+sample_pixelwidth, pt.y);
        if( pt_x0.x > 0 && pt_x1.x < depth.cols){
          const float dx = sample_pixelwidth/fx*z0;
          const float gx0 = (z0 - depth.at<float>(pt_x0)) / dx;
          const float gx1 = (depth.at<float>(pt_x1) - z0) / dx;
          if( (gx1-gx0)/dx < hessian_th)
            e |= HORIZONTAL;
        }
      }
#else
#endif
    } // cols
  } // rows
  return edge;
}


void OutlineEdgeDetectorWithSizelimit::PutDepth(const cv::Mat depth, float fx, float fy) {
  /*
    * convexity 정확하게 계산하는거 먼저. 최적화는 나중에.
  */
  dd_edges_ = ComputeDDEdges(depth); // concave_edge를 보완해주는 positive detection이 없음.
  concave_edges_ = ComputeConcaveEdges(depth, dd_edges_, fx, fy);
  cv::bitwise_or(dd_edges_ > 0, concave_edges_ > 0, outline_edges_);
  return;
}


cv::Mat MergeOcclusion(const cv::Mat depth,
                       const cv::Mat dd_edges,
                       const cv::Mat _marker0) {
  /*
  TODO 우선 merge occlusion 제대로 되는거 확인한다음,
  distanceTransform(GetBoundaryS(marker))를 또하는 대신, instance segmentation 단계에서 가져와, 연산낭비하는거 수정.
  */
  cv::Mat boundary = GetBoundary(_marker0);
  cv::Mat boundary_distance;
  cv::distanceTransform(boundary<1, boundary_distance, cv::DIST_L2, cv::DIST_MASK_3);
  std::map<int,float> marker_radius; // TODO dist_boundary, marker_radius를 Segmentor 에서 수집.

  const float sqrt2 = std::sqrt(2.);
  const cv::Size size = _marker0.size();
  int w1 = size.width-1;
  int h1 = size.height-1;

  struct Node {
    float cost;
    int x,y; // 아직 marker가 정해지지 않은 grid의 x,y
    int32_t m;   // 부모 marker.
    float z; // latest z
    uchar e; // expanding 과정에서 만난 edge type
    Node(int _x, int _y, float _cost, int32_t _m, float _z, uchar _e) : x(_x), y(_y), cost(_cost), m(_m), z(_z), e(_e) {
    }
    bool operator < (const Node& other) const {
      return cost > other.cost;
    }
  };

  std::priority_queue<Node> q1, q2;
  cv::Mat _marker = _marker0.clone();
  cv::Mat costs   = 999.*cv::Mat::ones(_marker0.rows, _marker0.cols, CV_32FC1);
  cv::Mat exptype = cv::Mat::zeros(_marker0.rows, _marker0.cols, CV_8UC1); // 확장도중 만난 edge type?

  for( int y = 0; y < size.height; y++ ) {
    for(int  x = 0; x < size.width; x++ ) {
      const int32_t& m0 = _marker0.at<int32_t>(y,x);
      if(m0<1)
        continue;
      marker_radius[m0] = std::max(marker_radius[m0], boundary_distance.at<float>(y,x) );
    }
  }

  for( int y = 0; y < size.height; y++ ) {
    for(int  x = 0; x < size.width; x++ ) {
      if(_marker.at<int32_t>(y,x)>0)
        continue;
      // 근처에 valid marker가 있으면 현재위치를 추가해야함.
      if(x > 0){
        const int32_t& m = _marker.at<int32_t>(y,x-1);
        const uchar& e   =   dd_edges.at<uchar>(y,x-1);
        const float& z   =      depth.at<float>(y,x-1);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(x < w1){
        const int32_t& m = _marker.at<int32_t>(y,x+1);
        const uchar& e   =   dd_edges.at<uchar>(y,x+1);
        const float& z   =      depth.at<float>(y,x+1);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(y > 0){
        const int32_t& m = _marker.at<int32_t>(y-1,x);
        const uchar& e   =   dd_edges.at<uchar>(y-1,x);
        const float& z   =      depth.at<float>(y-1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(y < h1){
        const int32_t& m = _marker.at<int32_t>(y+1,x);
        const uchar& e   =   dd_edges.at<uchar>(y+1,x);
        const float& z   =      depth.at<float>(y+1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
    } // for y
  } // for x

  const uchar FG = OutlineEdgeDetectorWithSizelimit::DdType::FG;
  const uchar BG = OutlineEdgeDetectorWithSizelimit::DdType::BG;

  /*
    * exptype에서 서로 다른 marker이 bg edge node끼리 많이 만나는 경우 -> merge pair.
    * exptype에서 fg-bg edge가 만나는 경우..
      - (가느다란 fg)를 지운 상태에서, 인접한 bg edge node들을 확장한 결과, (latest bg depth차가 작은것끼리)많이 만나는 경우 -> merge pair.
  */
  std::map<std::pair<int,int>, size_t> bgbg_contacts; // key : min(bg_m),max(bg_m)
  std::map<std::pair<int,int>, size_t> fgbg_contacts; // key : thin fg_m, bg_m
  float thin_th = 10.;

  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    uchar& e = exptype.at<uchar>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    e = k.e;
    cost = k.cost;
    m = k.m; // 주변 노드중, marker가 정해지지 않은 노드를 candidate에 추가
    if(k.e == FG) // contact counting을 위해 fg pixel에 marker만 맵핑하고 확장은 중단.
      continue;
    if(x > 0){
      const int32_t& m2 = _marker.at<int32_t>(y,x-1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x-1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x-1); // edge를 넘어서부턴 z update를 중단.
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x-1,y, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          //if(m2==56 && k.m==45) throw -1;
          q2.push(next); // m2 지우고 나서 확장을 다시시도.
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(x < w1 ){
      const int32_t& m2 = _marker.at<int32_t>(y,x+1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x+1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x+1); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x+1,y, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          //if(m2==56 && k.m==45) throw -1;
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(y > 0 && _marker.at<int32_t>(y-1,x)<1 ){
      const int32_t& m2 = _marker.at<int32_t>(y-1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y-1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y-1,x); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x,y-1, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(y < h1){
      const int32_t& m2 = _marker.at<int32_t>(y+1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y+1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y+1,x); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x,y+1, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
  } // while(!q1.empty())

  bgbg_contacts.clear(); // TODO remove
  std::map<int, std::set<int> > bg_neighbors; {
    std::map<int, std::list<int> > _bg_neighbors;
    for(auto it : fgbg_contacts){
      if(it.second > 20){
        printf("fgbg %d, %d - %ld\n", it.first.first, it.first.second, it.second);
        _bg_neighbors[it.first.first].push_back(it.first.second);
      }
    }
    for(auto it : _bg_neighbors){
      if(it.second.size()< 1)
        continue;
      for(auto it_key : it.second){
        for(auto it_n : it.second){
          if(it_key==it_n)
            continue;
          bg_neighbors[it_key].insert(it_n);
        }
      }
    }
  }


  {
    cv::Mat dst = GetColoredLabel(_marker,true);
    cv::imshow("marker1", dst);
  }
  for(auto it : marker_radius){
    if(it.second < 10.){
      cv::Mat mask = _marker==it.first;
      _marker.setTo(0, mask);
      costs.setTo(999., mask);
    }
  }

  const float max_zerr = 100.; // TODO remove
  while(!q2.empty()){
    Node k = q2.top(); q2.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    uchar& e = exptype.at<uchar>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    //e = k.e;
    cost = k.cost;
    m = k.m; // 주변 노드중, marker가 정해지지 않은 노드를 candidate에 추가

    if(x > 0){
      const int32_t& m2 = _marker.at<int32_t>(y,x-1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x-1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x-1);
      Node next(x-1,y, k.cost+1., k.m, k.z, e); // 더이상 update 중단.
      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(x < w1){
      const int32_t& m2 = _marker.at<int32_t>(y,x+1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x+1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x+1);
      Node next(x+1,y, k.cost+1., k.m, k.z, e);

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(y > 0){
      const int32_t& m2 = _marker.at<int32_t>(y-1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y-1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y-1,x);
      Node next(x,y-1, k.cost+1., k.m, k.z, e); // 더이상 update 중단.

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(y < h1){
      const int32_t& m2 = _marker.at<int32_t>(y+1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y+1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y+1,x);
      Node next(x,y+1, k.cost+1., k.m, k.z, e);

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
  }

  for(auto it : bgbg_contacts)
    if(it.second > 20)
      printf("bgbg %d, %d - %ld\n", it.first.first, it.first.second, it.second);

  std::cout << "-------------" << std::endl;



  {
    cv::Mat dst = GetColoredLabel(_marker,true);
    dst.setTo(CV_RGB(0,0,0), GetBoundary(_marker));
    cv::imshow("marker2", dst);
  }
  //cv::imshow("dist_boundary", 0.01*boundary_distance);

  return _marker0;
}
