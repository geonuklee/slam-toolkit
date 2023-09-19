#include "simd.h"
#include <immintrin.h> // sse or avx

/*
* Referenced tutorials, codes 
  * for SIMD programming : http://supercomputingblog.com/windows/image-processing-with-sse/
  * for image process    : https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/imgproc/src/segmentation.cpp#L88
*/

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

  T m_ratio = _mm256_set1_ps(0.1);
  T m_d_min = _mm256_set1_ps(0.001); // invalid depth
  T m_th_min = _mm256_set1_ps(2.);
  T m_ps_zero = _mm256_set1_ps(-0.0f); // 주의 :  -0.0f != 0.0f
  T m_i_zero  = _mm256_set1_ps(0.f);
  T m_dcp, m_ath, m_n1, m_n2, m_n3, m_n4, m_adiff, m_edges, m_results;
  long bit_mask;
  int i, j, k;
  for( i = 1; i < size.height-1; i++ ) {
    depth += dstep;
    edge += estep;
    for( j = 1; j < size.width-1-vstep; j+=vstep ) {
      const Real* d_cp = depth + j;
      uchar* e = edge + j;
      m_dcp     = _mm256_loadu_ps(d_cp);
      m_ath     = _mm256_mul_ps(m_dcp, m_ratio);
      m_ath     = _mm256_blendv_ps(m_ath, m_th_min,  _mm256_cmp_ps(m_th_min, m_ath, _CMP_GE_OS) );

      m_n1      = _mm256_loadu_ps(d_cp-1);
      m_n3      = _mm256_loadu_ps(d_cp-dstep);

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
                     const int offset,
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

  const int e_offset_step = offset*estep;
  const int g_offset_step = offset*gstep;
  for(v = 0; v < offset; v++){
    depth += dstep;
    gradx += gstep;
    grady += gstep;
    edge  += estep;
  }
  for( v = offset; v < size.height-offset; v++ ) {
    for( u = offset; u < size.width-offset-vstep; u+=vstep ) {
      const Real* d_cp = depth + u;
      const Real* gx_cp = gradx + u;
      const Real* gy_cp = grady + u;
      uchar* e = edge + u;
      m_cp       = _mm256_loadu_ps(d_cp);
      m_gx0      = _mm256_loadu_ps(gx_cp-offset);
      m_gx1      = _mm256_loadu_ps(gx_cp+offset);
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
  const int grad_sample_offset = 4;
  const int concave_sample_offset = 4;
  const float neg_hessian_threshold = -100.;

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
  WithSIMD::GetConcaveEdges(gradx_,grady_,depth,fx,fy,concave_sample_offset,neg_hessian_threshold, concave_edges_);
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

SegmentorNew::SegmentorNew() {

}
void SegmentorNew::Put(cv::Mat outline_edges, cv::Mat valid_mask) {

  return;
}


