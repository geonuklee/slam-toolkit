#include "simd.h"
#include <immintrin.h> // sse or avx

/*

* Referenced tutorials, codes 
  * for SIMD programming : http://supercomputingblog.com/windows/image-processing-with-sse/
  * for image process    : https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/imgproc/src/segmentation.cpp#L88


*/

void GetDDEdges(const cv::Mat _depth,
                   cv::Mat& _edge,
                   bool with_simd){
  if(_edge.empty())
    _edge = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);

  if(!with_simd){
    std::vector<cv::Point2i> samples = {
      cv::Point2i(-1,0),
      cv::Point2i(0,-1),
      //cv::Point2i(1,0),
      //cv::Point2i(0,1),
    };
    for(int r0 = 0; r0 < _depth.rows; r0++) {
      for(int c0 = 0; c0 < _depth.cols; c0++) {
        const cv::Point2i pt0(c0,r0);
        const float& z0 = _depth.at<float>(pt0);
        unsigned char& e = _edge.at<unsigned char>(pt0);
        for(int l=1; l < 2; l++){
          for(const auto& dpt : samples){
            const cv::Point2i pt1 = pt0+l*dpt;
            if(pt1.x < 0 || pt1.x >= _depth.cols || pt1.y < 0 || pt1.y >= _depth.rows)
              continue;
            const float& z1 = _depth.at<float>(pt1);
            float th = std::max<float>(0.1*z0, 2.);
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
  }
  else{
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
    T m_zero = _mm256_set1_ps(-0.0f); // 주의 :  -0.0f != 0.0f
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
        m_adiff   = _mm256_andnot_ps(m_zero, m_adiff); // 절대값
        m_edges   = _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS);
        m_results = _mm256_blendv_ps(m_edges, m_zero, _mm256_cmp_ps(m_n1, m_d_min, _CMP_LE_OS) ); // ignore results from invalid n1
        //m_results = m_edges;

        m_adiff   = _mm256_sub_ps(m_dcp, m_n3);
        m_adiff   = _mm256_andnot_ps(m_zero, m_adiff);
        m_edges   = _mm256_or_ps(m_edges, _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS));
        m_edges   = _mm256_blendv_ps(m_edges, m_zero, _mm256_cmp_ps(m_n3, m_d_min, _CMP_LE_OS) ); // ignore results from invalid n3
        m_results = _mm256_or_ps(m_results, m_edges);
        m_results = _mm256_blendv_ps(m_results, m_zero, _mm256_cmp_ps(m_dcp, m_d_min, _CMP_LE_OS) ); // ignore results from invalid cp

        bit_mask      = _mm256_movemask_ps(m_results);
        for (k = 0; k < vstep; k++) 
          e[k] = (bit_mask >> k) & 1;
      }
    }
  }
  return;
}


cv::Mat VisualizeGrad(const cv::Mat gradx, const cv::Mat grady) {
  cv::Mat dst = cv::Mat::zeros(gradx.rows, gradx.cols, CV_8UC3);
  for(int r = 0; r < gradx.rows; r++) {
    for(int c = 0; c < gradx.cols; c++) {
      const cv::Point2i pt(c,r);
      auto& color = dst.at<cv::Vec3b>(pt);
      const float& gx = gradx.at<float>(pt);
      const float& gy = grady.at<float>(pt);
      int val = std::min<int>(255, std::max<int>(0, std::abs(10.*gy) ) );
      if(gy > 0)
        color[0] = val;
      else
        color[1] = val;
      val = std::min<int>(255, std::max<int>(0, std::abs(10.*gx) ) );
      color[2] = val;

    }
  }

  return dst;
}

void GetGrad(const cv::Mat _depth, float fx, float fy, const cv::Mat _valid_mask,
             const int offset,
             cv::Mat& _gradx, cv::Mat& _grady, cv::Mat& _valid_grad, bool with_simd) {
  if(_gradx.empty())
    _gradx = cv::Mat::zeros(_depth.rows, _depth.cols, CV_32FC1);
  if(_grady.empty())
    _grady = cv::Mat::zeros(_depth.rows, _depth.cols, CV_32FC1);
  if(_valid_grad.empty())
    _valid_grad = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);

  if(!with_simd){
    const float s = 0.1;
    const float sfx = s*fx;
    const float sfy = s*fy;

    for(int r = 0; r < _depth.rows; r++) {
      for(int c = 0; c < _depth.cols; c++) {
        const cv::Point2i pt(c,r);
        if(!_valid_mask.at<uchar>(pt))
          continue;
        const float& z = _depth.at<float>(pt);
        if(z < 1e-3)
          continue;
        const int du = std::max<int>( (int) sfx/z, 1);
        const int dv = std::max<int>( (int) sfy/z, 1);
        cv::Point2i ptx0(c-du,r);
        cv::Point2i ptx1(c+du,r);
        cv::Point2i pty0(c,r-dv);
        cv::Point2i pty1(c,r+dv);
        if(ptx0.x < 0 || !_valid_mask.at<uchar>(ptx0) )
          continue;
        if(ptx1.x > _depth.cols || !_valid_mask.at<uchar>(ptx1) )
          continue;
        if(pty0.y < 0 || !_valid_mask.at<uchar>(pty0) )
          continue;
        if(pty1.y > _depth.rows || !_valid_mask.at<uchar>(pty1) )
          continue;
        const float zx0 = _depth.at<float>(ptx0);
        const float zx1 = _depth.at<float>(ptx1);
        const float zy0 = _depth.at<float>(pty0);
        const float zy1 = _depth.at<float>(pty1);
        if(zx0 < 1e-3 || zx1 < 1e-3 || zy0 < 1e-3 || zy1 < 1e-3)
          continue;
        _valid_grad.at<uchar>(pt) = true;
        float& gx = _gradx.at<float>(pt);
        float& gy = _grady.at<float>(pt);
        gx = (zx1 - zx0) / s;
        gy = (zy1 - zy0) / s;
      }
    }
  } else {
    cv::Size size = _depth.size();
    typedef float Real; // Type of depth
    typedef __m256 T;   // Type of vector
    const int vstep = sizeof(T) / sizeof(Real);
    Real* gradx = _gradx.ptr<Real>();
    Real* grady = _grady.ptr<Real>();
    const Real* depth = _depth.ptr<Real>();
    const int dstep = int(_depth.step/sizeof(depth[0]));
    const int gstep = int(_gradx.step/sizeof(gradx[0]));
    int v, u;
    T m_cp, m_dx0, m_dx1, m_dy0, m_dy1, m_gx, m_gy, m_zeromask;

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
        m_zeromask = _mm256_cmp_ps(m_cp, _mm256_setzero_ps(), _CMP_EQ_OQ);
        m_dx0     = _mm256_loadu_ps(cp-offset);
        m_dx1     = _mm256_loadu_ps(cp+offset);
        m_dy0     = _mm256_loadu_ps(cp-offset_step);
        m_dy1     = _mm256_loadu_ps(cp+offset_step);
        /*
        gx = [ z(u+1,v) - z(u-1,v) ] / dx,
        dx = 2. * offset * z(u,v) / fx 
        -> 
        gx =  .5 * inv_offset * fx * [ z(u+1,v) - z(u-1,v) ] / z(u,v)
        */
        m_gx      = _mm256_sub_ps(m_dx1, m_dx0);
        m_gx      = _mm256_div_ps(_mm256_mul_ps(m_hfx,m_gx), m_cp );
        m_gx       = _mm256_andnot_ps(m_zeromask, m_gx);

        m_gy      = _mm256_sub_ps(m_dy1, m_dy0);
        m_gy      = _mm256_div_ps(_mm256_mul_ps(m_hfy,m_gy), m_cp );
        m_gy      = _mm256_andnot_ps(m_zeromask, m_gy);

        _mm256_storeu_ps(gradx+u, m_gx);
        _mm256_storeu_ps(grady+u, m_gy);
      }
      depth += dstep;
      gradx += gstep;
      grady += gstep;
    }
  } // if !with_simd
  return;
}

static bool SameSign(const float& v1, const float& v2){
  if(v1 > 0.)
    return v2 > 0.;
  else if(v1 < 0.)
    return v2 < 0.;
  return (v1 == 0.) && (v2 == 0.);
}

void GetConcaveEdges(const cv::Mat& _gradx,
                     const cv::Mat& _grady,
                     const cv::Mat _depth,
                     const cv::Mat _valid_mask,
                     const int offset,
                     float fx, float fy,
                     cv::Mat& _edge,
                     bool with_simd) {
  if(_edge.empty())
    _edge = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);
  if(!with_simd){
    cv::Mat hessian = cv::Mat::zeros(_depth.rows, _depth.cols, CV_32FC1);
    const float s = .1; // sample offset. [meter]
    const float R = .5; // NMAS range. [meter]
    const float sfx = s*fx;
    const float sfy = s*fy;
    const float Rfx = R*fx;
    for(int r=0; r < _depth.rows-0; r++) {
      for(int c=0; c < _depth.cols-0; c++) {
        const cv::Point2i pt(c,r);
        if(!_valid_mask.at<uchar>(pt))
          continue;
        const float& z = _depth.at<float>(pt);
        if(z < 1e-3)
          continue;
        const int du = std::max<int>( (int) sfx/z, 1);
        const int dv = std::max<int>( (int) sfy/z, 1);

        cv::Point2i ptx0(c-du,r);
        cv::Point2i ptx1(c+du,r);
        cv::Point2i pty0(c,r-dv);
        cv::Point2i pty1(c,r+dv);
        if(ptx0.x < 0 || !_valid_mask.at<uchar>(ptx0) )
          continue;
        if(ptx1.x > _depth.cols || !_valid_mask.at<uchar>(ptx1) )
          continue;
        if(pty0.y < 0 || !_valid_mask.at<uchar>(pty0) )
          continue;
        if(pty1.y > _depth.rows || !_valid_mask.at<uchar>(pty1) )
          continue;
        const float zx0 = _depth.at<float>(ptx0);
        const float zx1 = _depth.at<float>(ptx1);
        const float zy0 = _depth.at<float>(pty0);
        const float zy1 = _depth.at<float>(pty1);
        if(zx0 < 1e-3 || zx1 < 1e-3 || zy0 < 1e-3 || zy1 < 1e-3)
          continue;

        float hxx = (_gradx.at<float>(ptx1) - _gradx.at<float>(ptx0) ) / s;
        float hyy = (_grady.at<float>(pty1) - _grady.at<float>(pty0) ) / s;
        hessian.at<float>(pt) = hxx + hyy;
      }
    }
    std::vector<cv::Point2i> samples = {
      cv::Point2i(1,0),
      cv::Point2i(-1,0),
      cv::Point2i(0,1),
      cv::Point2i(0,-1),
    };

    const int ioffset = 5;
    for(int r=ioffset; r < _depth.rows-ioffset; r++) {
      for(int c=ioffset; c < _depth.cols-ioffset; c++) {
        const cv::Point2i pt(c,r);
        if(!_valid_mask.at<uchar>(pt))
          continue;
        const float& z = _depth.at<float>(pt);
        if(z < 1e-3)
          continue;
        const float& h = hessian.at<float>(pt);
        int dL = std::max<int>( Rfx/z, 2);
        bool ismax = true;
        for(int dl=dL-1; dl>0; dl--){
          for(const auto& dpt : samples){
            const cv::Point2i pt1 = pt + dl * dpt;
            if(pt1.x < ioffset || pt1.y < ioffset || pt1.x >= _depth.cols-ioffset || pt1.y >= _depth.rows-ioffset ){
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
        if( h < -15.)
          _edge.at<uchar>(pt) = 1;
      }
    }
  }
  else { // if !with_simd
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
    T m_cp, m_gx0, m_gx1, m_gy0, m_gy1, m_hxx, m_hyy, m_zeromask, m_edges, m_results;

    T m_neg_h_th = _mm256_set1_ps(-100.);

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
        m_zeromask = _mm256_cmp_ps(m_cp, _mm256_setzero_ps(), _CMP_EQ_OQ);

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
        m_hxx      = _mm256_andnot_ps(m_zeromask, m_hxx);
        m_results  = _mm256_cmp_ps(m_hxx, m_neg_h_th, _CMP_LE_OS);

        m_hyy      = _mm256_sub_ps(m_gy1, m_gy0);
        m_hyy      = _mm256_div_ps(_mm256_mul_ps(m_hfy, m_hyy), m_cp);
        m_hyy      = _mm256_andnot_ps(m_zeromask, m_hyy);
        m_edges    = _mm256_cmp_ps(m_hyy, m_neg_h_th, _CMP_LE_OS);
        m_results = _mm256_or_ps(m_results, m_edges);

        bit_mask      = _mm256_movemask_ps(m_results);
        for (k = 0; k < vstep; k++) 
          e[k] = (bit_mask >> k) & 1;
      }
      depth += dstep;
      gradx += gstep;
      grady += gstep;
      edge  += estep;
    }


  } // if !with_simd
  return;
}

