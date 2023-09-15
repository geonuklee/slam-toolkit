#include "simd.h"
#include <immintrin.h> // sse or avx

cv::Mat GetDDEdges(const cv::Mat _depth, bool with_simd){
  cv::Mat _edge = cv::Mat::zeros(_depth.rows, _depth.cols, CV_8UC1);
  if(! with_simd){
    std::vector<cv::Point2i> samples = {
      cv::Point2i(1,0),
      cv::Point2i(-1,0),
      cv::Point2i(0,1),
      cv::Point2i(0,-1),
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
    // ref) https://github.com/opencv/opencv/blob/4790a3732e725b102f6c27858e7b43d78aee2c3e/modules/imgproc/src/segmentation.cpp#L88
    cv::Size size = _depth.size();
    typedef float Real;
    typedef __m256 T;
    const int vstep = sizeof(T) / sizeof(Real);
    const Real* depth = _depth.ptr<Real>();
    uchar* edge = _edge.ptr<uchar>();

    const int dstep = int(_depth.step/sizeof(depth[0]));
    const int estep = int(_edge.step/sizeof(edge[0]));

    __m256 m_th = _mm256_set1_ps(0.1);
    __m256 m_th_max = _mm256_set1_ps(2.);
    __m256 m_zero = _mm256_set1_ps(-0.0f);
    int i, j, k;
    for( i = 1; i < size.height-1; i++ ) {
      depth += dstep;
      edge += estep;
      for( j = 1; j < size.width-1-vstep; j+=vstep ) {
        const Real* d_cp = depth + j;
        uchar* e = edge + j;
        __m256 m_dcp   = _mm256_loadu_ps(d_cp);
        __m256 m_ath   = _mm256_mul_ps(m_dcp, m_th);
        m_ath = _mm256_blendv_ps(m_ath, m_th_max,  _mm256_cmp_ps(m_th_max, m_ath, _CMP_GE_OS) );

        __m256 m_n1    = _mm256_loadu_ps(d_cp-1);
        __m256 m_n2    = _mm256_loadu_ps(d_cp+1);
        __m256 m_n3    = _mm256_loadu_ps(d_cp-dstep);
        __m256 m_n4    = _mm256_loadu_ps(d_cp+dstep);

        __m256 m_adiff;
        __m256 m_edges;
        m_adiff = _mm256_sub_ps(m_dcp, m_n1);
        m_adiff = _mm256_andnot_ps(m_zero, m_adiff); // 절대값
        m_edges = _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS); // comp
        auto m_results = m_edges;

        m_adiff = _mm256_sub_ps(m_dcp, m_n3);
        m_adiff = _mm256_andnot_ps(m_zero, m_adiff);
        m_edges = _mm256_or_ps(m_edges, _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS));
        m_results = _mm256_or_ps(m_results, m_edges);

        m_adiff = _mm256_sub_ps(m_dcp, m_n2);
        m_adiff = _mm256_andnot_ps(m_zero, m_adiff);
        m_edges = _mm256_or_ps(m_edges, _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS));
        m_results = _mm256_or_ps(m_results, m_edges);

        m_adiff = _mm256_sub_ps(m_dcp, m_n4);
        m_adiff = _mm256_andnot_ps(m_zero, m_adiff);
        m_edges = _mm256_or_ps(m_edges, _mm256_cmp_ps(m_adiff, m_ath, _CMP_GE_OS));
        m_results = _mm256_or_ps(m_results, m_edges);

        long mask = _mm256_movemask_ps(m_results);
        for (k = 0; k < vstep; k++) 
          e[k] = (mask >> k) & 1;
      }
    }
  }
  return _edge;
}

