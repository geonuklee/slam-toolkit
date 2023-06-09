#include <flann/flann.hpp> // include it before opencv
#include "../include/tracker.h"
#include <list>

std::vector<cv::Scalar> colors = {
  CV_RGB(0,255,0),
  CV_RGB(255,0,0),
  CV_RGB(255,0,255),
  CV_RGB(255,255,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,0),
  CV_RGB(100,0,255),
  CV_RGB(255,0,100),
  CV_RGB(100,0,100),
  CV_RGB(0,0,100),
  CV_RGB(100,255,0),
  CV_RGB(255,100,0),
  CV_RGB(100,100,0),
  CV_RGB(180,0,0),
  CV_RGB(100,0,0),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};

cv::Mat VisulaizeMask(cv::Mat mask){
  cv::Mat dst = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);
  for(int r = 0; r < mask.rows; r++){
    for(int c = 0; c < mask.cols; c++){
      const int& v = mask.at<int>(r,c);
      if(v >=0){
        const auto& color = colors.at(v % colors.size());
        dst.at<cv::Vec3b>(r,c)[0] = color[0];
        dst.at<cv::Vec3b>(r,c)[1] = color[1];
        dst.at<cv::Vec3b>(r,c)[2] = color[2];
      }
    }
  }
  return dst;
}

DenseTracker::DenseTracker(const Eigen::Matrix<double, 3,3>& K, const float base) :
  base_(base),
  cvK_( (cv::Mat_<float>(3,3) << K(0,0), K(0,1), K(0,2),
                                 K(1,0), K(1,1), K(1,2),
                                 K(2,0), K(2,1), K(2,2)) ),
  n_cluster_(0)
{
#if 0
  left_matcher_ = cv::StereoSGBM::create(0, 128, 3);
  wls_filter_ = cv::ximgproc::createDisparityWLSFilter(left_matcher_);
  right_matcher_ = cv::ximgproc::createRightMatcher(left_matcher_);
#else
  left_matcher_ = cv::cuda::createStereoBM(128, 19);
#endif

  opticalflow_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5,false, 13);
  //opticalflow_ = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
  //opticalflow_ = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(31,31),2);

}

template<typename T>
cv::Mat Disparity2Depth(const float& fx, const float& base, const cv::Mat& disparity){
  cv::Mat depth = cv::Mat::zeros(disparity.rows, disparity.cols, CV_32F);
  for(int r = 0; r < disparity.rows; r++){
    for(int c = 0; c < disparity.cols; c++){
      const T& dx = disparity.at<T>(r,c);
      if(dx> 0)
        depth.at<float>(r,c) = base*fx/dx;
    }
  }
  return depth;
}

cv::Mat DenseTracker::GetEdge(const cv::Mat gray) const {
  cv::Mat edge;
  cv::Mat sobelx, thin_edge;
  cv::Sobel(gray, sobelx, CV_8UC1,1,0,3);
  cv::convertScaleAbs(sobelx, sobelx);
  cv::threshold(sobelx, thin_edge, 50, 255,CV_8UC1);
  cv::Mat dist_transform;
  cv::distanceTransform(~thin_edge, dist_transform, cv::DIST_L2, cv::DIST_MASK_3);
  cv::threshold(dist_transform, edge,5., 255, cv::THRESH_BINARY_INV);
  edge.convertTo(edge, CV_8UC1);
  return edge;
}

#if 0
template<typename T>
cv::Mat DenseTracker::GetDisparity(const cv::Mat gray,
                                   const cv::Mat gray_r,
                                   const cv::Mat edge) const {
  cv::Mat disparity;
  cv::Mat disp_l;

  left_matcher_->compute(gray, gray_r, disp_l);
  cv::Mat disp_r;
  right_matcher_->compute(gray_r, gray, disp_r);
  wls_filter_->filter(disp_l, gray, disparity, disp_r);
  disparity.convertTo(disparity, CV_32F);

  // https://stackoverflow.com/questions/28959440/how-to-access-the-disparity-value-in-opencv
  disparity = (float) 1./16. * disparity;
#else
template<typename T>
cv::Mat DenseTracker::GetDisparity(const cv::cuda::GpuMat& g_gray,
                                   const cv::cuda::GpuMat& g_gray_r,
                                   const cv::Mat edge) const {

  cv::Mat disparity; {
    cv::cuda::GpuMat g_disp;
    left_matcher_->compute(g_gray, g_gray_r, g_disp);
    cv::Mat tmp;
    g_disp.download(tmp);
    tmp.convertTo(disparity, CV_32F);
  }
  // TODO Disparity는 항상 주의해야함. cuda::StereoBM은 바로 pixel disparity를 계산하고 있나?
#endif


  for(int r= 0; r < disparity.rows; r++)
    for(int c= 0; c < disparity.cols; c++)
      if(edge.at<unsigned char>(r,c) == 0)
        disparity.at<T>(r,c) = 0;

  return disparity;
}

cv::Mat DenseTracker::GetFlow(const cv::cuda::GpuMat g_gray0,
                              const cv::cuda::GpuMat g_gray) const {
  cv::Mat flow;
  cv::cuda::GpuMat g_flow;
  // ref) https://android.googlesource.com/platform/external/opencv3/+/master/samples/gpu/optical_flow.cpp#189
  if(opticalflow_->getDefaultName() == "DenseOpticalFlow.BroxOpticalFlow"){
    cv::cuda::GpuMat g_f_gray,g_f_gray0;
    g_gray0.convertTo(g_f_gray0,CV_32FC1, 1./255.);
    g_gray.convertTo(g_f_gray,CV_32FC1, 1./255.);
    opticalflow_->calc(g_f_gray, g_f_gray0, g_flow);
  }
  else
    opticalflow_->calc(g_gray, g_gray0, g_flow);
  g_flow.download(flow);
  return flow;
}

float SampleMagnitude(const cv::Mat flow, float sample_position = 0.5){
  const int stride = 20;
  std::vector<float> squares; {
    float n = flow.rows*flow.cols;
    float m = stride*stride;
    int N = (int) n*m*1.1;
    squares.reserve(N);
  }
  for(int r =stride; r < flow.rows-stride; r+= stride){
    for(int c =stride; c < flow.cols-stride; c+= stride){
      const float& du = flow.at<cv::Vec2f>(r,c)[0];
      const float& dv = flow.at<cv::Vec2f>(r,c)[1];
      squares.push_back(du*du + dv*dv);
    }
  }
  std::sort(squares.begin(), squares.end() );
  return std::sqrt(squares.at(sample_position*squares.size()));
}

template<typename T>
void DenseTracker::SamplePoints(int stride,
                                const cv::Mat flow,
                                const cv::Mat disparity,
                                const cv::Mat depth1,
                                std::map<int, EpipPoint>& epip_points
                                ) const {
  float fx = cvK_.at<float>(0,0);
  float fy = cvK_.at<float>(1,1);
  float cx = cvK_.at<float>(0,2);
  float cy = cvK_.at<float>(1,2);

  int i = 0;
  for(int r =stride; r < flow.rows-stride; r+= stride){
    for(int c =stride; c < flow.cols-stride; c+= stride){
      const float& du = flow.at<cv::Vec2f>(r,c)[0];
      const float& dv = flow.at<cv::Vec2f>(r,c)[1];
      const auto& disp = disparity.at<T>(r,c);
      if(disp < 2)
        continue;
      if(disp > 50.)
        continue;
      const float& z1 = depth1.at<float>(r,c);
      EpipPoint pt;
      pt.img0_ = cv::Point2f(c+du,r+dv);
      pt.img1_ = cv::Point2f(c, r);
      pt.obj1_ = cv::Point3f( ((float)c-cx)/fx*z1,
                           ((float)r-cy)/fy*z1,
                           z1);
      pt.cluster_ = -1;
      epip_points[i++] = pt;
    }
  }
  return;
}

void DenseTracker::RansacCluster(const cv::Mat flow,
                                 const cv::Mat disparity1,
                                 const cv::Mat depth0,
                                 const cv::Mat depth1,
                                 const std::set<int>& unclustered,
                                 std::map<int, EpipPoint>& epip_points
                                ) {
  const float max_rprj_error = 2.;

  std::vector<cv::Point2f> img0_points, img1_points,
                           tmp_img0_points, tmp_img1_points, rprj_points;
  std::vector<cv::Point3f> obj1_points, tmp_obj1_points;
  std::vector<int> indices, tmp_indices;

  img0_points.reserve(unclustered.size());
  img1_points.reserve(unclustered.size());
  tmp_img0_points.reserve(unclustered.size());
  tmp_img1_points.reserve(unclustered.size());
  rprj_points.reserve(unclustered.size());

  obj1_points.reserve(unclustered.size());
  tmp_obj1_points.reserve(img0_points.size());

  indices.reserve(unclustered.size());
  tmp_indices.reserve(unclustered.size());

  for(const int& index : unclustered){
    const EpipPoint& pt  = epip_points.at(index);
    img0_points.push_back(pt.img0_);
    img1_points.push_back(pt.img1_);
    obj1_points.push_back(pt.obj1_);
    indices.push_back(index);
  }

  const float fx = cvK_.at<float>(0,0);
  cv::Mat cvDistortion = cv::Mat::zeros(4,1,cvK_.type());
  while(true) {
    if(obj1_points.size() < 10)
      break;
    cv::Mat rvec = cv::Mat::zeros(3,1,cvK_.type());
    cv::Mat tvec = cv::Mat::zeros(3,1,cvK_.type());

    cv::Mat inliers;
    cv::solvePnPRansac(obj1_points, img0_points, cvK_, cvDistortion, rvec, tvec,
                       true, 100, max_rprj_error,0.99,inliers);

    rprj_points.clear();
    cv::projectPoints(obj1_points, rvec, tvec, cvK_, cvDistortion, rprj_points);
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    tmp_img0_points.clear();
    tmp_img1_points.clear();
    tmp_obj1_points.clear();
    tmp_indices.clear();

    std::vector<int> step1_inliers;
    step1_inliers.reserve(rprj_points.size());
    for(size_t i = 0; i < rprj_points.size(); i++){
      const auto& uv0 = img0_points.at(i);
      const auto& obj_pt = obj1_points.at(i);
      const int& index = indices.at(i);

      bool inlier = false;
      if(inliers.empty()){
        const auto& pred = rprj_points.at(i);
        float rprj_err = cv::norm(uv0-pred);
        inlier = rprj_err < max_rprj_error;
      }
      else
        inlier = inliers.at<int>(i,0) ;

      const float& d0 = depth0.at<float>(std::round(uv0.y), std::round(uv0.x));
      if(inlier && d0 > 0.){
        const auto& obj_pt = obj1_points.at(i);
        cv::Mat Xw = (cv::Mat_<double>(3,1) << obj_pt.x, obj_pt.y, obj_pt.z );
        cv::Mat Xc = R.row(2)*Xw + tvec.at<double>(2);// {c} := {frame 0}
        float err = fx*base_*std::abs<double>( 1./d0 - 1./Xc.at<double>(0) );
        if(err > 3.) // disparity error [pixel]
          inlier = false;
      }

      if(!inlier){
        // outlier
        tmp_img0_points.push_back(uv0);
        tmp_img1_points.push_back(img1_points.at(i) );
        tmp_obj1_points.push_back(obj1_points.at(i) );
        tmp_indices.push_back(index);
      }
      else{
        step1_inliers.push_back(index);
      }
    } // for( ; i < rprj_points.size(); )

    if(step1_inliers.size() < 50)
      break;
    if(tmp_img0_points.size() < 50)
      break;

    // Euclidean cluster
    // Rusu, Radu Bogdan. "Semantic 3d object maps for everyday manipulation in human living environments."
    const flann::SearchParams param;

    std::vector<cv::Point2f> step1_img_points;
    std::vector<cv::Point3f> step1_obj_points;
    step1_img_points.reserve(step1_inliers.size());
    step1_obj_points.reserve(step1_inliers.size());
    for(const int& index : step1_inliers){
      const EpipPoint& pt = epip_points.at(index);
      step1_img_points.push_back(pt.img1_);
      step1_obj_points.push_back(pt.obj1_);
    }

    bool is_ground = n_cluster_ == 0; // TODO 이것보다 더 나은 기준이 필요.
    const int dim = is_ground ? 2 : 3;
    const float radius = is_ground? 20. : 0.5;
    const float r2 = radius*radius;

    float* ptr =  is_ground ? (float*)step1_img_points.data() : (float*)step1_obj_points.data();
    flann::Matrix<float> flann_points(ptr,
                                      step1_inliers.size(),
                                      dim);
    flann::Index< flann::L2<float> > flann_kdtree(flann_points, flann::KDTreeSingleIndexParams());
    flann_kdtree.buildIndex();

    // Euclidean cluster for unprocessed step1 inlier points.
    std::set<int> processed;
    for(int j = 0; j < step1_obj_points.size(); j++){
      if(processed.count(j))
        continue;

      std::set<int> flann_indices;
      std::set<int> Q = {j, };
      while( !Q.empty() ){
        int i = *Q.begin();
        Q.erase(i);
        flann_indices.insert(i);

        flann::Matrix<float> query(ptr+dim*i, 1, dim);
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > dists;
        flann_kdtree.radiusSearch(query, indices, dists, r2, param);
        for(const int& neighbor : indices.at(0) ){
          if(processed.count(neighbor) )
            continue;
          processed.insert(neighbor);
          Q.insert(neighbor);
        }
      } // while !Q.empty()

      if(flann_indices.size() >= 50){
        // Create new cluster
        // 이게 결과물을, std::map<int, std::vector<EpipPoint> > 에 입력하는 시점.
        std::vector<EpipPoint>& cluster_vector = clusters_[n_cluster_];
        cluster_ground_[n_cluster_] = n_cluster_== 0; // TODO 이것보다 더 나은 기준이 필요.
        std::cout << "cluster " <<  n_cluster_ << " inserted by RANSAC" << std::endl;

        cluster_vector.reserve(flann_indices.size());
        for(const int& flann_index : flann_indices){
          const int& point_index = step1_inliers.at(flann_index);
          EpipPoint& pt = epip_points.at(point_index);
          pt.cluster_ = n_cluster_;
          cluster_vector.push_back(pt);
        }
        n_cluster_++; // add cluster 1
      }
      else{
        // Outlier cluster
        for(const int& flann_index : flann_indices){
          const int& point_index = step1_inliers.at(flann_index);
          const EpipPoint& pt = epip_points.at(point_index);
          tmp_img0_points.push_back(pt.img0_);
          tmp_img1_points.push_back(pt.img1_);
          tmp_obj1_points.push_back(pt.obj1_);
          tmp_indices.push_back(point_index);
        }
      }
    } // for step1_obj_points

    int n_inlier = img0_points.size() - tmp_img0_points.size();
    if(n_inlier < 50)
      break;

    img0_points = tmp_img0_points;
    img1_points = tmp_img1_points;
    obj1_points = tmp_obj1_points;
    indices = tmp_indices;
  } //while(true)

  return;
}

cv::Mat DenseTracker::MakeMask(const int stride,
                               const int rows,
                               const int cols,
                               const std::map<int, EpipPoint>& epip_points) const {
  cv::Mat mask = - cv::Mat::ones(rows, cols, CV_32S); // int
  float radius = stride/2.;
  cv::Point2f offset(radius,radius);
  for(const auto& it : epip_points){
    const int& cluster = it.second.cluster_;
    const cv::Point2f& pt = it.second.img1_;
    const cv::Point2f pt0 = pt - offset;
    const cv::Point2f pt1 = pt + offset;
    cv::rectangle(mask, pt0, pt1, cluster, -1);
  }
  return mask;
}

void DenseTracker::EuclideanFilter(std::map<int, EpipPoint >& epip_points,
                                   std::set<int>& tracked_points) {
  const int dim = 3;
  const float radius = 0.5;
  const float r2 = radius*radius;
  const flann::SearchParams param;

  std::map<int, std::list<int> > clusters_before_filter;

  for(int pt_i : tracked_points){
    const int& k0 = epip_points.at(pt_i).cluster_;
    if( cluster_ground_.at(k0) ) // Ground cluster는 euclidean filter에서 제외.
      continue;
    clusters_before_filter[k0].push_back(pt_i);
  }


  for(auto it : clusters_before_filter){
    if(it.second.size() < 10)
      continue;
    const int k0 = it.first;
    std::vector<cv::Point3f> obj1_points;
    std::vector<int> flann2epip;
    obj1_points.reserve(it.second.size());
    flann2epip.reserve(it.second.size());
    for(const int& pt_i : it.second){
      const EpipPoint& pt = epip_points.at(pt_i);
      obj1_points.push_back(pt.obj1_);
      flann2epip.push_back(pt_i);
    }

    float* ptr = (float*) obj1_points.data();
    flann::Matrix<float> flann_points(ptr, obj1_points.size(), dim);
    flann::Index< flann::L2<float> > flann_kdtree (flann_points, flann::KDTreeSingleIndexParams());
    flann_kdtree.buildIndex();

    std::set<int> processed;
    std::map<int, std::set<int> > sub_cluster;
    int tmp_k = 0;
    for(int j = 0; j < epip_points.size(); j++){
      if(processed.count(j))
        continue;
      std::set<int> flann_indices;
      std::set<int> Q = {j, };
       while( !Q.empty() ){
        int i = *Q.begin();
        Q.erase(i);
        flann_indices.insert(i);

        flann::Matrix<float> query(ptr+dim*i, 1, dim);
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > dists;
        flann_kdtree.radiusSearch(query, indices, dists, r2, param);
        for(const int& neighbor : indices.at(0) ){
          if(processed.count(neighbor) )
            continue;
          processed.insert(neighbor);
          sub_cluster[tmp_k].insert(neighbor);
          Q.insert(neighbor);
        }
      } // while !Q.empty()
      tmp_k++;
    }
    // 분리된 sub cluster 중에서 가까운 feature의 숫자가 많은 cluster를 기존 cluster에 남기고, 나머지는 new cluster로 입력
    std::vector< std::pair<int, size_t> > n_near_points;
    n_near_points.reserve(sub_cluster.size());
    for(const auto& it_sub : sub_cluster){
      size_t n = 0;
      for(const int& flann_index : it_sub.second){
        const cv::Point3f& pt = obj1_points.at(flann_index);
        if(pt.z < 50.) // [meter] TODO parameterize definition of "near"
          n++;
      }
      n_near_points.push_back(std::make_pair(it_sub.first, n));
    }

    std::sort(n_near_points.begin(), n_near_points.end(),
              [](const std::pair<int, size_t>& a,
                 const std::pair<int, size_t>& b) {return a.second > b.second; });

    for(const auto& it_sub : n_near_points){
      int tmp_k = it_sub.first;
      int k = k0;
      const std::set<int>& sub_cluster_pt_indices = sub_cluster.at(tmp_k);
      if(it_sub.second < 20) // TODO Parameterize 'minimum number of near points'
        k = -1;
      //else if(it_sub.first != n_near_points.begin()->first ){
      //  k = n_cluster_; // add cluster 2
      //  cluster_ground_[k] = false;
      //  n_cluster_++;
      //}

      for(const int& flann_i : sub_cluster_pt_indices){
        const int& pt_i = flann2epip.at(flann_i);
        epip_points[pt_i].cluster_ = k;
      }

      if(k < 0)
        for(const int& flann_i : sub_cluster_pt_indices){
          const int& pt_i = flann2epip.at(flann_i);
          tracked_points.erase(pt_i);
        }
    }
  } // for(auto it : clusters_before_filter)
  return;
}

void DenseTracker::TrackCluster(const cv::Mat flow,
                                const cv::Mat disparity1,
                                const cv::Mat depth0,
                                const cv::Mat depth1,
                                std::set<int>& unclustered,
                                std::map<int, EpipPoint>& sampled_epip_points
                               ) {
  const float max_rprj_error = 2.;

  // 1. Sampled points에 대해, optical flow가 가리키는 prev_mask에 해당하는 candidate cluster를 확인,
  std::map<int, std::list<std::pair<int,EpipPoint> > > track_candidate;
  for(auto& it : sampled_epip_points){
    const auto& epip_point = it.second;
    const cv::Point2f& pt0 = epip_point.img0_;
    const cv::Point2f& pt1 = epip_point.img1_;
    int r0 = pt0.y;
    int c0 = pt0.x;
    if(r0 < 0 || r0 >= flow.rows)
      continue;
    if(c0 < 0 || c0 >= flow.cols)
      continue;
    const int& k0 = mask0_.at<int>(r0, c0);
    if(k0 < 0)
      continue;
    // k0가 candidate cluster일 가능성이 있다.
    track_candidate[k0].push_back( std::make_pair(it.first, epip_point) );
  }

  const float fx = cvK_.at<float>(0,0);
  cv::Mat cvDistortion = cv::Mat::zeros(4,1,cvK_.type());
  std::set<int> tracked_points;
  std::map<int, cv::Mat> rvecs, tvecs;
  for(auto& it : track_candidate){
    // 2. 각 클러스터에 대해..
    const int& k0 = it.first;
    const auto& lists = it.second;
    if(lists.size() < 20)
      continue;
    std::vector<int> sampled_point_indices;
    std::vector<cv::Point2f> img0_points;
    std::vector<cv::Point3f> obj1_points;
    img0_points.reserve(lists.size());
    obj1_points.reserve(lists.size());
    sampled_point_indices.reserve(lists.size());
    for(const auto& it_list : lists){
      img0_points.push_back(it_list.second.img0_);
      obj1_points.push_back(it_list.second.obj1_);
      sampled_point_indices.push_back(it_list.first);
    }
    // 2-1. solvePnPRansac의 rvec, tvec을 계산.
    cv::Mat rvec = cv::Mat::zeros(3,1,cvK_.type());
    cv::Mat tvec = cv::Mat::zeros(3,1,cvK_.type());
    cv::Mat inliers;
    {
      bool is_ground = cluster_ground_.at(k0);
      cv::solvePnPRansac(obj1_points, img0_points, cvK_, cvDistortion, rvec, tvec,
                         true, is_ground?200:100, max_rprj_error,0.99,inliers);
    }
    // 2-2. inlier sampled points의 cluster를 업데이트.
    int n_inlier = 0;
    for(int i = 0; i < inliers.rows; i++)
      if( inliers.at<int>(i,0) )
        n_inlier++;
    if(n_inlier < 20) // TODO Parameterize the magic number
      continue;
    rvecs[k0] = rvec;
    tvecs[k0] = tvec;
    for(int i = 0; i < inliers.rows; i++){
      bool b = inliers.at<int>(i,0);
      if(!b)
        continue;
      const int& pt_index = sampled_point_indices.at(i);
      sampled_epip_points[pt_index].cluster_ = k0;
      tracked_points.insert(pt_index);
    }
  }

  // 3. rprj error가 threshold 이하가 되는 rvec,tvec과 같은 cluster로 point 소속을 변환.
  // Oversegment를 막는 역할.
  {
    // 이때는 euclidean clustering 안하고 있음. 논리적으로 문제가 될 순 있을것같은데,.
    // 어차피 이후 step에 Euclidean filter가 있으니, 일단은 미뤄둠.
    const float max_rprj2 = max_rprj_error * max_rprj_error;
    std::vector<cv::Point2f> rprj_points;
    for(auto& it_track : track_candidate){
      const int& k0 = it_track.first;
      if(! cluster_ground_.at(k0) )
        continue;
      if(! rvecs.count(k0) )
        continue;
      const cv::Mat& rvec = rvecs.at(k0);
      const cv::Mat& tvec = tvecs.at(k0);

      // Check rprj error for tvec, rvec
      for(auto& it_pt : sampled_epip_points){
        if(tracked_points.count(it_pt.first) )
          continue;
        const std::vector<cv::Point3f> obj1_points = { it_pt.second.obj1_};
        rprj_points.clear();
        cv::projectPoints(obj1_points, rvec, tvec, cvK_, cvDistortion, rprj_points);
        cv::Point2f err = rprj_points.at(0) - it_pt.second.img0_;
        if( err.dot(err) > max_rprj2)
          continue;
        it_pt.second.cluster_ = k0;
        tracked_points.insert(it_pt.first);
      }
    }
  }

  {
    // 4. tracked sample points를 KDTree로 구현.
    std::vector<cv::Point3f> obj1_points;
    std::vector<int> epip_indices;
    obj1_points.reserve(tracked_points.size());
    epip_indices.reserve(tracked_points.size());
    for(const int& index : tracked_points){
      const EpipPoint& pt = sampled_epip_points.at(index);
      obj1_points.push_back(pt.obj1_);
      epip_indices.push_back(index);
    }
    if(obj1_points.size() > 10){
      float* ptr = (float*) obj1_points.data();
      flann::Matrix<float> flann_points(ptr, obj1_points.size(), 3);
      flann::Index< flann::L2<float> > flann_kdtree (flann_points, flann::KDTreeSingleIndexParams());
      flann_kdtree.buildIndex();

      // 5. untracked sampled points에 대해, 2D nearest neighbor를 찾고, 그 거리가 threshold 이하일 경우 해당 클러스터로 편입.
      const flann::SearchParams param;
      for(auto& it_pt : sampled_epip_points){
        if(tracked_points.count(it_pt.first) )
          continue;
        flann::Matrix<float> query( (float*) &it_pt.second.obj1_, 1, 3);
        std::vector<std::vector<int> > indices;
        std::vector<std::vector<float> > dists;
        flann_kdtree.knnSearch(query, indices, dists, 1, param);
        const int& nearest_pt_idex = epip_indices.at(indices.at(0).at(0));
        const float& distance = dists.at(0).at(0);
        if(distance > 0.5) // [meter] TODO Parameterize the magic number
          continue;
        const int& k0 = sampled_epip_points.at(nearest_pt_idex).cluster_;
        it_pt.second.cluster_ = k0;
        tracked_points.insert(it_pt.first);
      }
    }
  }

  // 6. Euclidean clustering을 통한 필터링.
  const int n_cluster0 = n_cluster_;
  EuclideanFilter(sampled_epip_points, tracked_points);


  { 
    // 7. tracked points와 업데이트된 cluster index를 clusters에 반영.
    std::map<int, std::set<int> > cluster_indecies;
    for(const int& pt_index : tracked_points){
      const EpipPoint& epip_pt = sampled_epip_points.at(pt_index);
      cluster_indecies[epip_pt.cluster_].insert(pt_index);
    }
    clusters_.clear();
    for(const auto& it : cluster_indecies){
      const int& k = it.first;
      const std::set<int>& indecies = it.second;
      clusters_[k].reserve( indecies.size() );

      for(const int& pt_i : indecies){
        const EpipPoint& pt = sampled_epip_points.at(pt_i);
        clusters_[k].push_back(pt);
      }
    }
  }

  for(const int& pt_index : tracked_points)
    unclustered.erase(pt_index);

  return;
}

void DenseTracker::Track(cv::Mat gray, cv::Mat gray_r) {
  cv::Mat edge = GetEdge(gray);
  typedef float DisparityType;
  float fx = cvK_.at<float>(0,0);

  cv::cuda::GpuMat g_gray;
  g_gray.upload(gray);

#if 0
  cv::Mat disparity1 = GetDisparity<DisparityType>(gray, gray_r, edge);
#else
  cv::cuda::GpuMat g_gray_r;
  g_gray_r.upload(gray_r);
  cv::Mat disparity1 = GetDisparity<DisparityType>(g_gray, g_gray_r, edge);
#endif
  if(depth0_.empty()){
    depth0_ = Disparity2Depth<DisparityType>(fx, base_, disparity1);
    gray0_ = gray;
    g_gray0_.upload(gray0_);
    return;
  }

  cv::Mat dst;
  cv::vconcat(gray, gray_r, dst);
  cv::imshow("left_right", dst );

  cv::Mat flow = GetFlow(g_gray0_, g_gray);
  float mag = SampleMagnitude(flow, 0.95);
  if(mag < 20.)
    return;

  cv::Mat depth1 = Disparity2Depth<DisparityType>(fx, base_, disparity1);

  // Step 0) 모든 pixel을 계산할 순 없으므로 points sampling.
  const int stride = 5;
  std::map<int, EpipPoint> epip_points;

  SamplePoints<DisparityType>(stride, flow, disparity1, depth1,
                              epip_points);

  // 1) 이전 프레임에서 분류된 cluster를 먼저 tracking
  std::set<int> unclustered;
  for(auto it : epip_points)
    unclustered.insert(it.first);

  if(! mask0_.empty() )
    TrackCluster(flow, disparity1, depth0_, depth1, unclustered, epip_points);

  // 2) 남은 points에 대해 Ransac으로 분류, cluster에 추가
  RansacCluster(flow, disparity1, depth0_, depth1, unclustered, epip_points);

  // 3) epip_points를 mask로 변환.
  mask0_= MakeMask(stride, flow.rows, flow.cols, epip_points);


  depth0_ = depth1;
  gray0_ = gray;
  g_gray0_.upload(gray0_);

  cv::imshow("edge", edge);
#if 1
  cv::Mat vis_disparity = cv::Mat::zeros(disparity1.rows,disparity1.cols,CV_8UC1);
  for(int r= 0; r< disparity1.rows; r++)
    for(int c= 0; c< disparity1.cols; c++){
      float v = (float) disparity1.at<DisparityType>(r,c);
      vis_disparity.at<unsigned char>(r,c) = std::min<float>(254, std::max<float>(0, v));
    }
  cv::imshow("disparity", vis_disparity);
#endif

  {
    // Tracking 없이 cluster visualization
    cv::Mat dst;
    cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);
    for(auto it : epip_points){
      int k = it.second.cluster_;
      const cv::Point2f& pt = it.second.img1_;
      cv::Scalar color;
      if(k < 0)
        color = CV_RGB(200,200,200);
      else
        color =  colors.at(k % colors.size() );
      cv::circle(dst, pt, 3, color, 1);
    }
    cv::imshow("dst", dst);
  }

  cv::imshow("mask", VisulaizeMask(mask0_) );
  return;
}
