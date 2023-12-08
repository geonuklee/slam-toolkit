#include "floorseg.h"
#include "util.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

cv::Mat Floorseg::Put(const cv::Mat depth,
                      const cv::Mat vis_rgb
                     ) {
  const int y0 = 100; // [pixel]
  const int interval = 8; // [pixel]
  const int half_interval = 4;
  const float init_threshold   = .2; // [meter]
  const float input_threshold  = .5; // [meter]
  const float output_threshold = .3;
  const int xoffset = 200;
  int interval2 = interval*interval;

  static bool init = false;
  static int s_tol = 100;
  static int s_z   = 100;
  static int s_u   = 100;
  if( (!vis_rgb.empty()) && (!init) ){
    cv::imshow("dst_segment", depth);
    cv::createTrackbar("s_tol", "dst_segment", &s_tol, 1000);
    cv::createTrackbar("s_u", "dst_segment", &s_u, 1000);
    cv::createTrackbar("s_z", "dst_segment", &s_z, 1000);
    init = true;
  }

#if 1
  const float ec_tolerance = 0.5; // [meter] 그냥 물리적 거리로 clustering하는게 최선.
#else
  const float ec_tolerance = 0.01 * float(s_tol) / 100.;
  const float _s_u         =  40. / fx_ * float(s_u)   / 100.;
  const float _s_z         =  5./base_line_ * float(s_z)   / 100.;
  //const float _s_z         =  2./base_line_ * float(s_z)   / 100.; // log
#endif


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<cv::Point> vec_uv;
  {
    cloud->points.reserve(depth.rows*depth.cols/interval2);
    vec_uv.reserve(cloud->points.capacity());
    for(int r=0; r<depth.rows; r+=interval){
      for(int c=0; c<depth.cols; c+=interval){
        pcl::PointXYZ pt;
        pt.z = depth.at<float>(r,c);
        if(pt.z < 1e-5)
          continue;
        pt.x = pt.z * (float(c)-cx_) / fx_;
        pt.y = pt.z * (float(r)-cy_) / fy_;
        cloud->push_back(pt);
        vec_uv.push_back(cv::Point(c,r));
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>);
  {
    pcl::PointIndices::Ptr c0_in_c(new pcl::PointIndices); // cloud0 in cloud
    c0_in_c->indices.reserve(vec_uv.size());
    for(int i=0; i<vec_uv.size(); i++){
      const cv::Point& uv = vec_uv.at(i);
      if(uv.y < depth.rows - y0)
        continue;
      if(uv.x < xoffset || uv.x > depth.cols-xoffset)
        continue;
      c0_in_c->indices.push_back(i);
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(c0_in_c);
    extract.filter(*cloud0);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointIndices::Ptr c1_in_c(new pcl::PointIndices); // cloud1 in cloud
  {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr init_coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(init_threshold);
    seg.setInputCloud(cloud0);
    seg.setMaxIterations(100);
    seg.segment(*inliers, *init_coefficients);
    const std::vector<float>& coef = init_coefficients->values; // [normal_x normal_y normal_z d]
    c1_in_c->indices.reserve(cloud->points.size());
    for(int i=0; i< cloud->points.size(); i++){
      const auto& pt = cloud->points.at(i);
      float err = coef[0]*pt.x + coef[1]*pt.y + coef[2]*pt.z + coef[3];
      if(std::abs(err) > input_threshold)
        continue;
      c1_in_c->indices.push_back(i);
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(c1_in_c);
    extract.filter(*cloud1);
  }

  pcl::PointIndices::Ptr plane_outliers(new pcl::PointIndices);
  pcl::PointCloud<pcl::PointXYZ>::Ptr normalized_cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  normalized_cloud_f->points.reserve(cloud->points.size());
  {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(output_threshold);
    seg.setInputCloud(cloud1);
    seg.setMaxIterations(100);
    seg.segment(*inliers, *coefficients);
    const std::vector<float>& coef = coefficients->values; // [normal_x normal_y normal_z d]
    // inlier가 아니라, inverse depth에 대한 chi2를 계산해야겠다..
    for(int i=0; i < cloud->points.size(); i++){
      const pcl::PointXYZ& pt = cloud->points.at(i);
      //float err = coef[0]*pt.x + coef[1]*pt.y + coef[2]*pt.z + coef[3];
      //if(std::abs(err) > .5)
      //  plane_outliers->indices.push_back(i);
      float x = pt.x / pt.z;
      float y = pt.y / pt.z;
      //0 = coef[0]*xz + coef[1]*yz + coef[2]*z + coef[3] , x,y는 각각 normalized image point
      //0 = (coef[0]*x + coef[1]*y + coef[2])*z + coef[3];
      //z = -coef[3] / (coef[0]*x + coef[1]*y + coef[2])
      //float z = -coef[3] / (coef[0]*x + coef[1]*y + coef[2]);
      float h_inv = - (coef[0]*x + coef[1]*y + coef[2]) / coef[3];
      float z_inv = 1./pt.z;
      if( std::abs(h_inv - z_inv) < 1e-2) // parameterize
        continue;
      plane_outliers->indices.push_back(i);
      pcl::PointXYZ npt;
#if 1
      npt.x = pt.x;
      npt.y = pt.y;
      npt.z = pt.z;
#else
      npt.x = _s_u*x;
      npt.y = _s_u*y;
      //npt.z = _s_z*std::log(z_inv);
      npt.z = _s_z*z_inv;
#endif
      normalized_cloud_f->points.push_back(npt);
    }
  }

  std::vector<pcl::PointIndices> cluster_indices; {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(normalized_cloud_f);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(ec_tolerance);
    ec.setMinClusterSize(5);
    //ec.setMaxClusterSize(1000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(normalized_cloud_f);
    ec.extract(cluster_indices);
  }

  int n_segment = 0;
  marker_ = cv::Mat::zeros(depth.size(), CV_32S); {
    for(const pcl::PointIndices& indices : cluster_indices){
      n_segment++;
      for(const int& io : indices.indices){
        const int& i = plane_outliers->indices.at(io);
        cv::Point uv = vec_uv.at(i) - cv::Point(half_interval, half_interval);
        cv::Point uv2(uv.x+interval, uv.y+interval);
        cv::rectangle(marker_, uv, uv2, n_segment, -1);
      } // for indices
    }
  }

  if(!vis_rgb.empty()){
    cv::Mat dst = GetColoredLabel(marker_,true);
    dst.setTo(CV_RGB(0,0,0), GetBoundary(marker_,4));
    cv::addWeighted(vis_rgb, .3, dst, .5, 1., dst);
    cv::imshow("dst_segment", dst);
  }

  marker_ += 1;
  return marker_;
}
