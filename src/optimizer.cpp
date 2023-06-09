#include "optimizer.h"
#include "frame.h"
#include "mappoint.h"
#include "camera.h"
#include "posetracker.h"
#include "orb_extractor.h"
#include "localmapper.h"
#include "pipeline_map.h"

#if defined(_OPENMP)
  Uncheck openmp (such as compile flag '-fopenmp'). g2o optimizer return wrong results with it.
  // Direct method에만 한정인것으로보여 골치. (마찬가지로 OpenMP를 쓰고있을) opencv image를 g2o Edge에서 불러오는 과정에 문제생기나보다.
#endif

DirectPyramid::DirectPyramid()
 :  ratio_(0.6), lv_(0), scale_(1.) {

}

void DirectPyramid::PutFrame(const Frame* frame) {
  if(org_gray_.count(frame))
    return;
  org_gray_[frame] = frame->GetImage();
}

void DirectPyramid::SetLv(int lv) {
  lv_ = lv;
  scale_ = std::pow(ratio_, lv_);
  cv::Mat im0 =  (*org_gray_.begin()).second;
  cv::Size dsize(im0.cols*scale_, im0.rows*scale_);
  for(auto it : org_gray_){
    cv::Mat dst;
    cv::resize(it.second, dst, dsize, cv::INTER_LINEAR);
    lv_gray_[it.first] = dst;
  }
  return;
}

void Pattern::GetPattern(int i, int& dx, int& dy) {
  switch(i){
    case 0: dx = 0; dy = 0; break;
    case 1: dx = 1; dy = -1; break;
    case 2: dx = -1; dy = 1; break;
    case 3: dx = 1; dy = 1; break;
    case 4: dx = 0; dy = 2; break;
    case 5: dx = 2; dy = 0; break;
    case 6: dx = 0; dy = -2; break;
    case 7: dx = -2; dy = 0; break;
    default:
      throw 1;
  }
  return;
}

double Pattern::GetRadius() {
  return 2.;
}

void VertexBrightenSE3::oplusImpl(const number_t* update_)  {
  Eigen::Map<const g2o::Vector6> xi(update_);
  Eigen::Map<const g2o::Vector2> delta_brightness(update_+6);
  BrightenSE3 est;
  est.Tcw_ = g2o::SE3Quat::exp(xi)*estimate().Tcw_;
  est.brightness_ += delta_brightness;
  setEstimate(est);
}

void VertexBrightenSE3::setToOriginImpl() {
  BrightenSE3 brightness;
  brightness.brightness_.setZero();
  setEstimate(brightness);
  return;
}

EdgeBrightenessPrior::EdgeBrightenessPrior(double info_a, double info_b) {
  Eigen::Matrix<double,2,2> info;
  info << info_a, 0., 0., info_b;
  setInformation(info);

  auto z = measurement();
  z.setZero();
  setMeasurement(z);
}

void EdgeBrightenessPrior::computeError() {
  const VertexBrightenSE3* v0 = static_cast<const VertexBrightenSE3*>(_vertices[0]);
  Eigen::Vector2d est = v0->estimate().brightness_;
  _error = est - Eigen::Vector2d(0,0);
  return;
}

void EdgeBrightenessPrior::linearizeOplus() {
  _jacobianOplusXi.setOnes();
  return;
}

EdgeProjectBrightenXYZ::EdgeProjectBrightenXYZ(DirectPyramid* pyramid,
                                               const Frame* frame,
                                               const Mappoint* mp)
: pyramid_(pyramid), frame_(frame), mp_(mp)
{
  pyramid->PutFrame(frame);
  pyramid->PutFrame(mp->GetRefFrame() );
  auto info = this->information();
  info.setIdentity();
  setInformation(info);
}

bool EdgeProjectBrightenXYZ::GetError(const cv::Mat& gray0,
                                      const cv::Mat& gray,
                                      const BrightenSE3& brighten_pose0,
                                      const Mappoint* mp,
                                      double image_scale,
                                      const BrightenSE3& brighten_pose,
                                      const Eigen::Vector3d Xw,
                                      Measurement& error) {
  const Frame* frame0 = mp->GetRefFrame();
  const Camera* camera = frame0->GetCamera();
  cv::KeyPoint kpt0 = frame0->GetKeypoint(mp);
  Eigen::Vector2d cp0(kpt0.pt.x, kpt0.pt.y);
  cp0 *= image_scale;
  const Eigen::Vector3d Xc = brighten_pose.Tcw_*Xw;
  Eigen::Vector2d cp = camera->Project(Xc);
  cp *= image_scale;

  //std::cout << "brighten_pose = \n " << brighten_pose.Tcw_ << std::endl;

  const Eigen::Vector2d& brightness = brighten_pose.brightness_;
  const Eigen::Vector2d& brightness0 = brighten_pose0.brightness_;
  const double exp_a  = std::exp(-brightness(0));
  const double exp_a0 = std::exp(-brightness0(0));
  const double& b = brightness(1);
  const double& b0 = brightness0(1);

  double patch_scale = mp->GetDepth()/Xc.z();

  g2o::SE3Quat Tcr = brighten_pose.Tcw_ * brighten_pose0.Tcw_.inverse();
  Eigen::Matrix2d s_Rcr = patch_scale*Tcr.rotation().toRotationMatrix().block<2,2>(0,0);

  bool has_computation = false;
  error.setZero();
  for(int n = 0; n < Dimension; n++){
    int dx, dy;
    Pattern::GetPattern(n, dx, dy);
    Eigen::Vector2d dxy(dx,dy);
    Eigen::Vector2d uv0 = cp0 + dxy;
    if(uv0.x() < 0 || uv0.y() < 0 || uv0.x() >= gray0.cols || uv0.y() >= gray0.rows)
      continue;
    double i0 = GetInetrpolatedIntensity(gray0, uv0);

    Eigen::Vector2d uv = cp + s_Rcr*dxy;
    if(uv.x() < 0 || uv.y() < 0 || uv.x() >= gray.cols || uv.y() >= gray.rows)
      continue;

    double i = GetInetrpolatedIntensity(gray, uv);
    error(n,0) = exp_a*(i-b) - exp_a0*(i0-b0);
    has_computation = true;
  }
  return has_computation;
}

void EdgeProjectBrightenXYZ::computeError() {
  const g2o::VertexSBAPointXYZ* v0 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const VertexBrightenSE3* v1 = static_cast<VertexBrightenSE3*>(_vertices[1]);
  double image_scale = pyramid_->GetScale();

  const Frame* ref_frame = mp_->GetRefFrame();
  cv::Mat gray0 = pyramid_->GetLvGray(ref_frame);
  cv::Mat gray  = pyramid_->GetLvGray(frame_);

  BrightenSE3 brighten_pose0(ref_frame->GetTcw(), ref_frame->GetBrightness());
  GetError(gray0, gray, brighten_pose0, mp_, image_scale,
           v1->estimate(), v0->estimate(), _error);
  return;
}

void EdgeProjectBrightenXYZ::linearizeOplus() {
  _jacobianOplusXi.setZero(); // It related to pose tracking.
  _jacobianOplusXj.setZero(); // It related to mapping
  const g2o::VertexSBAPointXYZ* v0 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const VertexBrightenSE3* v1 = static_cast<const VertexBrightenSE3*>(_vertices[1]);
  const g2o::SE3Quat& Tcw = v1->estimate().Tcw_;
  const Eigen::Vector2d& brightness = v1->estimate().brightness_;
  const Eigen::Vector3d& Xw = v0->estimate();
  const Eigen::Vector3d Xc = Tcw * Xw;
  const Camera* camera = frame_->GetCamera();
  double image_scale = pyramid_->GetScale();

  const Frame* ref_frame = mp_->GetRefFrame();
  cv::KeyPoint kpt0 = ref_frame->GetKeypoint(mp_);
  Eigen::Vector2d cp0(kpt0.pt.x, kpt0.pt.y);
  Eigen::Vector2d cp = camera->Project(Xc);
  cp0 *= image_scale;
  cp *= image_scale;

  g2o::SE3Quat Tcr = Tcw * ref_frame->GetTcw().inverse();
  cv::Mat gray = pyramid_->GetLvGray(frame_);

  const Eigen::Matrix<double,2,2>& du_dx = camera->GetK().block<2,2>(0,0);

  Eigen::Matrix<double,2, 3> dx_dXc;
  {
    double invZ = 1./Xc.z();
    double invZ2 = invZ*invZ;
    dx_dXc << invZ, 0., -Xc.x() * invZ2,
              0., invZ, -Xc.y() * invZ2;
  }

  Eigen::Matrix<double,3,6> dXc_dxi;
  {
    // Domain of g2o::SE3Quat : [omega, upsilon]
    dXc_dxi.block<3,3>(0,0)  = -g2o::skew(Xc);
    dXc_dxi.block<3,3>(0,3).setIdentity();
  }

  const double exp_a = std::exp(-brightness(0));
  const double& b = brightness(1);
  const Eigen::Matrix<double,3,3> Rcw = Tcw.rotation().toRotationMatrix();
  // exp_a and 's'cale is not related to du_dXc. But multiply those it before for-loop to reduce computational cost.
  const Eigen::Matrix<double,2,3> du_dXc = exp_a * image_scale * du_dx * dx_dXc;
  const Eigen::Matrix<double,2,6> du_dxi = du_dXc * dXc_dxi;
  const Eigen::Matrix<double,2,3> du_dXw = du_dXc * Rcw;

  double patch_scale = mp_->GetDepth()/Xc.z();
  Eigen::Matrix2d s_Rcr = patch_scale*Tcr.rotation().toRotationMatrix().block<2,2>(0,0);
  //s_Rcr.setIdentity();

  for(int n = 0; n < Dimension; n++){
    int dx, dy;
    Pattern::GetPattern(n, dx, dy);
    Eigen::Vector2d uv = cp + s_Rcr*Eigen::Vector2d(dx,dy);
    if(uv.x() < 0 || uv.y() < 0 || uv.x() >= gray.cols || uv.y() >= gray.rows)
      continue;
    double i = GetInetrpolatedIntensity(gray, uv);
    cv::Point2i iuv(uv.x(), uv.y());
    double ix = (gray.at<unsigned char>(iuv.y,iuv.x+1) - gray.at<unsigned char>(iuv.y,iuv.x-1))/2.;
    double iy = (gray.at<unsigned char>(iuv.y+1,iuv.x) - gray.at<unsigned char>(iuv.y-1,iuv.x))/2.;
    Eigen::Matrix<double,1,2> grad;
    grad << ix, iy;

    _jacobianOplusXi.block<1,3>(n,0) = grad * du_dXw;
    _jacobianOplusXj.block<1,6>(n,0) = grad * du_dxi;
    _jacobianOplusXj(n,6) = -exp_a * (i-b);
    _jacobianOplusXj(n,7) = -exp_a;
  }

  return;
}

class AlignEdge : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  AlignEdge(Eigen::Vector3d xyz)
    : xyz_(xyz)
  {

  }
  bool read(std::istream& is){ return false; }
  bool write(std::ostream& os) const{ return false; }

  void computeError()  {
    const g2o::VertexSE3Expmap* v = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    _error = v->estimate().map(_measurement) - xyz_;
  }

private:
  const Eigen::Vector3d xyz_;

};

void EdgeSE3::computeError(){
  const g2o::VertexSE3Expmap* v0 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
  const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  const g2o::SE3Quat& Tc0w = v0->estimate();
  const g2o::SE3Quat& Tc1w = v1->estimate();
  g2o::SE3Quat C(_measurement);
  g2o::SE3Quat Tc1c0 = Tc1w*Tc0w.inverse();
  g2o::SE3Quat error_= Tc1c0 * C.inverse();
  _error = error_.log();
}

g2o::SE3Quat AlignTrajectory(const EigenMap<int, g2o::SE3Quat>& gtTcws,
                             const std::map<int, Frame*>& frames,
                             const g2o::SE3Quat Tgt_est_prior
                             ){
  g2o::SparseOptimizer optimizer;
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver
    = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg* solver
    =  new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
  solver->setUserLambdaInit(1e-16);
  optimizer.setAlgorithm(solver);

  g2o::VertexSE3Expmap* vertex0 = new g2o::VertexSE3Expmap();
  vertex0->setId(0);
  vertex0->setEstimate(Tgt_est_prior);
  optimizer.addVertex(vertex0);
  vertex0->setFixed(true);

  g2o::VertexSE3Expmap* vertex1 = new g2o::VertexSE3Expmap();
  vertex1->setId(1);
  vertex1->setEstimate(g2o::SE3Quat());
  optimizer.addVertex(vertex1);


  Eigen::Vector3d t0;
  for(auto it_frame : frames){
    Frame* frame = it_frame.second;
    int idx = frame->GetIndex();
    if(idx==0)
      continue;
    g2o::SE3Quat gtTwc = gtTcws.at(idx).inverse();
    g2o::SE3Quat estTwc = frame->GetTcw().inverse();
    Eigen::Vector3d t1 = estTwc.translation();

    if( (t1-t0).norm() < 30.) // Interval 30[m]
      continue;
    t0 = t1;
    AlignEdge* edge = new AlignEdge(gtTwc.translation());
    edge->setVertex(0, vertex1);
    edge->setMeasurement(t1);
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  {
    auto edge = new EdgeSE3();
    edge->setVertex(0, vertex0);
    edge->setVertex(1, vertex1);
    g2o::SE3Quat z;
    edge->setMeasurement(z);
    Eigen::Matrix<double,6,6> wv;
    wv.setIdentity();
    edge->setInformation(wv);
    optimizer.addEdge(edge);
  }

  size_t n_iter = 10;
  optimizer.initializeOptimization();
  optimizer.optimize(n_iter);
  // Map estimation to ground truth cooridnate
  g2o::SE3Quat Tgt_est = vertex1->estimate();
  return Tgt_est;
}

cv::Mat Frame::PlotProjection(const std::set<Mappoint*>& mappoints) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!estimation_){
    throw std::invalid_argument("Try to plotmatch before estimate pose");
  }

  int sample_width = 5;
  int show_width = 50;
  cv::Size sample_size(sample_width,sample_width);
  cv::Size show_size(show_width,show_width);

  int n_cols = gray_.cols / show_width;
  int n_rows = 4;

  cv::Mat dst = cv::Mat::zeros(gray_.rows + 3*n_rows*show_width, gray_.cols, CV_8UC3);
  cv::Mat tmp = dst(cv::Rect(0,0,gray_.cols,gray_.rows));
  cv::cvtColor(gray_, tmp, cv::COLOR_GRAY2BGR);

  int n = 0;
  double SUM_i = 0;
  double SUM_b = 0;

  std::map<const Frame*, cv::Mat> rgbs, grayes;
  for(Mappoint* mp : mappoints){
    Frame* ref_frame = mp->GetRefFrame();
    if(ref_frame != this){
      cv::Mat gray = ref_frame->GetImage();
      cv::cvtColor(gray, rgbs[ref_frame], cv::COLOR_GRAY2BGR);
      grayes[ref_frame] = gray;
    }
  }
  cv::cvtColor(gray_, rgbs[this], cv::COLOR_GRAY2BGR);
  grayes[this] = gray_;

  for(Mappoint* mp : mappoints){
    if(n == n_rows*n_cols)
      break;
    Frame* ref_frame = mp->GetRefFrame();
    if(ref_frame == this)
      continue;
    Eigen::Vector3d Xc = estimation_->Tcw_*mp->GetXw();
    Eigen::Vector2d uv = camera_->Project(Xc);
    cv::Point2f pt(uv.x(),uv.y());
    if(pt.x < sample_width || pt.y < sample_width || pt.x > gray_.cols-sample_width || pt.y > gray_.rows-sample_width)
      continue;

    cv::Point2f kpt0 = ref_frame->GetKeypoint(mp).pt;
    cv::Mat im0 = rgbs.at(ref_frame);
    cv::Mat im = rgbs.at(this);
    cv::Mat patch0, patch;
    cv::getRectSubPix(im0, sample_size, kpt0, patch0);
    cv::getRectSubPix(im, sample_size, pt, patch);

    const auto& brightness0 = ref_frame->GetBrightness();
    const auto& brightness = estimation_->brightness_;
    const double exp_a  = std::exp(-brightness(0));
    const double exp_a0 = std::exp(-brightness0(0));
    const double& b = brightness(1);
    const double& b0 = brightness0(1);

    cv::Mat err_i = cv::Mat::zeros(sample_width, sample_width, CV_8UC1);
    cv::Mat err_b = cv::Mat::zeros(sample_width, sample_width, CV_8UC1);

    double sum_i = 0;
    double sum_b = 0;
    for(int i = 0; i < sample_width; i++){
      for(int j = 0; j < sample_width; j++){
        double intensity0 = patch0.at<cv::Vec3b>(j,i)[0];
        double intensity  = patch.at< cv::Vec3b>(j,i)[0];
        double e = std::abs<double>( exp_a*(intensity-b) - exp_a0*(intensity0 -b0) );
        double ei = std::abs<double>(intensity-intensity0);

        sum_b +=  e;
        sum_i +=  ei;

        err_b.at<unsigned char>(j,i) = std::min<double>(255,e);
        err_i.at<unsigned char>(j,i) = std::min<double>(255,ei);
      }
    }
    //std::cout << "i,b = " << sum_i << ", " << sum_b << std::endl;
    SUM_i += sum_i;
    SUM_b += sum_b;

    int c = n % n_cols;
    int r = n / n_cols;
    n++;

    cv::Rect rect0(c*show_width, gray_.rows+r*3*show_width,     show_width,show_width);
    cv::Rect rect1(c*show_width, gray_.rows+(r*3+1)*show_width, show_width,show_width);
    cv::Rect rect2(c*show_width, gray_.rows+(r*3+2)*show_width, show_width,show_width);

    cv::resize(patch0, patch0, show_size);
    cv::resize(patch, patch, show_size);
    cv::resize(err_b, err_b, show_size);
    cv::cvtColor(err_b, err_b, cv::COLOR_GRAY2BGR);

    EdgeProjectBrightenXYZ::Measurement error;
    const cv::Mat& gray0 = grayes.at(ref_frame);
    const cv::Mat& gray = grayes.at(this);
    const BrightenSE3 brighten_pose0( ref_frame->GetTcw(), ref_frame->GetBrightness());
    bool has_computation = EdgeProjectBrightenXYZ::GetError(gray0,
                                                            gray,
                                                            brighten_pose0,
                                                            mp,
                                                            1.,
                                                            *(this->estimation_),
                                                            mp->GetXw(),
                                                            error);
    bool is_inlier = this->mappoints_index_.count(mp);

    if(has_computation) {
      double l1_error = error.lpNorm<1>();
      cv::Scalar color = is_inlier?CV_RGB(0,255,0):CV_RGB(255,0,0);
      cv::putText(err_b, std::to_string( (int)l1_error ), cv::Point(5,10),  cv::FONT_HERSHEY_SIMPLEX, 0.3, color);

      cv::circle(dst, pt, 3, color);
      cv::line(dst, kpt0, pt, color);
    }

    patch0.copyTo(dst(rect0));
    patch.copyTo( dst(rect1));
    err_b.copyTo( dst(rect2));
  }

  std::cout << "all i,b = " << SUM_i << ", " << SUM_b << std::endl;

  return dst;
}


