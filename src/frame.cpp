/*
Copyright (c) 2020 Geonuk Lee

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#include "orb_extractor.h"

#include "frame.h"
#include "camera.h"
#include "mappoint.h"

size_t Frame::frame_n_ = 0;
size_t Frame::keyframe_n_ = 0;

Frame::Frame(cv::Mat gray,
             int id,
             const Camera* camera,
             ORB_SLAM2::ORBextractor* extractor)
  : gray_(gray),
  id_(id),
  kf_id_(-1),
  estimation_(nullptr),
  loop_query_(nullptr),
  loop_words_(0),
  loop_score_(0.),
  camera_(camera),
  extractor_(extractor)
{
  extractor->extract(gray, cv::noArray(), keypoints_, descriptions_);
  mappoints_.resize(keypoints_.size(), nullptr);

  const Eigen::Matrix<double,3,3> invK =  camera_->GetInvK();

  for(auto kpt: keypoints_) {
    Eigen::Vector2d uv(kpt.pt.x, kpt.pt.y);
    Eigen::Vector3d nuv = camera_->NormalizedUndistort(uv);
    normalized_keypoints_.push_back(nuv.head<2>());
  }

  // ref : https://github.com/mariusmuja/flann/blob/master/examples/flann_example.cpp
  flann_keypoints_
    = flann::Matrix<double>(new double[2*keypoints_.size()], keypoints_.size(), 2);
  flann_kdtree_
    = new flann::Index< flann::L2<double> >(flann_keypoints_, flann::KDTreeSingleIndexParams());
  for(size_t i = 0; i < keypoints_.size(); i++){
    const cv::KeyPoint& kpt = keypoints_.at(i);
    flann_keypoints_[i][0] = kpt.pt.x;
    flann_keypoints_[i][1] = kpt.pt.y;
  }
  flann_kdtree_->buildIndex();
}

StereoFrame::StereoFrame(cv::Mat gray, cv::Mat gray_right, int id, const StereoCamera* camera, ORB_SLAM2::ORBextractor* extractor)
  : Frame(gray, id, camera, extractor),
  gray_right_(gray_right)
{
  stereo_correspond_.resize(keypoints_.size(), -1);
}
/*
Frame::Frame(MementoFrame* memento,
             const Camera* camera,
             ORBextractor* extractor,
             ORBVocabulary* voc)
  : im_left_(memento->im_left_),
  im_right_(memento->im_right_),
  id_(memento->id_),
  kf_id_(memento->kf_id_),
  keypoints_(memento->keypoints_),
  descriptions_(memento->descriptions_),
  stereo_correspond_(memento->stereo_correspond_),
  r_keypoints_(memento->r_keypoints_),
  r_descriptions_(memento->r_descriptions_),
  Tcw_(new g2o::SE3Quat(memento->Tcw_)),
  loop_query_(nullptr),
  loop_words_(memento->loop_words_),
  loop_score_(memento->loop_score_),
  camera_(camera),
  extractor_(extractor),
  width_(memento->im_left_.cols),
  height_(memento->im_left_.rows)
{
  Frame::frame_n_ = std::max<size_t>(Frame::frame_n_, id_+1);
  Frame::keyframe_n_ = std::max<size_t>(Frame::keyframe_n_, kf_id_+1);
  mappoints_.resize(keypoints_.size(), nullptr);

  Eigen::Matrix<double,3,3> invK =  camera_->GetK().inverse();
  for(auto kpt: keypoints_) {
    Eigen::Vector3d uv(kpt.pt.x, kpt.pt.y, 1.);
    Eigen::Vector3d nuv = invK * uv;
    normalized_keypoints_.push_back(nuv.head<2>());
  }

  flann_keypoints_
    = flann::Matrix<double>(new double[2*keypoints_.size()], keypoints_.size(), 2);
  flann_kdtree_
    = new flann::Index< flann::L2<double> >(flann_keypoints_, flann::KDTreeSingleIndexParams());
  for(size_t i = 0; i < keypoints_.size(); i++){
    const cv::KeyPoint& kpt = keypoints_.at(i);
    flann_keypoints_[i][0] = kpt.pt.x;
    flann_keypoints_[i][1] = kpt.pt.y;
  }
  flann_kdtree_->buildIndex();

  ComputeBoW(voc);
}
*/

Frame::~Frame(){
  if(estimation_)
    delete estimation_;
  if(flann_keypoints_.ptr())
    delete flann_keypoints_.ptr();
  if(flann_kdtree_)
    delete flann_kdtree_;
}

StereoFrame::~StereoFrame(){

}

cv::Mat StereoFrame::PlotStereoMatch() const {
  std::lock_guard<std::mutex> lock(mutex_);
  cv::Mat dst;
  cv::cvtColor(gray_, dst, cv::COLOR_GRAY2RGB);

  for(size_t i = 0; i < keypoints_.size(); i++) {
    cv::KeyPoint kpt = keypoints_.at(i);
    cv::circle(dst, kpt.pt, 2, CV_RGB(255,0,0));

    int j = stereo_correspond_.at(i);
    if(j < 0)
      continue;
    cv::KeyPoint rkpt = r_keypoints_.at(j);
    cv::line(dst, kpt.pt, rkpt.pt, CV_RGB(0,255,0));
  }
  return dst;
}

std::set<int> Frame::SearchRadius(const Eigen::Vector2d& uv, double radius) const {
  std::lock_guard<std::mutex> lock(mutex_);
  flann::Matrix<double> quary((double*)uv.data(), 1, 2);
  std::set<int> inliers;
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;
  flann_kdtree_->radiusSearch(quary, indices, dists, radius*radius, param);
  for(int idx :  indices[0])
    inliers.insert(idx);
  return inliers;
}

std::vector<std::vector<int> > Frame::SearchRadius(const flann::Matrix<double>& points,
                                                   double radius) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;
  flann_kdtree_->radiusSearch(points, indices, dists, radius*radius, param);
  return indices;
}

void Frame::SearchNeareast(const Eigen::Vector2d& uv, int& kpt_index, double& distance) const {
  std::lock_guard<std::mutex> lock(mutex_);
  flann::Matrix<double> quary((double*)uv.data(), 1, 2);
  std::vector<std::vector<int> > indices;
  std::vector<std::vector<double> > dists;
  const flann::SearchParams param;

  kpt_index = -1;
  if(! flann_kdtree_->knnSearch(quary, indices, dists, 1, param) )
    return;
  kpt_index = indices.at(0).at(0);
  distance = dists.at(0).at(0);
  return;
}

const std::vector<cv::KeyPoint>& Frame::GetKeypoints() const {
  return keypoints_;
}

const cv::KeyPoint& Frame::GetKeypoint(const Mappoint* mp) const {
  int idx = GetIndex(mp);
  std::lock_guard<std::mutex> lock(mutex_);
  return keypoints_.at(idx);
}

const cv::Mat Frame::GetDescription(int i) const {
  std::lock_guard<std::mutex> lock(mutex_);
  cv::Mat desc = descriptions_.row(i);
  return desc;
}

const Eigen::Vector2d& Frame::GetNormalizedPoint(size_t kpt_idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return normalized_keypoints_.at(kpt_idx);
}

const cv::KeyPoint StereoFrame::GetRightKeypoint(const Mappoint* mp) const {
  int i = GetIndex(mp);
  std::lock_guard<std::mutex> lock(mutex_);
  int ridx = stereo_correspond_.at(i);
  const cv::KeyPoint& rkpt = r_keypoints_.at(ridx);
  return rkpt;
}

const std::vector<cv::KeyPoint>& StereoFrame::GetRightKeypoints() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return r_keypoints_;
}

const cv::Mat StereoFrame::GetRightDescription(int i) const {
  std::lock_guard<std::mutex> lock(mutex_);
  cv::Mat desc = r_descriptions_.row(i);
  return desc;
}

std::set<Mappoint*> SupplyMappoints(Frame* frame) {
  std::set<Mappoint*> supplied_mappoints;

  const auto& keypoints = frame->GetKeypoints();
  auto vec_mappoints = frame->GetVecMappoints(); //SupplyMappoints

  for(size_t i = 0; i < keypoints.size(); i++) {
    if(vec_mappoints.at(i))
      continue;
    cv::KeyPoint kpt = keypoints.at(i);
    Eigen::Vector3d Xc;
    if(! frame->GetDepth(i, Xc) )
      continue;
    double invd = 1./Xc.z();
    Mappoint* mp = new Mappoint(frame, invd, Mappoint::n_++);
    frame->SetMappoint(mp, i);
    supplied_mappoints.insert(mp);
  }

  return supplied_mappoints;
}

const std::vector<int> StereoFrame::GetStereoCorrespond() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stereo_correspond_;
}

void StereoFrame::SetStereoCorrespond(const std::vector<int> & correspond ) {
  std::lock_guard<std::mutex> lock(mutex_);
  stereo_correspond_ = correspond;
}

cv::Mat Frame::GetImage() const{
  //std::lock_guard<std::mutex> lock(mutex_); << Directmethod를 굉장히 느리게 함.
  //return gray_.clone();
  return gray_;
}

const Camera* Frame::GetCamera() const{
  return camera_;
}

const ORB_SLAM2::ORBextractor* Frame::GetExtractor() const {
  return extractor_;
}

void Frame::SetMappoint(Mappoint* mp, int index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!mp){
    throw std::invalid_argument("mp can't be nullptr!");
  }
  if(mappoints_index_.count(mp)){
    throw std::invalid_argument("Already matched mappoint!");
  }
  if(mappoints_.at(index)){
    throw std::invalid_argument("Erase previous mp before overwrite new mp.");
  }
  mappoints_[index] = mp;
  mappoints_index_[mp] = index;

  if(kf_id_ >= 0)
    mp->AddKeyframe(this);
  return;
}

void Frame::SetMappoitIfEmpty(Mappoint* mp, int index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(mappoints_.at(index))
    return;
  mappoints_[index] = mp;
  mappoints_index_[mp] = index;

  if(kf_id_ >= 0)
    mp->AddKeyframe(this);
  return;
}

int Frame::GetIndex(const Mappoint* mp) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if(mappoints_index_.count(mp))
    return mappoints_index_.at(mp);
  return -1;
}

int Frame::GetIndex() const {
  return id_;
}

int Frame::GetKeyframeIndex() const{
  std::lock_guard<std::mutex> lock(mutex_);
  return kf_id_;
}

const std::vector<Mappoint*>& Frame::GetVecMappoints() const {
  return mappoints_;
}

Mappoint* Frame::GetMappoint(int kpt_idx) const {
  std::lock_guard<std::mutex> lock(mutex_); // TODO 이게 굉장히 느린 연산임.
  return mappoints_.at(kpt_idx);
}

int Frame::EraseMappoint(Mappoint* mp) {
  int index = GetIndex(mp);
  std::lock_guard<std::mutex> lock(mutex_);
  mappoints_[index] = nullptr;
  mappoints_index_.erase(mp);
  return index;
}

const std::set<Mappoint*> Frame::GetMappoints() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::set<Mappoint*> mappoints(mappoints_.begin(), mappoints_.end());
  if(mappoints.count(nullptr)){
    mappoints.erase(nullptr);
  }
  return mappoints;
}

void Frame::SetTcw(const g2o::SE3Quat& Tcw) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!estimation_)
    estimation_ = new BrightenSE3();
  estimation_->Tcw_ = Tcw;
  return;
}

g2o::SE3Quat Frame::GetTcw() const{
  std::lock_guard<std::mutex> lock(mutex_);
  return estimation_->Tcw_;
}

void Frame::SetBrightness(const Eigen::Vector2d& brightness) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!estimation_)
    estimation_ = new BrightenSE3();
  estimation_->brightness_ = brightness;
}

Eigen::Vector2d Frame::GetBrightness() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return estimation_->brightness_;
}

const BrightenSE3 Frame::GetBrightenPose() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return *estimation_;
}

void StereoFrame::ExtractRightKeypoints() {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!r_keypoints_.empty())
    throw std::logic_error("Unexpected call of Frame::ExtractRightKeypoints.");
  extractor_->extract(gray_right_, cv::noArray(), r_keypoints_, r_descriptions_);
}

bool StereoFrame::GetDepth(int kpt_index, Eigen::Vector3d& Xc) const {
  int j = stereo_correspond_.at(kpt_index);
  if(j < 0)
    return false;
  cv::KeyPoint kpt = keypoints_.at(kpt_index);
  cv::KeyPoint rkpt = r_keypoints_.at(j);
  const double& fx = camera_->GetK()(0,0);
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(camera_);
  const double baseline = - camera->GetTlr().translation().x();
  double dx = kpt.pt.x - rkpt.pt.x;
  if(dx < 0.){
    throw std::invalid_argument("StereoMatch should filter negative depth");
    return false;
  }
  double depth = fx * baseline / dx ;
  Xc.head<2>() = normalized_keypoints_.at(kpt_index) * depth;
  Xc(2,0) = depth;
  return true;
}

std::vector<cv::Mat> toDescriptorVector(cv::Mat descriptions){
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(descriptions.rows);
  for (int j=0;j<descriptions.rows;j++)
    vDesc.push_back(descriptions.row(j));
  return vDesc;
}

void Frame::ComputeBoW(ORB_SLAM2::ORBVocabulary* voc) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(bowvec_.empty() || featvec_.empty()) {
    std::vector<cv::Mat> vdesc = toDescriptorVector(descriptions_);
    // Feature vector associate features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    voc->transform(vdesc,bowvec_,featvec_,4);
  }
}

const DBoW2::BowVector& Frame::GetBowVec() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return bowvec_;
}

const DBoW2::FeatureVector& Frame::GetFeatureVec() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return featvec_;
}

bool Frame::IsKeyframe() const {
  //std::lock_guard<std::mutex> lock(mutex_);
  return kf_id_ >= 0;
}

void Frame::SetKeyframe(ORB_SLAM2::ORBVocabulary* voc, int kf_id) {
  if(IsKeyframe()){
    throw std::invalid_argument("Already keyframe");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    kf_id_ = kf_id;
  }

  for(Mappoint* mp : mappoints_){
    if(!mp)
      continue;
    mp->AddKeyframe(this);
  }

  if(voc)
    ComputeBoW(voc);
  return;
}

bool Frame::IsInFrame(const Eigen::Vector2d& uv) const {
  return camera_->IsInImage(uv);
}

void Frame::GetNeighbors(std::set<Frame*>& neighbor_keyframes,
                         std::set<Mappoint*>& covisible_mappoints,
                         int min_covisibility
                         ) const {
  const std::set<Mappoint*> mappoints = this->GetMappoints();
  std::map<Frame*, std::set<Mappoint*> > covisibilities;
  for(Mappoint* mp : mappoints){
    std::set<Frame*> keyframes = mp->GetKeyframes();
    if(mp->IsBad())
      throw std::logic_error("Try to call bad mp 1");
    for(Frame* nkf : keyframes){
      covisibilities[nkf].insert(mp);
    }
  }

  if(min_covisibility > 0){
    for(auto it : covisibilities){
      if(it.second.size() < (size_t) min_covisibility)
        continue;
      if(it.first != this)
        neighbor_keyframes.insert(it.first);
      for(Mappoint* mp : it.second)
        covisible_mappoints.insert(mp);
    }
  }
  else{
    for(auto it : covisibilities){
      if(it.first != this)
        neighbor_keyframes.insert(it.first);
      for(Mappoint* mp : it.second)
        covisible_mappoints.insert(mp);
    }
  }
  return;
}

void Frame::GetNeighbors(std::set<Frame*>& neighbor_keyframes,
                         int min_covisibility
                         ) const {
  std::set<Mappoint*> neighbor_mappoints;
  GetNeighbors(neighbor_keyframes, neighbor_mappoints, min_covisibility);
  return;
}

void Frame::GetNeighbors(std::vector<Frame*>& neighbor_keyframes,
                         int min_covisibility
                         ) const {
  std::set<Frame*> tmp;
  GetNeighbors(tmp, min_covisibility);
  neighbor_keyframes.clear();
  neighbor_keyframes.reserve(tmp.size());
  for(auto nkf : tmp)
    neighbor_keyframes.push_back(nkf);
  return;
}

std::vector<Frame*> Frame::GetBestCovisibilityKeyFrames(int num_of_keyframes) const {
  std::map<Frame*, size_t> covisibilities;
  const std::set<Mappoint*>& mappoints = this->GetMappoints();
  for(Mappoint* mp : mappoints){
    std::set<Frame*> keyframes = mp->GetKeyframes();
    for(Frame* nkf : keyframes){
      if(nkf == this)
        continue;
      covisibilities[nkf]++;
    }
  }

  if(covisibilities.count((Frame*)this))
    covisibilities.erase((Frame*)this);

  std::vector< std::pair<size_t, Frame* > > sorting_vec;

  for(auto it : covisibilities){
    sorting_vec.push_back( std::make_pair(it.second, it.first));
  }

  std::sort(sorting_vec.begin(), sorting_vec.end(),
            [](const std::pair<size_t,Frame*> a, const std::pair<size_t,Frame*> b){
              return a.first > b.first;
            } );

  std::vector<Frame*> result;
  for(size_t i = 0; i < sorting_vec.size(); i++){
    if(num_of_keyframes > 0)
      if(i >= (size_t)num_of_keyframes)
        break;
    result.push_back(sorting_vec.at(i).second);
  }
  return result;
}

void Frame::SetLoopQuery(Frame* loop_query) {
  std::lock_guard<std::mutex> lock(mutex_);
  loop_query_ = loop_query;
}

Frame* Frame::GetLoopQuery() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loop_query_;
}

void Frame::SetLoopWords(size_t loop_words) {
  std::lock_guard<std::mutex> lock(mutex_);
  loop_words_ = loop_words;
}

size_t Frame::GetLoopWords() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loop_words_;
}

void Frame::SetLoopScore(float score) {
  std::lock_guard<std::mutex> lock(mutex_);
  loop_score_ = score;
}

float Frame::GetLoopScore() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loop_score_;
}

void Frame::ReduceMemSize(){
  std::lock_guard<std::mutex> lock(mutex_);
  gray_ = cv::Mat();
}

void StereoFrame::ReduceMemSize(){
  Frame::ReduceMemSize();
  std::lock_guard<std::mutex> lock(mutex_);
  gray_right_ = cv::Mat();
}

