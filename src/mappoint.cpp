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

#include "mappoint.h"
#include "frame.h"
//#include "memento.h"

#ifdef MEMENTO_H_
Mappoint::Mappoint(Frame* ref, MementoMappoint* memento)
: ref_(ref),
id_(memento->id_),
bad_flag_(memento->bad_flag_)
{
  Mappoint::n_ = std::max<size_t>(Mappoint::n_, id_+1);
  SetInvD(memento->invd_);
}
#endif

size_t Mappoint::n_ = 0;

Mappoint::Mappoint(Frame* ref, double invd, int id)
  : ref_(ref), id_(id), bad_flag_(false) {
  SetInvD(invd);
  AddKeyframe(ref);
}



Frame* Mappoint::GetRefFrame() const{
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");
  std::lock_guard<std::mutex> lock(mutex_);
  return ref_;
}

void Mappoint::SetInvD(double invd){
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");
  std::lock_guard<std::mutex> lock(mutex_);
  if(invd < 0.001){
    // Ignore too smal inverse depth depth which cause NaN error.
    invd = 0.001;
  }
  invd_ = invd;
  return;
}


double Mappoint::GetDepth() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return 1./invd_;
}

cv::Mat Mappoint::GetDescription() const {
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");

  std::lock_guard<std::mutex> lock(mutex_);
  int idx = ref_->GetIndex(this);
  return ref_->GetDescription(idx);
}

cv::KeyPoint Mappoint::GetKeypoint() const {
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");
  std::lock_guard<std::mutex> lock(mutex_);
  int idx = ref_->GetIndex(this);
  return ref_->GetKeypoints().at(idx);
}

void Mappoint::AddKeyframe(Frame* keyframe) {
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");

  std::lock_guard<std::mutex> lock(mutex_);
  if(!keyframe->IsKeyframe()){
    throw std::invalid_argument("Frame::AddKeyframe(), given frame is not a keyframe");
  }
  keyframes_.insert(keyframe);
}

const std::set<Frame*>& Mappoint::GetKeyframes() const {
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");

  std::lock_guard<std::mutex> lock(mutex_);
  std::set<Frame*> keyframes;
  for(Frame* f : keyframes_){
    keyframes.insert(f);
  }
  return keyframes_;
}

int Mappoint::GetIndex() const{
  std::lock_guard<std::mutex> lock(mutex_);
  return id_;
}
bool Mappoint::IsBad() const{
  std::lock_guard<std::mutex> lock(mutex_);
  return bad_flag_;
}

void Mappoint::SetBad() {
  std::lock_guard<std::mutex> lock(mutex_);
  bad_flag_ = true;
}

Eigen::Vector3d Mappoint::GetXw() const {
  if(IsBad())
    throw std::logic_error("Try to access bad mappoitns");

  std::lock_guard<std::mutex> lock(mutex_);
  g2o::SE3Quat Tcw = ref_->GetTcw();
  size_t kpt_idx = ref_->GetIndex(this);
  Eigen::Vector2d nuv = ref_->GetNormalizedPoint(kpt_idx);
  Eigen::Vector3d Xc(nuv[0]/invd_, nuv[1]/invd_,  1./invd_);
  return Tcw.inverse().map(Xc);
}
