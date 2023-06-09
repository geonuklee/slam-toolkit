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

#include "pipeline_map.h"
#include "frame.h"
#include "mappoint.h"
// #include "memento.h"

#include "thirdparty/DBoW2/DBoW2/BowVector.h"
#include "thirdparty/DBoW2/DBoW2/FeatureVector.h"

PipelineMap::PipelineMap(ORB_SLAM2::ORBVocabulary*const voc) 
: voc_(voc){
  inverted_file_.resize(voc->size());
}

PipelineMap::~PipelineMap(){
  delete voc_;
}

const std::map<int, Frame*> PipelineMap::GetFrames(bool already_locked) const {
  if(already_locked)
    return frames_;

  std::lock_guard<std::mutex> lock(mutex_);
  return frames_;
}

size_t  PipelineMap::GetFramesNumber(bool already_locked) const {
  if(already_locked)
    return frames_.size();

  std::lock_guard<std::mutex> lock(mutex_);
  return frames_.size();
}

bool PipelineMap::HasFrame(Frame* frame, bool already_locked) const {
  if(!already_locked)
    mutex_.lock();
  return set_frames_.count(frame);
}

Frame* PipelineMap::GetFrame(int frame_idx, bool already_locked) const {
  if(!already_locked)
    mutex_.lock();

  Frame* f;
  if(frames_.count(frame_idx))
    f = frames_.at(frame_idx);
  else
    f = nullptr;

  if(!already_locked)
    mutex_.unlock();
  return f;
}

Frame* PipelineMap::GetLatestFrame() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if(frames_.empty())
    return nullptr;
  Frame* frame = frames_.rbegin()->second;
  return frame;
}

std::set<Frame*> PipelineMap::GetLatestFrames(int number, bool already_locked) const {
  std::set<Frame*> frames;
  if(!already_locked)
    mutex_.lock();
  for(auto it = frames_.rbegin(), it_end = frames_.rend(); it != it_end; it++){
    frames.insert(it->second);
    if(frames.size() == (size_t)number)
      break;
  }
  if(!already_locked)
    mutex_.unlock();
  return frames;
}

void PipelineMap::CullingOldFrames(int reserv_frame_range) {
  std::lock_guard<std::mutex> lock(mutex_);
  //Frame* latest_keyframe = frames_.rbegin()->second;
  int i = 0;
  for(auto r_iter = frames_.rbegin(), r_end = frames_.rend();
      r_iter != r_end; r_iter++){
    Frame* frame = r_iter->second;
    auto frame_mappoints = frame->GetVecMappoints(); // CullingOldFrames
    // Check
    if(frame->IsKeyframe()){
      for(Mappoint* mp : frame_mappoints){
        if(!mp)
          continue;
        if(! mp->GetKeyframes().count(frame)){
          throw std::logic_error("Culling : invalid connection");
        }
      }
    }
    if(i++ < reserv_frame_range)
      continue;
    else if(frame->IsKeyframe()){
      frame->ReduceMemSize();
      continue;
    }
    frames_.erase(frame->GetIndex());
    set_frames_.erase(frame);
    delete frame;
  }
  return;
}

void PipelineMap::AddFrame(Frame* frame, bool already_locked) {
  if(!already_locked)
    mutex_.lock();

  frames_[frame->GetIndex()] = frame;
  set_frames_.insert(frame);

  if(frame->IsKeyframe()){
    // Add Inverted files
    auto bowvec = frame->GetBowVec();
    for(DBoW2::BowVector::const_iterator vit= bowvec.begin(),
        vend=bowvec.end(); vit!=vend; vit++)
      inverted_file_[vit->first].push_back(frame);
  }

  if(!already_locked)
    mutex_.unlock();
  return;
}

std::vector<Frame*> PipelineMap::DetectLoopCandidates(Frame* pKF, float minScore, bool already_locked) const {

  if(!already_locked)
    mutex_.lock();
  std::set<Frame*> spConnectedKeyFrames;
  pKF->GetNeighbors(spConnectedKeyFrames);
  std::list<Frame*> lKFsSharingWords;
  if(!already_locked)
    mutex_.unlock();

  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe
  {
    auto bowvec = pKF->GetBowVec();
    for(DBoW2::BowVector::const_iterator vit=bowvec.begin(),
        vend=bowvec.end(); vit != vend; vit++) {
      const list<Frame*> &lKFs = inverted_file_.at(vit->first);

      for(auto lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++) {
        Frame* pKFi=*lit;
        if(pKF == pKFi)
          continue;

        if(pKFi->GetLoopQuery() != pKF ) {
          pKFi->SetLoopWords(0);
          if(!spConnectedKeyFrames.count(pKFi))
          {
            pKFi->SetLoopQuery(pKF);
            lKFsSharingWords.push_back(pKFi);
          }
        }
        pKFi->SetLoopWords(pKFi->GetLoopWords()+1);
      }
    }
  }

  if(lKFsSharingWords.empty())
    return std::vector<Frame*>();

  std::list<std::pair<float,Frame*> > lScoreAndMatch;

  // Only compare against those keyframes that share enough words
  int maxCommonWords=0;
  for(std::list<Frame*>::iterator lit=lKFsSharingWords.begin(),
      lend= lKFsSharingWords.end(); lit!=lend; lit++)
  {
    if((*lit)->GetLoopWords() > maxCommonWords)
      maxCommonWords=(*lit)->GetLoopWords();
  }

  int minCommonWords = maxCommonWords*0.8f;

  int nscores=0;

  // Compute similarity score. Retain the matches whose score is higher than minScore
  for(std::list<Frame*>::iterator lit=lKFsSharingWords.begin(),
      lend= lKFsSharingWords.end(); lit!=lend; lit++)
  {
    Frame* pKFi = *lit;

    if(pKFi->GetLoopWords() > minCommonWords)
    {
      nscores++;
      float si = voc_->score(pKF->GetBowVec(), pKFi->GetBowVec() );
      pKFi->SetLoopScore(si);
      if(si>=minScore)
        lScoreAndMatch.push_back(make_pair(si,pKFi));
    }
  }

  if(lScoreAndMatch.empty())
    return vector<Frame*>();

  std::list<std::pair<float,Frame*> > lAccScoreAndMatch;
  float bestAccScore = minScore;

  // Lets now accumulate score by covisibility
  for(std::list<std::pair<float,Frame*> >::iterator it=lScoreAndMatch.begin(),
      itend=lScoreAndMatch.end(); it!=itend; it++) {
    Frame* pKFi = it->second;
    vector<Frame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = it->first;
    Frame* pBestKF = pKFi;
    for(std::vector<Frame*>::iterator vit=vpNeighs.begin(),
        vend=vpNeighs.end(); vit!=vend; vit++) {
      Frame* pKF2 = *vit;
      if(pKF2->GetLoopQuery() == pKF && pKF2->GetLoopWords() > minCommonWords) {
        accScore+=pKF2->GetLoopScore();
        if(pKF2->GetLoopScore()>bestScore) {
          pBestKF=pKF2;
          bestScore = pKF2->GetLoopScore();
        }
      }
    }

    lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
    if(accScore>bestAccScore)
      bestAccScore=accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f*bestAccScore;

  std::set<Frame*> spAlreadyAddedKF;
  std::vector<Frame*> vpLoopCandidates;
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  for(std::list<std::pair<float,Frame*> >::iterator it=lAccScoreAndMatch.begin(),
      itend=lAccScoreAndMatch.end(); it!=itend; it++) {
    if(it->first>minScoreToRetain) {
      Frame* pKFi = it->second;
      if(!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpLoopCandidates;
}

ORB_SLAM2::ORBVocabulary* PipelineMap::GetVocaBulary() const{
  return voc_;
}

