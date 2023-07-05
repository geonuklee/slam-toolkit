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

#include "loopdetector.h"
#include "frame.h"
#include "ORBVocabulary.h"

size_t CovisibilityConsistencyTh = 5;

LoopDetector::LoopDetector(const PipelineMap* map)
  : map_(map),
  mnCovisibilityConsistencyTh(CovisibilityConsistencyTh),
  last_loop_kf_(nullptr)
{

}

bool LoopDetector::DetectLoop(Frame* curr_frame) {
  ORB_SLAM2::ORBVocabulary* voc = map_->GetVocaBulary();
  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  int min_covisibility = 20;
  std::vector<Frame*> vpConnectedKeyFrames;
  map_->Lock();
  const bool already_locked = true;
  curr_frame->GetNeighbors(vpConnectedKeyFrames, min_covisibility);
  map_->UnLock();

  const DBoW2::BowVector &CurrentBowVec = curr_frame->GetBowVec();
#if 0
  float minScore = 1;
  for(size_t i=0; i<vpConnectedKeyFrames.size(); i++) {
    Frame* pKF = vpConnectedKeyFrames[i];
    const DBoW2::BowVector &BowVec = pKF->GetBowVec();
    float score = voc->score(CurrentBowVec, BowVec);
    if(score<minScore)
      minScore = score;
  }
  // TODO : "Switchable LooCloser"
#else
  // min_covisibility threshold 추가후에도, 위 방식은 회전구간때문에
  // minScore가 너무 낮게 나와 loop detection이 잘못(위양성) 되는 경우가 많아, 아래처럼 rule을 바꿈
  float bestScore = 0.;
  for(size_t i=0; i<vpConnectedKeyFrames.size(); i++) {
    Frame* pKF = vpConnectedKeyFrames[i];
    //if(pKF->isBad())
    //  continue;
    const DBoW2::BowVector &BowVec = pKF->GetBowVec();
    float score = voc->score(CurrentBowVec, BowVec);

    if(score>bestScore)
      bestScore = score;
  }
  float minScore = 0.7 * bestScore;
#endif

  // Query the database imposing the minimum score
  std::vector<Frame*> vpCandidateKFs = map_->DetectLoopCandidates(curr_frame, minScore, already_locked);

  // If there are no loop candidates, just add new keyframe and return false
  if(vpCandidateKFs.empty()) {
    mvConsistentGroups.clear();
    return false;
  }
  //std::cout << "vpCandidateKFs from inverted file exists" << std::endl;

  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
  // A group is consistent with a previous group if they share at least a keyframe
  // We must detect a consistent loop in several consecutive keyframes to accept it
  mvpEnoughConsistentCandidates.clear();

  std::vector<ConsistentGroup> vCurrentConsistentGroups;
  std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);

  for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++) {
    Frame* pCandidateKF = vpCandidateKFs[i];

    std::set<Frame*> spCandidateGroup;
    pCandidateKF->GetNeighbors(spCandidateGroup, min_covisibility);
    spCandidateGroup.insert(pCandidateKF);

    bool bEnoughConsistent = false;
    bool bConsistentForSomeGroup = false;
    for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
    {
      std::set<Frame*> sPreviousGroup = mvConsistentGroups[iG].first;

      bool bConsistent = false;
      for(std::set<Frame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++) {
        if(sPreviousGroup.count(*sit))
        {
          bConsistent=true;
          bConsistentForSomeGroup=true;
          break;
        }
      }

      if(bConsistent)
      {
        int nPreviousConsistency = mvConsistentGroups[iG].second;
        int nCurrentConsistency = nPreviousConsistency + 1;
        if(!vbConsistentGroup[iG])
        {
          ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
          vCurrentConsistentGroups.push_back(cg);
          vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
        }
        if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent) {
          mvpEnoughConsistentCandidates.push_back(pCandidateKF);
          bEnoughConsistent=true; //this avoid to insert the same candidate more than once
        }
      }
    }

    // If the group is not consistent with any previous group insert with consistency counter std::set to zero
    if(!bConsistentForSomeGroup)
    {
      ConsistentGroup cg = make_pair(spCandidateGroup,0);
      vCurrentConsistentGroups.push_back(cg);
    }
  }

  // Update Covisibility Consistent Groups
  mvConsistentGroups = vCurrentConsistentGroups;

  // Add Current Keyframe to database
  if(mvpEnoughConsistentCandidates.empty())
    return false;


  return true;
}

const std::vector<Frame*>& LoopDetector::GetLoopCandidates() const {
  return mvpEnoughConsistentCandidates;
}

