#include "segslam.h"
#include "orb_extractor.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

class ExtractorNode {
public:
    ExtractorNode():bNoMore(false){}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    int wh;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4) {
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);
    n1.wh = n2.wh = n3.wh = n4.wh = std::min(halfX, halfY);
    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());
    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());
    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++) {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x) {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
    return;
}

/*
  TODO vToDistributeKeys에 class_id로 keypoint id를 저장해놓고선, 이 함수의 결과물로
  Mappoint를 화면 골고루에 뿌릴 준비를 한다.
  min_distance : 특징점 사이의 최소거리.
  ORB_SLAM2::ORBextractor::DistributeOctTree의 알고리즘 그대로 가져가되, member variable dependency만 삭제.
*/

std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                            const int &minX,
                                            const int &maxX,
                                            const int &minY,
                                            const int &maxY,
                                            const int &nFeaturesPerLevel,
                                            const int &min_distance = 10,
                                            const cv::Mat& mask = cv::Mat())
{
  const int double_of_min_distance= 2*min_distance;
  // Compute how many initial nodes   
  const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
  const float hX = static_cast<float>(maxX-minX)/nIni;
  std::list<ExtractorNode> lNodes;
  std::vector<ExtractorNode*> vpIniNodes;
  vpIniNodes.resize(nIni);
  for(int i=0; i<nIni; i++) {
    ExtractorNode ni;
    ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
    ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
    ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
    ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
    ni.wh = std::min(ni.UR.x-ni.UL.x, ni.BR.y - ni.UL.y );
    assert(ni.wh > 0);
    ni.vKeys.reserve(vToDistributeKeys.size());
    lNodes.push_back(ni);
    vpIniNodes[i] = &lNodes.back();
  }
  //Associate points to childs
  for(size_t i=0;i<vToDistributeKeys.size();i++) {
    const cv::KeyPoint &kp = vToDistributeKeys[i];
    vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
  }
  std::list<ExtractorNode>::iterator lit = lNodes.begin();
  while(lit!=lNodes.end()) {
    if(lit->vKeys.size()==1) {
      lit->bNoMore=true;
      lit++;
    }
    else if(lit->vKeys.empty())
      lit = lNodes.erase(lit);
    else
      lit++;
  }
  bool bFinish = false;
  int iteration = 0;
  std::vector<std::pair<int,ExtractorNode*> > vSizeAndPointerToNode;
  vSizeAndPointerToNode.reserve(lNodes.size()*4);
  while(!bFinish) {
    iteration++;
    int prevSize = lNodes.size();
    lit = lNodes.begin();
    int nToExpand = 0;
    vSizeAndPointerToNode.clear();
    while(lit!=lNodes.end()) {
      //if(lit->bNoMore) {
      if(lit->bNoMore || lit->wh < double_of_min_distance) {
        /* 1) If node only contains one point
           2) , or if width of sub node will be smaller than given min distance
           do not subdivide and continue */ 
        //std::cout << "lit->wh = " << lit->wh << std::endl;
        lit++;
        continue;
      }
      else {
        // If more than one point, subdivide
        ExtractorNode n1,n2,n3,n4;
        lit->DivideNode(n1,n2,n3,n4);
        // Add childs if they contain points
        if(n1.vKeys.size()>0) {
          lNodes.push_front(n1);                    
          if(n1.vKeys.size()>1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if(n2.vKeys.size()>0) {
          lNodes.push_front(n2);
          if(n2.vKeys.size()>1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if(n3.vKeys.size()>0) {
          lNodes.push_front(n3);
          if(n3.vKeys.size()>1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if(n4.vKeys.size()>0) {
          lNodes.push_front(n4);
          if(n4.vKeys.size()>1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        lit=lNodes.erase(lit);
        continue;
      }
    }       

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    if((int)lNodes.size()>=nFeaturesPerLevel || (int)lNodes.size()==prevSize) {
      bFinish = true;
    }
    else if(((int)lNodes.size()+nToExpand*3)>nFeaturesPerLevel) {
      while(!bFinish) {
        prevSize = lNodes.size();
        std::vector<std::pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
        vSizeAndPointerToNode.clear();

        std::sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
        for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--) {
          ExtractorNode n1,n2,n3,n4;
          vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

          // Add childs if they contain points
          if(n1.vKeys.size()>0) {
            lNodes.push_front(n1);
            if(n1.vKeys.size()>1) {
              vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if(n2.vKeys.size()>0) {
            lNodes.push_front(n2);
            if(n2.vKeys.size()>1) {
              vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if(n3.vKeys.size()>0) {
            lNodes.push_front(n3);
            if(n3.vKeys.size()>1) {
              vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if(n4.vKeys.size()>0) {
            lNodes.push_front(n4);
            if(n4.vKeys.size()>1) {
              vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);
          if((int)lNodes.size()>=nFeaturesPerLevel)
            break;
        }
        if((int)lNodes.size()>=nFeaturesPerLevel || (int)lNodes.size()==prevSize)
          bFinish = true;
      }
    }
  }

  // Retain the best point in each node
  std::vector<cv::KeyPoint> vResultKeys;
  vResultKeys.reserve(vToDistributeKeys.size());
  for(std::list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++) {
    std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
    cv::KeyPoint* pKP = &vNodeKeys[0]; // Node 내에서 가장 response가 강한 keypoint만 남긴다.
    float maxResponse = pKP->response;
    for(size_t k=1;k<vNodeKeys.size();k++) {
      if(vNodeKeys[k].response>maxResponse) {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }
    vResultKeys.push_back(*pKP);
  }
  return vResultKeys;
}

namespace SEG {
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


OrbSlam2FeatureDescriptor::OrbSlam2FeatureDescriptor(int nfeatures, float scale_factor, int nlevels, int initial_fast_th, int min_fast_th)
  : FeatureDescriptor() {
  auto ptr = new ORB_SLAM2::ORBextractor(nfeatures, scale_factor, nlevels, initial_fast_th, min_fast_th);
  extractor_ = std::shared_ptr<ORB_SLAM2::ORBextractor>(ptr);
}

void OrbSlam2FeatureDescriptor::Extract(const cv::Mat gray, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  extractor_->extract(gray, mask, keypoints, descriptors);
  return;
}

double OrbSlam2FeatureDescriptor::GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const {
  return ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
}

CvFeatureDescriptor::CvFeatureDescriptor() 
  : FeatureDescriptor() {
  nfeatures = 4000;
  scaleFactor = 1.2f;
  nlevels = 3;
  iniThFAST = 20;
  min_kpt_distance = 10.; // For each level
  // ---------------------------------------------
  mvScaleFactor.resize(nlevels);
  mvLevelSigma2.resize(nlevels);
  mvScaleFactor[0]=1.0f;
  mvLevelSigma2[0]=1.0f;
  for(int i=1; i<nlevels; i++) {
    mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
    mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
  }
  mvInvScaleFactor.resize(nlevels);
  mvInvLevelSigma2.resize(nlevels);
  for(int i=0; i<nlevels; i++) {
    mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
    mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
  }
  mvImagePyramid.resize(nlevels);
  mnFeaturesPerLevel.resize(nlevels);
  float factor = 1.0f / scaleFactor;
  float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));
  int sumFeatures = 0;
  for( int level = 0; level < nlevels-1; level++ ) {
    mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += mnFeaturesPerLevel[level];
    nDesiredFeaturesPerScale *= factor;
  }
  mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
  const int npoints = 512;
  const cv::Point* pattern0 = (const cv::Point*)ORB_SLAM2::bit_pattern_31_;
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
  //This is for orientation
  // pre-compute the end of a row in a circular patch
  umax.resize(HALF_PATCH_SIZE + 1);
  int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
  const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
  for (v = 0; v <= vmax; ++v)
    umax[v] = cvRound(sqrt(hp2 - v * v));
  // Make sure we are symmetric
  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
    while (umax[v0] == umax[v0 + 1])
      ++v0;
    umax[v] = v0;
    ++v0;
  }
}

void CvFeatureDescriptor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];
    cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
    cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
    cv::Mat temp(wholeSize, image.type()), masktemp;
    mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
    // Compute the resized image
    if( level != 0 ) {
      cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
      copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);            
    }
    else {
      copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     cv::BORDER_REFLECT_101);            
    }
  }
  return;
}

static float IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max) {
    int m_01 = 0, m_10 = 0;
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));
    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];
    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    return cv::fastAtan2((float)m_01, (float)m_10);
}

static void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                               const std::vector<int>& umax) {
  for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
       keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {
    keypoint->angle = IC_Angle(image, keypoint->pt, umax);
  }
}

void CvFeatureDescriptor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, const cv::Mat& mask) {
  const float W = 30;
  allKeypoints.resize(nlevels);
  for (int level = 0; level < nlevels; ++level) {
    const int minBorderX = EDGE_THRESHOLD-3;
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
    const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;
    std::vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(nfeatures*10);
    const float width = (maxBorderX-minBorderX);
    const float height = (maxBorderY-minBorderY);
    const int nCols = width/W;
    const int nRows = height/W;
    const int wCell = ceil(width/nCols);
    const int hCell = ceil(height/nRows);
    const float& scale = mvScaleFactor[level];
    for(int i=0; i<nRows; i++) {
      const float iniY =minBorderY+i*hCell;
      float maxY = iniY+hCell+6;
      if(iniY>=maxBorderY-3)
        continue;
      if(maxY>maxBorderY)
        maxY = maxBorderY;
      for(int j=0; j<nCols; j++) {
        const float iniX =minBorderX+j*wCell;
        float maxX = iniX+wCell+6;
        if(iniX>=maxBorderX-6)
          continue;
        if(maxX>maxBorderX)
          maxX = maxBorderX;
        std::vector<cv::KeyPoint> vKeysCell;
        cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,iniThFAST,true);
        //if(vKeysCell.empty()) // TODO 이걸 써야해 말아야해?
        //  cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,minThFAST,true);
        if(!vKeysCell.empty()) {
          for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++) {
            cv::Point2f& pt =vit->pt;
            pt.x+=j*wCell;
            pt.y+=i*hCell;
            if(mask.at<uchar>(scale*(pt+cv::Point2f(minBorderX,minBorderY)) ) > 0) //  mask가 1 이상인 영역은 keypoint 제외
              continue;
            vToDistributeKeys.push_back(*vit);
          }
        }
      }
    } // for(i=0; i<nRows; i++)

    std::vector<cv::KeyPoint> & keypoints = allKeypoints[level];
    keypoints = DistributeQuadTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY,
                                   mnFeaturesPerLevel[level], min_kpt_distance);
    const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

    // Add border to coordinates and scale information
    const int nkps = keypoints.size();
    for(int i=0; i<nkps ; i++) {
      keypoints[i].pt.x+=minBorderX;
      keypoints[i].pt.y+=minBorderY;
      keypoints[i].octave=level;
      keypoints[i].size = scaledPatchSize;
    }
  }
  // compute orientations
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
  return;
}


const float factorPI = (float)(CV_PI/180.f);
static void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat& img, const cv::Point* pattern,
                                 uchar* desc) {
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]
    for (int i = 0; i < 32; ++i, pattern += 16) {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;
        desc[i] = (uchar)val;
    }
    #undef GET_VALUE
}

static void computeDescriptors(const cv::Mat& image,
                               std::vector<cv::KeyPoint>& keypoints, 
                               cv::Mat& descriptors,
                               const std::vector<cv::Point>& pattern)
{
    descriptors =cv:: Mat::zeros((int)keypoints.size(), 32, CV_8UC1);
    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}


void CvFeatureDescriptor::Extract(const cv::Mat gray,
                                  cv::InputArray mask,
                                  std::vector<cv::KeyPoint>& _keypoints,
                                  cv::Mat& descriptors) {
  assert(gray.type() == CV_8UC1 );
  cv::Mat _mask = mask.getMat();
  // Pre-compute the scale pyramid
  ComputePyramid(gray);
  std::vector < std::vector< cv::KeyPoint> > allKeypoints;
  ComputeKeyPointsOctTree(allKeypoints, _mask); // level 별로 nfeatures 개수만큼 featur extraction.
#if 1
  int nkeypoints = 0;
  for (int level = 0; level < nlevels; ++level)
      nkeypoints += (int)allKeypoints[level].size();
  if( nkeypoints > 0)
      descriptors = cv::Mat(nkeypoints, 32, CV_8U);
  _keypoints.clear();
  _keypoints.reserve(nkeypoints);
  int offset = 0;
  for (int level = 0; level < nlevels; ++level) {
    std::vector<cv::KeyPoint>& scaled_keypoints = allKeypoints[level];
      int nkeypointsLevel = (int)scaled_keypoints.size();
      if(nkeypointsLevel==0)
          continue;
      // preprocess the resized image
      cv::Mat workingMat = mvImagePyramid[level].clone();
      cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
      // Compute the descriptors
      cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
      computeDescriptors(workingMat, scaled_keypoints, desc, pattern);
      offset += nkeypointsLevel;
      // Scale keypoint coordinates
      if (level != 0) {
          float scale = mvScaleFactor[level];
          for (std::vector<cv::KeyPoint>::iterator keypoint = scaled_keypoints.begin(),
               keypointEnd = scaled_keypoints.end(); keypoint != keypointEnd; ++keypoint)
              keypoint->pt *= scale;
      }
      // And add the keypoints to the output
      _keypoints.insert(_keypoints.end(), scaled_keypoints.begin(), scaled_keypoints.end());
  }
#endif
  //static auto orb_ = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
  //orb_->detect(gray, keypoints);
  //orb_->compute(gray, keypoints, descriptors);
  return;
}

double CvFeatureDescriptor::GetDistance(const cv::Mat& desc0, const cv::Mat& desc1) const {
  return ORB_SLAM2::ORBextractor::DescriptorDistance(desc0, desc1);
}


} // namespace OLD_SEG
