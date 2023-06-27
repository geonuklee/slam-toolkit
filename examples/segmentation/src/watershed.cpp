#include "../include/seg.h"
#include "../include/util.h"

cv::Mat FlowDifference2Edge(cv::Mat score) {
  /* 
    1. Seed 판정 : score > high_threshold.
    2. Seed 확장 : score > low_threshold 인 경우에 모두 확장.
    - 거리 제한? : 이건 추후에 생각해보자..
  */
  struct Node {
    int r_, c_;
    int lv_;

    Node(): r_(-1), c_(-1), lv_(-1){
    }

    Node(int r, int c, int lv)
    : r_(r), c_(c), lv_(lv)  {
      
    }
    bool operator < (const Node& other) const {
      return lv_ < other.lv_;
    }
  };

  const float high_threshold = .5;
  const float low_threshold  = .2;
  const int   R = 10;
  //cv::Mat mask = score > high_threshold;
  cv::Mat mask = cv::Mat::zeros(score.rows,score.cols,CV_32SC1);
  std::priority_queue<Node> open_lists;

  // Init open lists
  for(int r0=0; r0<score.rows; r0++){
    for(int c0=0; c0<score.cols; c0++){
      const auto& s0 = score.at<float>(r0,c0);
      auto& m = mask.at<int>(r0,c0);
      if(s0 < low_threshold){
        m = -1;
        continue;
      }
      else if(s0 < high_threshold){
        continue;
      }
      m = 2;
      open_lists.push( Node(r0,c0,0) );
    }
  }

  // 너비 우선 탐색
  while(!open_lists.empty()){
    Node key = open_lists.top();
    const int& r0 = key.r_;
    const int& c0 = key.c_;
    for(int r1=std::max(0,r0-R); r1<std::min(score.rows,r0+1+R); r1++){
      for(int c1=std::max(0,c0-R); c1<std::min(score.cols,c0+1+R); c1++){
        auto& m = mask.at<int>(r1,c1);
        if(m != 0)
          continue;
        m = 1;
        open_lists.push( Node(r1,c1,key.lv_+1) );
      }
    }
    open_lists.pop();
  }
  mask = mask > 0;
  return mask;
}
/*
cv::Mat EdgeWatershed(const cv::Mat score) {
  struct Node {
    int r_, c_;
    float score_;

    Node(): r_(-1), c_(-1), score_(-1){
    }

    Node(int r, int c, float score)
    : r_(r), c_(c), score_(score)  {
      
    }
    bool operator < (const Node& other) const {
      return score_ < other.score_;
    }
  };
  const float high_threshold = .5;
  const float low_threshold  = .1;
  const int   R = 10;
  cv::Mat mask = cv::Mat::zeros(score.rows,score.cols,CV_32SC1);
  std::priority_queue<Node> open_lists;

  // Init open lists
  for(int r0=0; r0<score.rows; r0++){
    for(int c0=0; c0<score.cols; c0++){
      const auto& s0 = score.at<float>(r0,c0);
      auto& m = mask.at<int>(r0,c0);
      if(s0 < low_threshold){
        m = -1;
        continue;
      }
      else if(s0 < high_threshold){
        continue;
      }
      m = 2;
      open_lists.push( Node(r0,c0,s0) );
    }
  }
  while(!open_lists.empty()){
    Node key = open_lists.top();
    const int& r0 = key.r_;
    const int& c0 = key.c_;

    std::priority_queue<Node> childs;
    for(int r1=std::max(0,r0-R); r1<std::min(score.rows,r0+1+R); r1++){
      for(int c1=std::max(0,c0-R); c1<std::min(score.cols,c0+1+R); c1++){
        const auto& m = mask.at<int>(r1,c1);
        if(m != 0)
          continue;
        const auto& s1 = score.at<float>(r1,c1);
        // open_lists.push( Node(r1,c1,s1) );
        childs.push(Node(r1,c1,s1));
      }
    }
    const int n_child = childs.size();
    if(!childs.empty()){
      Node child = childs.top();
      //cv::circle(mask, cv::Point2f(child.c_,child.r_), R, 1, -1);
      cv::circle(mask, cv::Point2f(child.c_,child.r_), 1, 1, -1);
      open_lists.push( child );
      childs.pop();
      while(!childs.empty()){
        Node other = childs.top();
        cv::circle(mask, cv::Point2f(child.c_,child.r_), 1, -1, -1);
        childs.pop();
      }
    }
    open_lists.pop();
  }
  mask = mask > 0;
  return mask;
}
*/

cv::Mat DistanceWatershed(cv::Mat edges) {
  cv::Mat marker = cv::Mat::zeros(edges.rows, edges.cols, CV_32SC1);
  // TODO 거리값을 잘 활용해야한다.
  cv::Mat outer_dist;
  cv::distanceTransform(~edges, outer_dist, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);

  cv::Mat inner_dist;
  cv::distanceTransform(edges, inner_dist, cv::DIST_L2, cv::DIST_MASK_3, CV_32FC1);


  {
    cv::Mat n0 = convertMat(outer_dist, 0., 50.);
    cv::Mat n1 = convertMat(inner_dist, 0., 10.);
    cv::Mat z  = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {n0, n1, z};
    cv::Mat dst;
    cv::merge(vec,dst);
    cv::imshow("edge_dist", dst);
  }

  return marker;
};

