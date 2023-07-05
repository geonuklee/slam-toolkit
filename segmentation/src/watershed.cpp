#include "../include/seg.h"
#include "../include/util.h"

cv::Mat FlowError2Edge(const cv::Mat flow_errors, const cv::Mat expd_diffedges, const cv::Mat valid_mask) {
  /* Done
  * [x] edge의 threshold는 solvePnPRansac inliers의 median rprj error를 참고해서 지정
  * [x] "a)valid mask 내에서, b)threshold이하의 flow error를 가지는 영역" 을 outline edge로 정의.
    -> outline edge로부터 DistanceWatershed를 수행해, dynamic instance segmentation 결과물 획득.
  */

  cv::Mat outline_edges = flow_errors < 0.1;
  cv::bitwise_and(outline_edges, valid_mask, outline_edges);
  /*
  {
    cv::Mat zeros = cv::Mat::zeros(flow_errors.rows, flow_errors.cols, CV_8UC1);
    std::vector<cv::Mat> vec = {255*outline_edges, 100*expd_diffedges, zeros};
    cv::Mat dst; cv::merge(vec, dst);
    cv::imshow("err_edges", dst);
  }
  */
  cv::bitwise_and(outline_edges, expd_diffedges, outline_edges);
  //cv::imshow("only_err_edges", outline_edges*255);
  return outline_edges;
}

cv::Mat FlowDifference2Edge(cv::Mat score) {
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

  const float high_threshold = .3;
  const float low_threshold  = .1;
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

int Convert(const std::map<int,int>& convert_lists,
             const std::set<int>& leaf_seeds,
             const cv::Mat& filtered_edge, 
             cv::Mat& seed ){
  int bg_idx = std::max<int>(100, (*leaf_seeds.rbegin()) + 1);
  int* ptr = (int*) seed.data;
  const int n = seed.rows*seed.cols;

  for(int r = 0; r < n; r++){
      int& idx = ptr[r];
      if(convert_lists.count(idx) )
        idx = convert_lists.at(idx);
      else if(!leaf_seeds.count(idx) )
        idx = 0;
  }
  return bg_idx;
}

cv::Mat Unify(cv::Mat _marker){
cv::Mat _output = cv::Mat::zeros(_marker.rows, _marker.cols, CV_32SC1);
int* marker = _marker.ptr<int>();
int* output = _output.ptr<int>();
std::map<int,int> cvt_map;
cvt_map[0] = 0;
int n = _marker.total();
for(int i=0; i<n; i++){
  const int& m0 = marker[i];
  if(!cvt_map.count(m0))
    cvt_map[m0] = cvt_map.size();
  output[i] = cvt_map[m0];
}
return _output;
}

bool IsTooSmallSeed(const cv::RotatedRect& obb, const int& rows, const int& cols){
  // 화면 구석 너무 작은 seed 쪼가리 때문에 oversegment 나는것 방지.
  const cv::Point2f& cp = obb.center;
  const float offset = 10.;
  if(cp.x < offset)
    return true;
  else if(cp.x > cols - offset)
    return true;
  else if(cp.y < offset)
    return true;
  else if(cp.y > rows - offset)
    return true;
  float wh = std::max(obb.size.width, obb.size.height);
  return wh < 20.;
}

cv::Mat Segment(const cv::Mat outline_edge, const cv::Mat rgb4vis) {
  const bool verbose = !rgb4vis.empty();
  cv::Mat marker;
  cv::Mat dist_fromoutline; {
#if 1
    cv::distanceTransform(outline_edge<1, dist_fromoutline, cv::DIST_L2, cv::DIST_MASK_PRECISE);
#else
    cv::Mat divided;
    cv::bitwise_and(depthmask, ~outline_edge, divided);
    cv::bitwise_and(vignett8U_, divided, divided);
    if(divided.type()!=CV_8UC1)
      divided.convertTo(divided, CV_8UC1); // distanceTransform asks CV_8UC1 input.
    cv::distanceTransform(divided, dist_transform, cv::DIST_L2, cv::DIST_MASK_3);
#endif
    /*for(int r=0; r<dist_fromoutline.rows; r++)
      for(int c=0; c<dist_fromoutline.cols; c++)
        if(!validmask.at<unsigned char>(r,c))
          dist_fromoutline.at<float>(r,c) = 0.;*/

  }
  const int rows = outline_edge.rows;
  const int cols = outline_edge.cols;

  cv::Mat seedmap = cv::Mat::zeros(rows, cols, CV_32SC1);
  cv::Mat seed_contours;
  std::set<int> leaf_nodes;
  int bg_idx = -1;
  const int mode   = cv::RETR_TREE; // RETR_CCOMP -> RETR_EXTERNAL
  const int method = cv::CHAIN_APPROX_SIMPLE;
  double dth = 4.; 
  //if(verbose)
  //  dth = 8.; // For readability of figure
  int n = 100./dth; // max level should be limitted.
  //int n = 1;
  float min_width = 10.;
  std::map<int,std::set<int> > seed_childs;
  std::map<int,int> seed_parents;
  std::map<int,cv::RotatedRect> seed_obbs;
  std::map<int,float> seed_areas;
  std::map<int,int> seed_levels;
  std::map<int,int> seed_dists;
    
  std::vector<std::vector<cv::Point> > vis_contours, vis_invalid_contours;
  {
    int idx = 1; // Assign 1 for edge. Ref) utils.cpp, GetColoredLabel
    for(int lv = 0; lv < n; lv++){
      int th_distance = dth*(float)lv;
      cv::Mat local_seed;
      cv::threshold(dist_fromoutline, local_seed, th_distance, 255, cv::THRESH_BINARY);
      local_seed.convertTo(local_seed, CV_8UC1); // findContour support only CV_8UC1

      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(local_seed, contours, hierarchy, mode, method);
      int n_insertion=0;
      for(size_t j=0; j < contours.size(); j++){
        const std::vector<cv::Point>& cnt = contours.at(j);
        const cv::Vec4i& vec = hierarchy.at(j);
        if(vec[3] > -1)
          continue;
        const int& x = cnt.at(0).x;
        const int& y = cnt.at(0).y;
        const int exist_idx = seedmap.at<int>(y,x);
        const cv::RotatedRect ar = cv::minAreaRect(cnt);
        if(std::min(ar.size.width,ar.size.height) < min_width){
          if(verbose)
            vis_invalid_contours.push_back(cnt);
          continue;
        }
        idx += 1;
        assert(exist_idx!=idx);
        n_insertion++;
        seed_childs[exist_idx].insert(idx);
        seed_parents[idx] = exist_idx;
        seed_obbs[idx] = ar;
        seed_areas[idx] = cv::contourArea(cnt);
        seed_levels[idx] = lv;
        seed_dists[idx] = th_distance;
        std::vector<std::vector<cv::Point> > cnts = { cnt, };
        cv::drawContours(seedmap, cnts, 0, idx,-1);
        seed_childs[idx];
        if(verbose)
          vis_contours.push_back(cnt);
      }
      if(n_insertion < 1)
        break;
    }
  }
  seed_contours = seedmap.clone();
  for(const auto& it : seed_childs){
    const int& parent = it.first;
    const std::set<int>& childs = it.second;
    if(parent == 0){
      for(const int& child : childs)
        if(seed_childs.at(child).empty())
          leaf_nodes.insert(child);
    }
    //else if(childs.size() == 1){ // No sibling.
    //  int single_child = *childs.begin();
    //  // When parent has no sibling, the single child is the leaf.
    //  if(seed_childs.at(single_child).empty())
    //    leaf_nodes.insert(single_child);
    //}
    else if( childs.empty() )
      leaf_nodes.insert(parent);
  }
  std::map<int, int> convert_lists;
  // The pairs of highest and lowest contours for each instance.
  std::map<int,int> lowest2highest;
  // The below loop updates convert_lists
  for(const int& idx : leaf_nodes){
    if(convert_lists.count(idx))
      continue;
    std::set<int> contours_under_pole;
    std::priority_queue<int> q1;
    q1.push(idx);
    while(!q1.empty()){
      // Descent from pole to lower contour
      const int keyidx = q1.top();
      q1.pop();
      int parent = seed_parents.at(keyidx);
      std::set<int> siblings_of_key;
      if(seed_childs.count(parent)){
        siblings_of_key = seed_childs.at(parent);
      }
      contours_under_pole.insert(keyidx);
      if(parent==0){
        continue; // No q1.push(parent). Stop descendent.
      }
      else if(siblings_of_key.size() == 1){
        // 자식노드 숫자 count해서 내려가기전,..
        //const cv::RotatedRect& keyidx_obb = seed_obbs.at(keyidx);
        //const cv::RotatedRect& parent_obb = seed_obbs.at(parent);
        //const int& lv = seed_levels.at(keyidx);
        //if(lv > 0 &&  std::min(keyidx_obb.size.width,keyidx_obb.size.height) > 50) {
        //  const float expectation =(keyidx_obb.size.width+2.*dth)*(keyidx_obb.size.height+2.*dth);
        //  const float parent_area = parent_obb.size.width*parent_obb.size.height;
        //  const float err_ratio = std::abs(expectation-parent_area)/parent_area;
        //  if(err_ratio > 0.5)
        //    continue; // No q1.push(parent). Stop descendent.
        //}
        q1.push(parent); // Keep descent to a lower contour.
      }
      else if(siblings_of_key.size()==2) {
#if 1
        continue;
#else
        float sum = 0;
        for(const int& sibling : siblings_of_key)
          sum += seed_areas.at(sibling);
        float parent_area = seed_areas.at(parent);
        const cv::RotatedRect& parent_obb = seed_obbs.at(parent);
        const float ratio = (parent_obb.size.width-2.*dth)*(parent_obb.size.width-2.*dth)/
          (parent_obb.size.width*parent_obb.size.width);
        const float expected_area_sum = ratio*parent_area;
        if(sum > .8*expected_area_sum)
          continue; // No q1.push(parent). Stop descendent.
        else{
          q1.push(parent); // Keep descent to a lower contour.
          // sibling > 1 이지만, 하나로 묶어야 하는 경우 <<<
          for(const int& sibling : siblings_of_key){
            if(sibling == keyidx)
              continue;
            // Get all upper contours of sibling.
            std::queue<int> opened;
            opened.push(sibling);
            while(!opened.empty()){
              int lkeyidx = opened.front();
              opened.pop();
              contours_under_pole.insert(lkeyidx);
              for(const int& child : seed_childs.at(lkeyidx) )
                opened.push(child);
            }
          }
        }
#endif
      }
    }
    int lowest_contour = *contours_under_pole.begin(); // lowest boundary = max(contours_under_pole)
    int highest_contour = *contours_under_pole.rbegin(); // pole = min(contours_under_pole)
    const cv::RotatedRect& lowest_obb = seed_obbs.at(lowest_contour);
    if(IsTooSmallSeed(lowest_obb,seedmap.rows,seedmap.cols)){
      for(const int& i : contours_under_pole)
        convert_lists[i] = 0;
    }
    else{
      lowest2highest[lowest_contour] = highest_contour;
      for(const int& i : contours_under_pole){
        if(i != lowest_contour)
          convert_lists[i] = lowest_contour;
      }
    }
  } // compute leaf_seeds
  if(!lowest2highest.empty()) {
    // Convert elements of seed from exist_idx to convert_lists
    bg_idx = Convert(convert_lists, leaf_nodes, outline_edge, seedmap);
  }
  else
    bg_idx = 2;

  seedmap = cv::max(seedmap,0);
  marker = Unify(seedmap);

  cv::Mat vis_arealimitedflood, vis_rangelimitedflood, vis_onedgesflood;
  cv::Mat vis_heightmap, vis_seed;
  if(verbose){
    vis_arealimitedflood  = cv::Mat::zeros(rows,cols, CV_8UC3);
    vis_rangelimitedflood = cv::Mat::zeros(rows,cols, CV_8UC3);
    vis_onedgesflood      = cv::Mat::zeros(rows,cols, CV_8UC3);
    cv::normalize(-dist_fromoutline, vis_heightmap, 255, 0, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(vis_heightmap,vis_heightmap, CV_GRAY2BGR);
    for(int r=0; r<vis_heightmap.rows; r++){
      for(int c=0; c<vis_heightmap.cols; c++){
        if(outline_edge.at<unsigned char>(r,c) <1)
          continue;
        vis_heightmap.at<cv::Vec3b>(r,c)[0] 
          = vis_heightmap.at<cv::Vec3b>(r,c)[1] = 0;
      }
    }
    cv::Mat fg = marker > 0;
    const int mode   = cv::RETR_EXTERNAL;
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fg,contours,hierarchy,mode,method);
    for(int i = 0; i < contours.size(); i++){
      const std::vector<cv::Point>& contour = contours.at(i);
      float min_ed = 99999.;
      for(const cv::Point& pt :  contours.at(i) ){
        const float& ed = dist_fromoutline.at<float>(pt.y,pt.x);
        min_ed = std::min(min_ed, ed);
      }
      //const int thick = 1+2*min_ed;
      //const int thick = std::min<int>(100,1+2*min_ed );
      const int thick = std::min<int>(500,1+2*min_ed );
      cv::drawContours(fg, contours, i, 1, thick);
    }

    vis_seed = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::Mat weights = cv::Mat::zeros(rows, cols, CV_32F);
    for(int i = 0; i < vis_contours.size(); i++){
      const std::vector<cv::Point>& contour = vis_contours.at(i);
      const auto pt = contour.at(0);
      float w = float(vis_heightmap.at<cv::Vec3b>(pt.y, pt.x)[0]) / 255.;
      w = std::min<float>(1.,w);
      w = std::max<float>(0.,w);
      cv::drawContours(weights, vis_contours, i, w, -1);
    }

    for(int r=0; r<marker.rows; r++){
      for(int c=0; c<marker.cols; c++){
        auto& cs = vis_seed.at<cv::Vec3b>(r,c);
        auto& ca = vis_arealimitedflood.at<cv::Vec3b>(r,c);
        auto& w = weights.at<float>(r,c);
        const int& m = marker.at<int>(r,c);
        if(m > 0){
          cs[0] = w * colors.at(m%colors.size())[0];
          cs[1] = w * colors.at(m%colors.size())[1];
          cs[2] = w * colors.at(m%colors.size())[2];
          ca[0] = .5 * colors.at(m%colors.size())[0];
          ca[1] = .5 * colors.at(m%colors.size())[1];
          ca[2] = .5 * colors.at(m%colors.size())[2];
          //ca[0] = ca[1] = ca[2] = 0;
          continue;
        }
        else
          ca[0] = ca[1] = ca[2] = 255;
        const float& ed = dist_fromoutline.at<float>(r,c);
        if(ed > 0){
          if(fg.at<uchar>(r,c) > 0)
            cs[0] = cs[1] = cs[2] = 100;
          else
            cs[0] = cs[1] = cs[2] = 255;
          continue;
        }
        else{
          // ca[0] = ca[1] = 0; ca[2] = 255; // Draw edges as red later.
          vis_seed.at<cv::Vec3b>(r,c)[2] = 255;
          vis_seed.at<cv::Vec3b>(r,c)[0] = vis_seed.at<cv::Vec3b>(r,c)[1] = 0;
        }
      }
    }
    for(int i = 0; i < vis_contours.size(); i++){
      const std::vector<cv::Point>& contour = vis_contours.at(i);
      const auto pt = contour.at(0);
      //uchar w = 255. * weights.at<float>(pt.y,pt.x);
      uchar w = 200;
      cv::drawContours(vis_seed, vis_contours, i, CV_RGB(w,w,w), 1);
    }
    for(int i = 0; i < vis_invalid_contours.size(); i++){
      cv::drawContours(vis_seed, vis_invalid_contours, i, CV_RGB(0,0,0), 1);
    }

  } // if(verbose)
  DistanceWatershed(dist_fromoutline, marker,
                    vis_arealimitedflood,
                    vis_rangelimitedflood,
                    vis_onedgesflood);
  marker = cv::max(marker, 0);
  if(verbose){
    {
      cv::Mat fg = seedmap > 0;
      const int mode   = cv::RETR_EXTERNAL;
      const int method = cv::CHAIN_APPROX_SIMPLE;
      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(fg,contours,hierarchy,mode,method);
      for(int i = 0; i < contours.size(); i++)
        cv::drawContours(vis_arealimitedflood, contours, i, CV_RGB(0,0,0), 2);
      for(int r=0; r<marker.rows; r++){
        for(int c=0; c<marker.cols; c++){
          auto& ca = vis_arealimitedflood.at<cv::Vec3b>(r,c);
          if(outline_edge.at<uchar>(r,c) < 1)
            continue;
          ca[0] = ca[1] = 0; ca[2] = 255;
        }
      }
    }
    //cv::imshow(name_+"marker", GetColoredLabel(marker,true) );
    //cv::imshow(name_+"outline", outline_edge*255);
    //cv::imshow(name_+"seed contour", GetColoredLabel(seed_contours));
    //cv::imshow(name_+"seed", Overlap(rgb,seedmap) );
    //cv::Mat dst_marker = Overlap(rgb,marker);
    //cv::imshow(name_+"final_marker", dst_marker );
    {
      cv::Mat dst = rgb4vis.clone();
      for(int r=0; r<dst.rows; r++){
        for(int c=0; c<dst.cols; c++){
          auto& cd = dst.at<cv::Vec3b>(r,c);
          if(outline_edge.at<uchar>(r,c) < 1)
            continue;
          cd[0] = cd[1] = 0; cd[2] = 255;
        }
      }
      cv::imshow("vis_rgb",      dst );
      cv::imwrite("vis_rgb.png", dst );
    }

    cv::imshow("vis_heightmap", vis_heightmap);
    cv::imwrite("vis_heightmap.png", vis_heightmap);

    cv::imshow("vis_seed", vis_seed);
    cv::imwrite("vis_seed.png", vis_seed);

    cv::imshow("vis_arealimitedflood", vis_arealimitedflood);
    cv::imwrite("vis_arealimitedflood.png", vis_arealimitedflood);

    cv::imshow("vis_rangelimitedflood", vis_rangelimitedflood);
    cv::imwrite("vis_rangelimitedflood.png", vis_rangelimitedflood);
    cv::imshow("vis_onedgesflood", vis_onedgesflood);
    cv::imwrite("vis_onedgesflood.png", vis_onedgesflood);

    cv::imshow("after merge", GetColoredLabel(marker));
    //cv::Mat norm_depth, norm_dist;
    //cv::normalize(depth, norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imwrite(name_+"depth.png", norm_depth);
    //cv::imwrite(name_+"rgb.png", rgb);
    //cv::imwrite(name_+"heightmap.png", vis_heightmap);
    //cv::imwrite(name_+"seed.png", GetColoredLabel(seedmap));
    //cv::imwrite(name_+"marker.png", GetColoredLabel(marker));
  }
  return marker;
}

void DistanceWatershed(const cv::Mat _dist_fromedge,
                       cv::Mat& _marker,
                       cv::Mat& vis_arealimitedflood,
                       cv::Mat& vis_rangelimitedflood,
                       cv::Mat& vis_onedgesflood
                       ){

  const int IN_QUEUE = -2; // Pixel visited
  const int WSHED = -1;    // Pixel belongs to watershed
  const cv::Size size = _marker.size();
  cv::Mat _expandmap = cv::Mat::zeros(_marker.rows,_marker.cols, CV_32FC1);
  struct Node {
    int* m;
    int* m_parent;
    float* expd; // expand distance
    const float* ed;

    Node(int* _m, int* _m_parent,
         float* _expd, float* _expd_parent,
         const float* _ed)
      :m(_m), m_parent(_m_parent), expd(_expd), ed(_ed){
        m[0] = IN_QUEUE;
        expd[0] = 1. + _expd_parent[0];
    }

    bool operator < (const Node& other) const{
      int a = static_cast<int>(*ed);
      int b = static_cast<int>(*other.ed);
      if( a < b )
        return true;
      else if( a > b)
        return false;
      else if(expd[0] > other.expd[0])
        return true;
      return false;
    }
  };
  std::priority_queue<Node> q1, q2;

  struct ENode {
    int* m;
    int* m_parent;
    float expd; // expand distance

    ENode(int* _m, int* _m_parent, float _expd)
      :m(_m), m_parent(_m_parent), expd(_expd){
        m[0] = IN_QUEUE;
      }
    bool operator < (const ENode& other) const{
      if(expd > other.expd)
        return true;
      return false;
    }
  };
  std::priority_queue<ENode> q3;


  // Current pixel in input image
  int* marker = _marker.ptr<int>();
  // Step size to next row in input image
  // ref) https://answers.opencv.org/question/3854/different-step-size-output-for-cvmatstep1/
  const int mstep = int(_marker.step/sizeof(marker[0]));
  const float* edge_distance = _dist_fromedge.ptr<float>();
  float* expand_distance = _expandmap.ptr<float>();
  const int dstep = int(_dist_fromedge.step/sizeof(edge_distance[0]));

  // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
  int i, j;
  for( j = 0; j < size.width; j++ )
    marker[j] = marker[j + mstep*(size.height-1)] = WSHED;

  int n_instance = 0;
  for( i = 1; i < size.height-1; i++ ) {
    marker += mstep;
    edge_distance += dstep;
    expand_distance += dstep;
    marker[0] = marker[size.width-1] = WSHED; // boundary pixels

    // initial phase: put all the neighbor pixels of each marker to the priority queue -
    // determine the initial boundaries of the basins
    for( j = 1; j < size.width-1; j++ ) {
      int* m = marker + j;
      float* expd     = expand_distance + j;
      const float* ed = edge_distance   + j;
      if(*ed <.1)
        *m = WSHED;
      n_instance = std::max(*m, n_instance);
      if( m[0] != 0)
        continue;
      int n = q1.size();
      if(m[-1] > 0)
        q1.push(Node(m, m-1, expd, expd-1, ed));
      else if(m[1] > 0)
        q1.push(Node(m, m+1, expd, expd+1, ed));
      else if(m[-mstep] > 0)
        q1.push(Node(m, m-mstep, expd, expd-dstep, ed));
      else if(m[mstep] > 0)
        q1.push(Node(m, m+mstep, expd, expd+dstep, ed));
    }
  }
  n_instance += 1;

  std::vector<int> remain_expand_areas;
  remain_expand_areas.resize(n_instance, 200);
  marker = _marker.ptr<int>();
  for( i = 1; i < size.height-1; i++ ) {
    marker += mstep;
    for( j = 1; j < size.width-1; j++ ) {
      int* m = marker + j;
      if(*m < 1)
        continue;
      int& s = remain_expand_areas[*m];
      s = std::max(s--, 0);
    }
  }

  std::vector<std::map<int, size_t> > direct_boundary_counts;
  direct_boundary_counts.resize(n_instance);
  std::vector<std::map<int, size_t> > indirect_boundary_counts;
  indirect_boundary_counts.resize(n_instance);

  int iter = 0;
  /*
  cv::VideoWriter writer; 
  int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  cv::Size sizeFrame(640,480);
  writer.open("modified_watershed.mp4", codec, 15, sizeFrame, true);
  */
  // First step - Expand each instance in limited area

// Counts boundary with edges
#define ws_check(idx){ \
  if(k.m[idx]>0){ \
    if(k.m[idx]!=*k.m) \
      direct_boundary_counts[std::max(*k.m,k.m[idx])][std::min(*k.m,k.m[idx])]++;\
  }  \
}

#define ws_push(idx){ \
  if(k.m[idx]==WSHED) \
    neighbor_shed = true; \
  else if(k.m[idx] == 0) \
    q1.push(Node(k.m+idx, k.m, k.expd+idx, k.expd, k.ed+idx) ); \
}
  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    *k.m = *k.m_parent;
    ws_check(-1);
    ws_check(1);
    ws_check(-mstep);
    ws_check(mstep);

    int& area = remain_expand_areas[*k.m];
    //if(*k.ed < 20. && area < 1){
    if(area < 1){
      *k.m = IN_QUEUE;
      q2.push(k);
      continue;
    }
    area--;
    int n = q1.size();
    bool neighbor_shed = false;
    ws_push(-1);
    ws_push(1);
    ws_push(-mstep);
    ws_push(mstep);
    if(q1.size() == n && neighbor_shed){
      *k.m = IN_QUEUE;
      q3.push( ENode(k.m,k.m_parent,*k.expd) );
    }
    /* if( (++iter) % 10 == 0){
      cv::Mat dst = GetColoredLabel(_marker);
      cv::imshow("ModifiedWatershed1", dst);
      if('q' == cv::waitKey(1))
        exit(1);
      //writer.write(dst);
    }*/
  }
#undef ws_push

  if(!vis_arealimitedflood.empty()){
    cv::Mat fg = _marker > 0;
    const int mode   = cv::RETR_EXTERNAL;
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fg,contours,hierarchy,mode,method);
    for(i = 0; i < contours.size(); i++){
      const std::vector<cv::Point>& contour = contours.at(i);
      float ed = 99999.;
      for(const cv::Point& ipt : contours.at(i))
        ed = std::min(ed, _dist_fromedge.at<float>(ipt.y,ipt.x) );
      const cv::Point& pt = contours.at(i).at(0);
      const int& m = _marker.at<int>(pt.y,pt.x);
      const int thick = 1+2*ed;
      cv::drawContours(fg, contours, i, 1, thick);
    }
    cv::findContours(fg,contours,hierarchy,mode,method);

    for(int r=0; r<_marker.rows; r++){
      for(int c=0; c<_marker.cols; c++){
        const int& m = _marker.at<int>(r,c);
        cv::Vec3b& ca = vis_arealimitedflood.at<cv::Vec3b>(r,c);
        if(m < 1){
          if(fg.at<uchar>(r,c) >0){
            ca[0] = ca[1] = ca[2] = 100;
          }
          continue;
        }
        //if(ca[0] != 255 || ca[1] != 255 || ca[2] != 255)
        //  continue;
        const auto& co = colors.at(m%colors.size());
        ca[0] = co[0]; ca[1] = co[1]; ca[2] = co[2];
      }
    }
    //for(i = 0; i < contours.size(); i++)
    //  cv::drawContours(vis_arealimitedflood, contours, i, CV_RGB(0,0,0), 2);
  }

  // Second step - Expand each instance in limited range
  {
    cv::Mat _fg = _marker > 0;
    const int mode   = cv::RETR_EXTERNAL;
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(_fg,contours,hierarchy,mode,method);
    const float margin_ratio = 1.4; // sqrt(2)
    for(i = 0; i < contours.size(); i++){
      const std::vector<cv::Point>& contour = contours.at(i);
#if 0
      int n_contour = contour.size();
      for(j=0; j < n_contour; j++){
        const cv::Point& pt0 = contour.at(j);
        const cv::Point& pt1 = contour.at((j+1)%n_contour);
        const int thick = 1+2*_dist_fromedge.at<float>(pt0.y,pt0.x);
        cv::line(_fg,pt0, pt1, 1, thick);
      }
#else
      float ed = std::max(_marker.rows, _marker.cols);
      for(const cv::Point& ipt : contours.at(i))
        ed = std::min(ed, _dist_fromedge.at<float>(ipt.y,ipt.x) );
      const cv::Point& pt = contours.at(i).at(0);
      const int& m = _marker.at<int>(pt.y,pt.x);
      const int thick = 1+margin_ratio*2.*ed;
      cv::drawContours(_fg, contours, i, 1, thick);
#endif
    }
    unsigned char* fg = _fg.ptr<unsigned char>();
    marker = _marker.ptr<int>();
    for(i = 0; i < _fg.total(); i++)
      if(fg[i] < 1)
        marker[i] = WSHED;
  }

#define ws_push(idx){ \
  if(k.m[idx]==WSHED) \
    neighbor_shed = true; \
  else if(k.m[idx] == 0) \
    q2.push(Node(k.m+idx, k.m, k.expd+idx, k.expd, k.ed+idx) ); \
}
  while(!q2.empty()){
    Node k = q2.top(); q2.pop();
    *k.m = *k.m_parent;
    ws_check(-1);
    ws_check(1);
    ws_check(-mstep);
    ws_check(mstep);

    int n = q2.size();
    bool neighbor_shed = false;
    ws_push(-1);
    ws_push(1);
    ws_push(-mstep);
    ws_push(mstep);
    if(q2.size() == n && neighbor_shed){
      *k.m = IN_QUEUE;
      q3.push( ENode(k.m,k.m_parent,0.) );
    }
    /* if( (++iter) % 100 == 0){
      cv::Mat dst = GetColoredLabel(_marker);
      cv::imshow("ModifiedWatershed2", dst);
      if(cv::waitKey(1) == 'q')
        exit(1);
    } */
  }
#undef ws_push
#undef ws_check

  if(!vis_rangelimitedflood.empty()){
    //vis_rangelimitedflood = GetColoredLabel(_marker);
    for(int r=0; r<_marker.rows; r++){
      for(int c=0; c<_marker.cols; c++){
        const int& m = _marker.at<int>(r,c);
        auto& vr = vis_rangelimitedflood.at<cv::Vec3b>(r,c);
        if(m > 0){
          const auto& co = colors.at(m%colors.size());
          vr[0] = co[0]; vr[1] = co[1]; vr[2] = co[2];
          continue;
        }
        if(_dist_fromedge.at<float>(r,c) < 2.){
          vr[0] = vr[1] = 0; vr[2] = 255;
        }
        else
          vr[0] = vr[1] = vr[2] = 255;
      }
    }
  }

#define ws_check(idx){ \
  if(k.m[idx]>0){ \
    if(k.m[idx]!=*k.m) \
      indirect_boundary_counts[std::max(*k.m,k.m[idx])][std::min(*k.m,k.m[idx])]++;\
  }  \
}
#define ws_push(idx){ \
  if(k.m[idx] == 0) \
    q3.push(ENode(k.m+idx, k.m, k.expd+1.) ); \
}

  {
    cv::Mat marker0 = _marker.clone();
    marker = _marker.ptr<int>();
    for( j = 0; j < size.width; j++ )
      marker[j] = marker[j + mstep*(size.height-1)] = WSHED;
    for( i = 1; i < size.height-1; i++ ) {
      marker += mstep;
      marker[0] = marker[size.width-1] = WSHED; // boundary pixels
      for( j = 1; j < size.width-1; j++ ) {
        int* m = marker + j;
        if(*m == WSHED)
          *m = 0;
      }
    }
    while(!q3.empty()){
      ENode k = q3.top(); q3.pop();
      *k.m = *k.m_parent;
      if(k.expd > 5.) // Marker extend range
        continue;
      ws_check(-1);
      ws_check(1);
      ws_check(-mstep);
      ws_check(mstep);

      ws_push(-1);
      ws_push(1);
      ws_push(mstep);
      ws_push(-mstep);
    }

    //cv::imshow("ext_marker", GetColoredLabel(_marker,true));
    //_marker = marker0;
  }
#undef ws_push
  if(!vis_onedgesflood.empty()){
    //vis_onedgesflood = GetColoredLabel(_marker);
    for(int r=0; r<_marker.rows; r++){
      for(int c=0; c<_marker.cols; c++){
        auto& ce = vis_onedgesflood.at<cv::Vec3b>(r,c);
        int& m = _marker.at<int>(r,c);
        if(m > 0){
          const auto& co = colors.at(m%colors.size());
          ce[0] = co[0]; ce[1] = co[1]; ce[2] = co[2];
        }
        else{
          ce[0] = ce[1] = ce[2] = 255;
        }
      }
    }
  }

  int n_merge = 0;
  {  // Need fix with bug
    struct DSNode{ // Disjoint-set structure
      DSNode* parent;
      DSNode() : parent(nullptr) { }
    };
    std::vector<DSNode> _nodes;
    _nodes.resize(n_instance);
    DSNode* nodes = _nodes.data();
    for(int m0 =0; m0 < n_instance; m0++){
      const auto& counts = direct_boundary_counts[m0];
      for(const auto& it : counts){
        int m1 = it.first;
        if(m1 < 1)
          continue;
        float n_contact = it.second;
        float n_edge = 0.;
        if(indirect_boundary_counts[m0].count(m1))
          n_edge = indirect_boundary_counts[m0][m1];
        float n_all = n_contact+n_edge;
        if(n_edge > .5*n_all)
          continue;
        //printf("Pair between %d,%d. Contact %.2f/%.f\n",std::min(m0,m1),std::max(m0,m1),n_contact,n_all);
        DSNode* child = nodes+std::max(m0,m1);
        child->parent = nodes+std::min(m0,m1);
        n_merge += 1;
      }
    }
    std::vector<int> convert_list;
    convert_list.resize(n_instance,-1);
    for(i=1; i < n_instance; i++) {
      DSNode* keynode = nodes+i;
      while(keynode->parent){
        //std::cout << "move to parent, #" << i << std::endl;
        keynode = keynode->parent;
      }
      convert_list[i] = keynode - nodes;
    }
    //cv::imshow("beforemerge", GetColoredLabel(_marker,true));
    marker = _marker.ptr<int>();
    for( i = 1; i < size.height-1; i++ ) {
      marker += mstep;
      for( j = 1; j < size.width-1; j++ ) {
        int* m = marker + j;
        if(*m < 1)
          continue;
        const int& new_m = convert_list[*m];
        if(new_m < 0)
          continue;
        *m = new_m;
      }
    }
    //cv::imshow("aftermerge", GetColoredLabel(_marker,true));
    //if('q'==cv::waitKey(n_merge>0?0:1))
    //  exit(1);
  }
  //writer.release();
  return;
}

