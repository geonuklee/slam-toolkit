#include "simd.h"
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <list>
//#include "../../segmentation/include/util.h"

std::vector<cv::Scalar> colors = {
  CV_RGB(0,180,0),
  CV_RGB(0,100,0),
  CV_RGB(255,0,255),
  CV_RGB(100,0,255),
  CV_RGB(100,0,100),
  CV_RGB(0,0,180),
  CV_RGB(0,0,100),
  CV_RGB(255,255,0),
  CV_RGB(100,255,0),
  CV_RGB(100,100,0),
  CV_RGB(100,0,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text){
  cv::Mat dst = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);

  std::set<int> labels;

  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(mask.type() == CV_8UC1 && idx == 0)
        continue;
      else if(mask.type() == CV_32S && idx < 0)
        continue;

      cv::Scalar bgr;
      if( idx == 0)
        bgr = CV_RGB(100,100,100);
      //else if (idx == 1)
      //  bgr = CV_RGB(255,255,255);
      else
        bgr = colors.at( idx % colors.size() );

      dst.at<cv::Vec3b>(i,j)[0] = bgr[0];
      dst.at<cv::Vec3b>(i,j)[1] = bgr[1];
      dst.at<cv::Vec3b>(i,j)[2] = bgr[2];

      labels.insert(idx);
    }
  }

  if(put_text){
    for(int l : labels){
      cv::Mat fg = mask == l;
      cv::rectangle(fg, cv::Rect(cv::Point(0,0), cv::Point(fg.cols-1,fg.rows-1)), false, 2);

      cv::Mat dist;
      cv::distanceTransform(fg, dist, cv::DIST_L2, cv::DIST_MASK_3);
      double minv, maxv;
      cv::Point minloc, maxloc;
      cv::minMaxLoc(dist, &minv, &maxv, &minloc, &maxloc);
      const auto& c0 = colors.at( l % colors.size() );
      const auto color = (c0[0]+c0[1]+c0[2] > 255*2) ? CV_RGB(0,0,0) : CV_RGB(0,0,0);
      cv::putText(dst, std::to_string(l), maxloc, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
  }
  return dst;
}


namespace OLD {

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

void DistanceWatershed(const cv::Mat _dist_fromedge,
                       cv::Mat& _marker,
                       bool limit_expand_range,
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
  if(limit_expand_range) {
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
        if(n_edge > .2*n_all)
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

cv::Mat Segment(const cv::Mat outline_edge,
                   cv::Mat valid_mask,
                   bool limit_expand_range,
                   cv::Mat rgb4vis) {
  const bool verbose = !rgb4vis.empty();
  if(valid_mask.empty())
    valid_mask = cv::Mat::ones(outline_edge.rows, outline_edge.cols, CV_8UC1);

  cv::Mat marker;
  cv::Mat dist_fromoutline; {
    //cv::distanceTransform(outline_edge<1, dist_fromoutline, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat dinput;
    cv::bitwise_and(outline_edge<1, valid_mask, dinput);
    cv::distanceTransform(dinput, dist_fromoutline, cv::DIST_L2, cv::DIST_MASK_PRECISE);
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
  DistanceWatershed(dist_fromoutline, marker, limit_expand_range,
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

} // namespace OLD

#include <immintrin.h> // sse or avx
namespace NEW {
struct MarkerNode {
  int x, y;
  int* m;
  int* m_parent;
  int lv;
  MarkerNode(int _x, int _y, int* _m, int* _m_parent, int _lv)
    : x(_x), y(_y), m(_m), m_parent(_m_parent), lv(_lv) {
    }
  bool operator < (const MarkerNode& other) const{
    return lv > other.lv;
  }
};

int BreadthFirstSearch(cv::Mat& _marker){
  const int L_EDGE = -1;
  int l_max = 0;
  const cv::Size size = _marker.size();

  int w1 = size.width-1;
  int h1 = size.height-1;

  std::priority_queue<MarkerNode> q1;
  int* marker = _marker.ptr<int>();
  const int mstep = int(_marker.step/sizeof(marker[0]));
  int y, x;
  for( y = 0; y < size.height; y++ ) {
    for( x = 0; x < size.width; x++ ) {
      int* m = marker+x;
      l_max = std::max(l_max, *m);
      if( m[0] != 0)
        continue;
      // 자신의 marker가 미정인데, 이웃한 label이 있다면
      if(x > 0 && m[-1] > 0)
        q1.push( MarkerNode(x, y, m, m-1,0) );
      else if(x < w1 && m[1] > 0)
        q1.push( MarkerNode(x,y,m, m+1,0) );
      else if(y > 0 && m[-mstep] > 0)
        q1.push( MarkerNode(x, y, m, m-mstep,0) );
      else if(y < h1 && m[mstep] > 0)
        q1.push( MarkerNode(x, y, m, m+mstep,0) );
    }
    marker += mstep;
  }

  while(!q1.empty()){
    MarkerNode k = q1.top(); q1.pop();
    if(*k.m != 0)
      continue;
    *k.m = *k.m_parent;
    if(k.x > 0) {
      int* m_other = k.m-1;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x-1,k.y,m_other, k.m,k.lv+1) );
    }
    if(k.x < w1 ) {
      int* m_other = k.m+1;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x+1,k.y,m_other, k.m,k.lv+1) );
    }
    if(k.y > 0){
      int* m_other = k.m-mstep;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x,k.y-1,m_other, k.m,k.lv+1) );
    }
    if(k.y < h1){
      int* m_other = k.m+mstep;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x,k.y+1,m_other, k.m,k.lv+1) );
    }
  }

  return l_max;
}

void Merge(cv::Mat& _marker,float min_direct_contact_ratio_for_merge=.7){
  #define ws_key(i1, i2){ \
    std::make_pair(std::min(i1,i2),std::max(i1,i2))\
  }
  const cv::Size size = _marker.size();
  int* marker = _marker.ptr<int>();
  const int mstep = int(_marker.step/sizeof(marker[0]));
  int w1 = size.width-1;
  int h1 = size.height-1;
  std::map< std::pair<int, int>, size_t > contact_before_rm_edge;
  int x,y;
  for( y = 0; y < size.height; y++ ) {
    for( x = 0; x < size.width; x++ ) {
      int* m = marker+x;
      if( m[0] < 0)
        continue;
      else if( m[0] > 0){
        // 자신과 다른 이웃 label숫자 세기.
        if(x < w1 && m[1] > 0 && *m != m[1])
          contact_before_rm_edge[ ws_key(*m, m[1]) ]++;
        else if(y < h1 && m[mstep] > 0 && *m != m[mstep])
          contact_before_rm_edge[ ws_key(*m, m[mstep]) ]++;
      }
    }
    marker += mstep;
  }

  _marker.setTo(0, _marker < 0);
  std::map< std::pair<int, int>, size_t > contact_after_rm_edge;
  std::priority_queue<MarkerNode> q1;
  marker = _marker.ptr<int>();
  for( y = 0; y < size.height; y++ ) {
    for( x = 0; x < size.width; x++ ) {
      int* m = marker+x;
      if( m[0] != 0 )
        continue;
      if(x > 0 && m[-1] > 0)
        q1.push( MarkerNode(x, y, m, m-1,0) );
      else if(x < w1 && m[1] > 0)
        q1.push( MarkerNode(x,y,m, m+1,0) );
      else if(y > 0 && m[-mstep] > 0)
        q1.push( MarkerNode(x, y, m, m-mstep,0) );
      else if(y < h1 && m[mstep] > 0)
        q1.push( MarkerNode(x, y, m, m+mstep,0) );
    }
    marker += mstep;
  }
  // TODO  여기서 마저 작업해야한다. edge 지우고 난다음에 extention
  while(!q1.empty()){
    MarkerNode k = q1.top(); q1.pop();
    if(*k.m != 0)
      continue;
    *k.m = *k.m_parent;
    if(k.x > 0) {
      int* m_other = k.m-1;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x-1,k.y,m_other, k.m,k.lv+1) );
      else 
        contact_after_rm_edge[ ws_key(*k.m,*m_other) ]++;
    }
    if(k.x < w1 ) {
      int* m_other = k.m+1;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x+1,k.y,m_other, k.m,k.lv+1) );
      else
        contact_after_rm_edge[ ws_key(*k.m,*m_other) ]++;
    }
    if(k.y > 0){
      int* m_other = k.m-mstep;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x,k.y-1,m_other, k.m,k.lv+1) );
      else
        contact_after_rm_edge[ ws_key(*k.m,*m_other) ]++;
    }
    if(k.y < h1){
      int* m_other = k.m+mstep;
      if(*m_other ==0 )
        q1.push( MarkerNode(k.x,k.y+1,m_other, k.m,k.lv+1) );
      else
        contact_after_rm_edge[ ws_key(*k.m,*m_other) ]++;
    }
  }
  #undef ws_key

  // using adjacency list representation
  class Graph {
 private:
   std::map<int, std::list<int> > adj; // Adjacency list

 public:
   Graph() { }

   void addEdge(int u, int v) {
     adj[u].push_back(v);
     adj[v].push_back(u); // For an undirected graph, add both directions
   }

   // Depth-First Search to find connected components
   void DFS(int v, std::map<int, bool>& visited, std::list<int>& component) {
     visited[v] = true;
     component.push_back(v);

     for (int neighbor : adj[v]) {
       if (!visited[neighbor]) {
         DFS(neighbor, visited, component);
       }
     }
   }

   // Get connected components
   std::vector<std::list<int>> getConnectedComponents() {
     //std::vector<bool> visited(V, false);
     std::map<int, bool> visited;
     std::vector<std::list<int>> components;
     for (auto it : adj) {
       const int i = it.first;
       if (!visited[i]) {
         std::list<int> component;
         DFS(i, visited, component);
         component.sort();
         components.push_back(component);
       }
     }

     return components;
   }
  };


  Graph g;
  for(auto it : contact_before_rm_edge){
    const size_t n_direct = it.second;
    const size_t n_indirect = contact_after_rm_edge[it.first];
    //printf(" (%d,%d) %ld / %ld\n", it.first.first, it.first.second, n_direct,n_direct+n_indirect);
    const size_t n_threshold = min_direct_contact_ratio_for_merge* (float)(n_direct+n_indirect);
    if(n_direct > n_threshold)
      g.addEdge(it.first.first, it.first.second);
  }
  auto groups = g.getConnectedComponents();
  std::map<int, int> converts;
  for(auto g_it : groups ){
    int l_o = *g_it.begin();
    g_it.pop_front();
    for(const int& l : g_it)
      converts[l] = l_o;
  }

  marker = _marker.ptr<int>();
  for( y = 0; y < size.height; y++ ) {
    for( x = 0; x < size.width; x++ ) {
      int& m = marker[x];
      if(converts.count(m))
        m = converts[m];
    }
    marker += mstep;
  }
  /*
  for(auto g_it : groups ){
    std::cout << "{";
    for(int elem : g_it){
      std::cout << elem << ", " ;
    }
    std::cout << "}" << std::endl;
  }
  cv::imshow("after merge", GetColoredLabel(_marker,true) );
  if('q' == cv::waitKey() ){
    exit(1);
  }
  */
  return;
}

void Segment(const cv::Mat outline_edges,
             int n_octave,
             int n_downsample,
             cv::Mat& output) {
  //int n_octave = 6;

  if(output.empty())
    output = cv::Mat::zeros( outline_edges.size(), CV_32S);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  auto t0 = std::chrono::steady_clock::now();
  std::vector<cv::Mat> pyr_edges, pyr_markers;
  pyr_edges.reserve(n_octave);
  pyr_edges.push_back(outline_edges);
  while(pyr_edges.size() < n_octave){
    cv::Mat prev = *pyr_edges.rbegin();
    cv::Mat next;
    cv::dilate(prev, next, kernel);
    cv::resize( prev, next, cv::Size(int(prev.cols/2), int(prev.rows/2)), 0, 0);
    cv::dilate(next, next, kernel);
    pyr_edges.push_back(next>0);
  }
  std::reverse(pyr_edges.begin(), pyr_edges.end());
  //for(int lv=0; lv < pyr_edges.size(); lv++)
  //  std::cout << "edge.size() = " << pyr_edges.at(lv).size() << std::endl;

  cv::Mat label_curr;
  const int Lv = pyr_edges.size()-n_downsample;
  for(int lv=0; lv+1 < Lv; lv++){
    const cv::Mat& edge_curr = pyr_edges.at(lv);
    if(lv==0)
      cv::connectedComponents(~edge_curr,label_curr,4, CV_32SC1); // else. get from prev.
    const cv::Mat& edge_next = pyr_edges.at(lv+1);

    cv::resize(label_curr, label_curr, edge_next.size(), 0, 0, cv::INTER_NEAREST);
    label_curr.setTo(-1, edge_next);

    /* if(!label_curr.empty()){
      cv::Mat dst0;
      cv::resize(label_curr, dst0, outline_edges.size(), 0, 0, cv::INTER_NEAREST);
      cv::imshow("before bfs", GetColoredLabel(dst0) );
      cv::imshow("before bfs"+ std::to_string(lv), GetColoredLabel(label_curr) );
    }
    */

    const bool final_lv = lv+2 == Lv;
    const bool merge = final_lv; // 마지막 pyr 에서 merge 수행.
    int l_max = BreadthFirstSearch(label_curr);
    cv::Mat zero_label = label_curr == 0;
    cv::Mat new_labels;
    cv::connectedComponents( zero_label, new_labels,4, CV_32SC1); // else. get from prev.
    new_labels += l_max;
    new_labels.setTo(0, ~zero_label);
    label_curr += new_labels;

    if(merge)
      Merge(label_curr, n_downsample > 0 ? .4 : .7); // downsample일 경우, 병목지점의 직접접촉에 비해, edge가 상대적으로 두꺼운거 감안.

    if(!final_lv)
      label_curr.setTo(0, edge_next);
  }

  if(n_downsample > 0)
    cv::resize(label_curr, label_curr, outline_edges.size(), 0, 0, cv::INTER_NEAREST);
  /*
  auto t1 = std::chrono::steady_clock::now();
  std::ostringstream os;
  os << std::setprecision(3) << std::chrono::duration<float, std::milli>(t1-t0).count() << "[milli sec]";
  std::cout << "etiem = " << os.str() << std::endl;
  */
  /*
  cv::Mat dst0;
  cv::resize(label_curr, dst0, outline_edges.size(), 0, 0, cv::INTER_NEAREST);
  cv::imshow("after bfs", GetColoredLabel(dst0) );
  cv::imshow("after bfs"+ std::to_string(lv), GetColoredLabel(label_curr) );
  */
  output = label_curr;
  return;
}

} // namespace NEW

