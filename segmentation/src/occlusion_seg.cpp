#include "occlusion_seg.h"
#include "seg.h"
#include "util.h"

cv::Mat OutlineEdgeDetectorWithSizelimit::ComputeDDEdges(const cv::Mat depth) const {
  // 원경에서는 FP가 안생기게, invd(disparity)를 기준으로 찾아냄.
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  float l = 2;
  std::vector<cv::Point2i> samples = {
    cv::Point2i(l,0),
    cv::Point2i(0,l),
  };

  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt(c0,r0);
      for(const auto& dpt : samples){
        const cv::Point2i pt0 = pt-dpt;
        if(pt0.x < 0 || pt0.x >= depth.cols || pt0.y < 0 || pt0.y >= depth.rows)
          continue;
        const float invz0 = 1. / depth.at<float>(pt0);

        const cv::Point2i pt1 = pt0+dpt;
        if(pt1.x < 0 || pt1.x >= depth.cols || pt1.y < 0 || pt1.y >= depth.rows)
          continue;
        const float invz1 = 1. / depth.at<float>(pt1);
        float th = invz0 * 0.1;
        float diff = invz1 - invz0;
        if(std::abs(diff) < th)
          continue;
        edge.at<unsigned char>(pt0) = diff < 0 ? DdType::FG : DdType::BG;
        edge.at<unsigned char>(pt1) = diff > 0 ? DdType::FG : DdType::BG;
      } // samples
    }
  }

  return edge;
}


cv::Mat OutlineEdgeDetectorWithSizelimit::ComputeConcaveEdges(const cv::Mat depth, const cv::Mat dd_edges, float fx, float fy) const {
  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  const float sample_pixelwidth = 10.; // half height
  const float min_obj_width = 1.; // [meter]

  for(int r0 = sample_pixelwidth; r0 < depth.rows-sample_pixelwidth; r0++) {
    for(int c0 = sample_pixelwidth; c0 < depth.cols-sample_pixelwidth; c0++) {
      const cv::Point2i pt(c0,r0);
      const float& z0 = depth.at<float>(pt);
      uchar& e = edge.at<uchar>(pt);
      if( dd_edges.at<uchar>(pt) )
        continue;
#if 1
      bool left_dd = false;
      float hw = fx * min_obj_width / z0;
      for(int l = 1; l < hw; l++) {
        const cv::Point2i pt_l(pt.x-l, pt.y);
        if(pt_l.x < 0)
          break;
        if(z0 - depth.at<float>(pt_l) > min_obj_width)
          break;
        if(dd_edges.at<uchar>(pt_l) < 1)
          continue;
        left_dd = true;
        break;
      }
      bool right_dd = false;
      for(int l = 1; l < hw; l++) {
        const cv::Point2i pt_r(pt.x + l, pt.y);
        if(pt_r.x >= depth.rows)
          break;
        if(z0 - depth.at<float>(pt_r) > min_obj_width)
          break;
        if(dd_edges.at<uchar>(pt_r) < 1)
          continue;
        right_dd = true;
        break;
      }
      if(left_dd || right_dd)
        continue;
#endif
      // Convexity 계산
#if 1
      float hessian_th = -20.;
      {
        const cv::Point2i pt_y0(pt.x, pt.y-sample_pixelwidth);
        const cv::Point2i pt_y1(pt.x, pt.y+sample_pixelwidth);
        if( pt_y0.y > 0 && pt_y1.y < depth.rows){
          const float dy = sample_pixelwidth/fy*z0;
          const float gy0 = (z0 - depth.at<float>(pt_y0)) / dy;
          const float gy1 = (depth.at<float>(pt_y1) - z0) / dy;
          // atan2(dz0, dy) , atan2(dz1,dy)으로 내각을 구해야... 하지만..
          if( (gy1-gy0)/dy < hessian_th)
            e |= VERTICAL;
        }
      }
      {
        const cv::Point2i pt_x0(pt.x-sample_pixelwidth, pt.y);
        const cv::Point2i pt_x1(pt.x+sample_pixelwidth, pt.y);
        if( pt_x0.x > 0 && pt_x1.x < depth.cols){
          const float dx = sample_pixelwidth/fx*z0;
          const float gx0 = (z0 - depth.at<float>(pt_x0)) / dx;
          const float gx1 = (depth.at<float>(pt_x1) - z0) / dx;
          if( (gx1-gx0)/dx < hessian_th)
            e |= HORIZONTAL;
        }
      }
#else
#endif
    } // cols
  } // rows
  return edge;
}


void OutlineEdgeDetectorWithSizelimit::PutDepth(const cv::Mat depth, float fx, float fy) {
  /*
    * convexity 정확하게 계산하는거 먼저. 최적화는 나중에.
  */
  dd_edges_ = ComputeDDEdges(depth); // concave_edge를 보완해주는 positive detection이 없음.
  concave_edges_ = ComputeConcaveEdges(depth, dd_edges_, fx, fy);
  cv::bitwise_or(dd_edges_ > 0, concave_edges_ > 0, outline_edges_);
  return;
}

cv::Mat MergeOcclusion(const cv::Mat depth,
                       const cv::Mat dd_edges,
                       const cv::Mat _marker0) {
  /*
  TODO 우선 merge occlusion 제대로 되는거 확인한다음,
  distanceTransform(GetBoundaryS(marker))를 또하는 대신, instance segmentation 단계에서 가져와, 연산낭비하는거 수정.
  */
  cv::Mat boundary = GetBoundary(_marker0);
  cv::Mat boundary_distance;
  cv::distanceTransform(boundary<1, boundary_distance, cv::DIST_L2, cv::DIST_MASK_3);
  std::map<int,float> marker_radius; // TODO dist_boundary, marker_radius를 Segmentor 에서 수집.

  const float sqrt2 = std::sqrt(2.);
  const cv::Size size = _marker0.size();
  int w1 = size.width-1;
  int h1 = size.height-1;

  struct Node {
    float cost;
    int x,y; // 아직 marker가 정해지지 않은 grid의 x,y
    int32_t m;   // 부모 marker.
    float z; // latest z
    uchar e; // expanding 과정에서 만난 edge type
    Node(int _x, int _y, float _cost, int32_t _m, float _z, uchar _e) : x(_x), y(_y), cost(_cost), m(_m), z(_z), e(_e) {
    }
    bool operator < (const Node& other) const {
      return cost > other.cost;
    }
  };

  std::priority_queue<Node> q1, q2;
  cv::Mat _marker = _marker0.clone();
  cv::Mat costs   = 999.*cv::Mat::ones(_marker0.rows, _marker0.cols, CV_32FC1);
  cv::Mat exptype = cv::Mat::zeros(_marker0.rows, _marker0.cols, CV_8UC1); // 확장도중 만난 edge type?

  for( int y = 0; y < size.height; y++ ) {
    for(int  x = 0; x < size.width; x++ ) {
      const int32_t& m0 = _marker0.at<int32_t>(y,x);
      if(m0<1)
        continue;
      marker_radius[m0] = std::max(marker_radius[m0], boundary_distance.at<float>(y,x) );
    }
  }

  for( int y = 0; y < size.height; y++ ) {
    for(int  x = 0; x < size.width; x++ ) {
      if(_marker.at<int32_t>(y,x)>0)
        continue;
      // 근처에 valid marker가 있으면 현재위치를 추가해야함.
      if(x > 0){
        const int32_t& m = _marker.at<int32_t>(y,x-1);
        const uchar& e   =   dd_edges.at<uchar>(y,x-1);
        const float& z   =      depth.at<float>(y,x-1);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(x < w1){
        const int32_t& m = _marker.at<int32_t>(y,x+1);
        const uchar& e   =   dd_edges.at<uchar>(y,x+1);
        const float& z   =      depth.at<float>(y,x+1);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(y > 0){
        const int32_t& m = _marker.at<int32_t>(y-1,x);
        const uchar& e   =   dd_edges.at<uchar>(y-1,x);
        const float& z   =      depth.at<float>(y-1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
      if(y < h1){
        const int32_t& m = _marker.at<int32_t>(y+1,x);
        const uchar& e   =   dd_edges.at<uchar>(y+1,x);
        const float& z   =      depth.at<float>(y+1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m, z, e) );
      }
    } // for y
  } // for x

  const uchar FG = OutlineEdgeDetectorWithSizelimit::DdType::FG;
  const uchar BG = OutlineEdgeDetectorWithSizelimit::DdType::BG;

  /*
    * exptype에서 서로 다른 marker이 bg edge node끼리 많이 만나는 경우 -> merge pair.
    * exptype에서 fg-bg edge가 만나는 경우..
      - (가느다란 fg)를 지운 상태에서, 인접한 bg edge node들을 확장한 결과, (latest bg depth차가 작은것끼리)많이 만나는 경우 -> merge pair.
  */
  std::map<std::pair<int,int>, size_t> bgbg_contacts; // key : min(bg_m),max(bg_m)
  std::map<std::pair<int,int>, size_t> fgbg_contacts; // key : thin fg_m, bg_m
  float thin_th = 10.;

  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    uchar& e = exptype.at<uchar>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    e = k.e;
    cost = k.cost;
    m = k.m; // 주변 노드중, marker가 정해지지 않은 노드를 candidate에 추가
    if(k.e == FG) // contact counting을 위해 fg pixel에 marker만 맵핑하고 확장은 중단.
      continue;
    if(x > 0){
      const int32_t& m2 = _marker.at<int32_t>(y,x-1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x-1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x-1); // edge를 넘어서부턴 z update를 중단.
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x-1,y, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          //if(m2==56 && k.m==45) throw -1;
          q2.push(next); // m2 지우고 나서 확장을 다시시도.
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(x < w1 ){
      const int32_t& m2 = _marker.at<int32_t>(y,x+1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x+1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x+1); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x+1,y, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          //if(m2==56 && k.m==45) throw -1;
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(y > 0 && _marker.at<int32_t>(y-1,x)<1 ){
      const int32_t& m2 = _marker.at<int32_t>(y-1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y-1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y-1,x); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x,y-1, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
    if(y < h1){
      const int32_t& m2 = _marker.at<int32_t>(y+1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y+1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y+1,x); 
      const uchar e = k.e > 0 ? k.e : e2;
      Node next(x,y+1, k.cost+1., k.m, z, e);
      if(m2 < 1)
        q1.push( next );
      else if(m2 != k.m){
        if(e2==FG && k.e==BG && marker_radius[m2] < thin_th){
          q2.push(next);
          fgbg_contacts[std::pair<int,int>(m2, k.m)]++;
        }
        else if(e2==BG && k.e==BG)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
      }
    }
  } // while(!q1.empty())

  bgbg_contacts.clear(); // TODO remove
  std::map<int, std::set<int> > bg_neighbors; {
    std::map<int, std::list<int> > _bg_neighbors;
    for(auto it : fgbg_contacts){
      if(it.second > 20){
        printf("fgbg %d, %d - %ld\n", it.first.first, it.first.second, it.second);
        _bg_neighbors[it.first.first].push_back(it.first.second);
      }
    }
    for(auto it : _bg_neighbors){
      if(it.second.size()< 1)
        continue;
      for(auto it_key : it.second){
        for(auto it_n : it.second){
          if(it_key==it_n)
            continue;
          bg_neighbors[it_key].insert(it_n);
        }
      }
    }
  }


  {
    cv::Mat dst = GetColoredLabel(_marker,true);
    cv::imshow("marker1", dst);
  }
  for(auto it : marker_radius){
    if(it.second < 10.){
      cv::Mat mask = _marker==it.first;
      _marker.setTo(0, mask);
      costs.setTo(999., mask);
    }
  }

  const float max_zerr = 100.; // TODO remove
  while(!q2.empty()){
    Node k = q2.top(); q2.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    uchar& e = exptype.at<uchar>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    //e = k.e;
    cost = k.cost;
    m = k.m; // 주변 노드중, marker가 정해지지 않은 노드를 candidate에 추가

    if(x > 0){
      const int32_t& m2 = _marker.at<int32_t>(y,x-1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x-1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x-1);
      Node next(x-1,y, k.cost+1., k.m, k.z, e); // 더이상 update 중단.
      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(x < w1){
      const int32_t& m2 = _marker.at<int32_t>(y,x+1);
      const uchar& e2   =  dd_edges.at<uchar>(y,x+1);
      const float z = e > 0 ? k.e : depth.at<float>(y,x+1);
      Node next(x+1,y, k.cost+1., k.m, k.z, e);

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(y > 0){
      const int32_t& m2 = _marker.at<int32_t>(y-1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y-1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y-1,x);
      Node next(x,y-1, k.cost+1., k.m, k.z, e); // 더이상 update 중단.

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
    if(y < h1){
      const int32_t& m2 = _marker.at<int32_t>(y+1,x);
      const uchar& e2   =  dd_edges.at<uchar>(y+1,x);
      const float z = e > 0 ? k.e : depth.at<float>(y+1,x);
      Node next(x,y+1, k.cost+1., k.m, k.z, e);

      if(m2 <1)
        q2.push( next );
      else if(m2 != k.m && bg_neighbors[k.m].count(m2) )
        if( std::abs(z-k.z) < max_zerr)
          bgbg_contacts[std::pair<int,int>(std::min(m2,k.m), std::max(m2,k.m))]++;
    }
  }

  for(auto it : bgbg_contacts)
    if(it.second > 20)
      printf("bgbg %d, %d - %ld\n", it.first.first, it.first.second, it.second);

  std::cout << "-------------" << std::endl;



  {
    cv::Mat dst = GetColoredLabel(_marker,true);
    dst.setTo(CV_RGB(0,0,0), GetBoundary(_marker));
    cv::imshow("marker2", dst);
  }
  //cv::imshow("dist_boundary", 0.01*boundary_distance);

  return _marker0;
}

cv::Mat TotalSegmentor::GetEdge(const cv::Mat depth, const cv::Mat invmap,
                                const uchar FG, const uchar BG, const uchar CE,
                                float dd_so, float ce_so, float min_obj_width, float hessian_th) const {
  std::vector<cv::Point2i> dd_samples = {
    cv::Point2i(dd_so,0),
    cv::Point2i(0,dd_so),
  };

  cv::Mat edge = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt0(c0,r0);
      for(const auto& dpt : dd_samples){
        const cv::Point2i pt1 = pt0-dpt;
        const cv::Point2i pt2 = pt0+dpt;
        if(pt1.x<0 || pt1.y<0)
          continue;
        if(pt2.x>=depth.cols || pt2.y>=depth.rows)
          continue;
        const float& invz0 = depth.at<float>(pt0);
        const float& invz1 = depth.at<float>(pt1);
        const float& invz2 = depth.at<float>(pt2);
        float diff = invz2 - invz1;
        if(std::abs(diff) < 0.1 * invz0)
          continue;
        if(diff < 0){
          edge.at<uchar>(pt1) = BG;
          edge.at<uchar>(pt2) = FG;
        }
        else {
          edge.at<uchar>(pt1) = FG;
          edge.at<uchar>(pt2) = BG;
        }
      }
    } // for c0
  } // for r0, dd_edges

  // 2. Compute Concave edges
  cv::Mat dist_dd;
  cv::distanceTransform(edge < 1,dist_dd, cv::DIST_L2, cv::DIST_MASK_3, CV_32F);
  const float fmin_objwidht = fx_ * min_obj_width;
  for(int r0 = 0; r0 < depth.rows; r0++) {
    for(int c0 = 0; c0 < depth.cols; c0++) {
      const cv::Point2i pt0(c0,r0);
      if(dist_dd.at<float>(pt0) < 1.+ce_so)
        continue;
      const float& z0 = depth.at<float>(pt0);
      const float& invz0 = invmap.at<float>(pt0);
      uchar& e = edge.at<uchar>(pt0);
      const cv::Point2i pt_y0(pt0.x, pt0.y-ce_so);
      const cv::Point2i pt_y1(pt0.x, pt0.y+ce_so);
      if( pt_y0.y > 0 && pt_y1.y < depth.rows){
        const float dy = ce_so/fy_*z0;
        const float gy0 = (z0-depth.at<float>(pt_y0)) / dy;
        const float gy1 = (depth.at<float>(pt_y1)-z0) / dy;
        if( (gy1-gy0)/dy < hessian_th)
          e = CE;
      }
      if(e)
        continue;
      if(dist_dd.at<float>(pt0) < fmin_objwidht * invz0)
        continue; // 수직선 모양의 concave edge는 dd edge에서 떨어진 부분만 확인.
      //수평선 모양 concave edge는 가느다란 막대기 instance때문에 이렇게 접근할 수 없다.

      const cv::Point2i pt_x0(pt0.x-ce_so, pt0.y);
      const cv::Point2i pt_x1(pt0.x+ce_so, pt0.y);
      if( pt_x0.x > 0 && pt_x1.x < depth.cols){
        const float dx = ce_so/fx_*z0;
        const float gx0 = (z0-depth.at<float>(pt_x0)) / dx;
        const float gx1 = (depth.at<float>(pt_x1)-z0) / dx;
        if( (gx1-gx0)/dx < hessian_th)
          e = CE;
      }
    }
  }

  return edge;
}

int TotalSegmentor::Dijkstra(cv::Mat& _marker) const {
  const int L_EDGE = -1;
  int l_max = 0;
  const cv::Size size = _marker.size();
  const int w1 = size.width-1;
  const int h1 = size.height-1;
  const float sqrt2 = std::sqrt(2.);
  struct Node {
    float cost;
    int x,y; // 아직 marker가 정해지지 않은 grid의 x,y
    int32_t m;   // 부모 marker.
    Node(int _x, int _y, float _cost, int32_t _m) : x(_x), y(_y), cost(_cost), m(_m) {
    }
    bool operator < (const Node& other) const {
      return cost > other.cost;
    }
  };

  std::priority_queue<Node> q1;
  cv::Mat costs   = 999.*cv::Mat::ones(_marker.rows, _marker.cols, CV_32FC1);
  for( int y = 0; y < size.height; y++ ) {
    for(int  x = 0; x < size.width; x++ ) {
      const int m0 = _marker.at<int32_t>(y,x);
      if(m0 != 0){ // -1:= edge, >0:=labeled marker
        continue;
        l_max = std::max(l_max, m0);
      }
      // 근처에 valid marker가 있으면 현재위치를 추가
      if(x > 0){
        const int32_t& m = _marker.at<int32_t>(y,x-1);
        if(m > 0)
          q1.push( Node(x,y, 1., m) );
      }
      if(x < w1){
        const int32_t& m = _marker.at<int32_t>(y,x+1);
        if(m > 0)
          q1.push( Node(x,y, 1., m) );
      }
      if(y > 0){
        const int32_t& m = _marker.at<int32_t>(y-1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m) );
      }
      if(y < h1){
        const int32_t& m = _marker.at<int32_t>(y+1,x);
        if(m > 0)
          q1.push( Node(x,y, 1., m) );
      }
    }
  }

  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    cost = k.cost;
    m = k.m;
    if(x > 0 && _marker.at<int32_t>(y,x-1)==0)
      q1.push( Node(x-1,y,k.cost+1.,k.m) );
    if(x < w1 && _marker.at<int32_t>(y,x+1)==0)
      q1.push( Node(x+1,y,k.cost+1.,k.m) );
    if(y > 0 && _marker.at<int32_t>(y-1,x)==0)
      q1.push( Node(x,y-1,k.cost+1.,k.m) );
    if(y < h1 && _marker.at<int32_t>(y+1,x)==0)
      q1.push( Node(x,y+1,k.cost+1.,k.m) );
    if(x > 0 && y > 0 && _marker.at<int32_t>(y-1,x-1)==0)
      q1.push( Node(x-1,y-1,k.cost+sqrt2,k.m) );
    if(x < w1 && y > 0 && _marker.at<int32_t>(y-1,x+1)==0)
      q1.push( Node(x+1,y-1,k.cost+sqrt2,k.m) );
    if(x > 0 && y < h1 && _marker.at<int32_t>(y+1,x-1)==0)
      q1.push( Node(x-1,y+1,k.cost+sqrt2,k.m) );
    if(x < w1 && y < h1 && _marker.at<int32_t>(y+1,x+1)==0)
      q1.push( Node(x+1,y+1,k.cost+sqrt2,k.m) );
  } // while(!q1.empty())

  return l_max;
}


void TotalSegmentor::Merge(const cv::Mat& _depth,
                           const cv::Mat& _edge,
                           cv::Mat& _marker, float min_direct_contact_ratio_for_merge, bool keep_boundary) const {
  #define pairkey(i1, i2){ \
    std::make_pair(std::min(i1,i2),std::max(i1,i2))\
  }
  cv::Mat given_edge, given_depth; 
  cv::resize( _edge, given_edge, _marker.size(), 0, 0, cv::INTER_NEAREST);
  cv::resize( _depth, given_depth, _marker.size(), 0, 0, cv::INTER_NEAREST);
  cv::Mat expanded_edge = cv::Mat::zeros(given_edge.size(), given_edge.type());
  cv::Mat dist_edge;
  cv::distanceTransform(given_edge<1, dist_edge, cv::DIST_L2, cv::DIST_MASK_3, CV_32F);

  const cv::Size size = _marker.size();
  const int w1 = size.width-1;
  const int h1 = size.height-1;
  const float sqrt2 = std::sqrt(2.);
  struct Node {
    float cost;
    int x,y; // 아직 marker가 정해지지 않은 grid의 x,y
    int32_t m;   // 부모 marker.
    uchar e;
    Node(int _x, int _y, float _cost, int32_t _m, uchar _e) : x(_x), y(_y), cost(_cost), m(_m), e(_e) {
    }
    bool operator < (const Node& other) const {
      return cost > other.cost;
    }
  };

  std::map<int, float> marker_inradius;
  std::map<int, std::list<float> > marker_inradiuses;
  std::map<int, std::list<float> > marker_depths;
  std::map< std::pair<int, int>, size_t > contact_before_rm_edge;
  int* marker = _marker.ptr<int>();
  const int mstep = int(_marker.step/sizeof(marker[0]));
  const float sfx = float(given_edge.cols)/float(_edge.cols) * fx_;
  for(int y = 0; y < size.height; y++ ) {
    for(int x = 0; x < size.width; x++ ) {
      int* m = marker+x;
      if( m[0] < 0)
        continue;
      else if( m[0] > 0){
        float& radius = marker_inradius[m[0]];
        radius = std::max(radius, dist_edge.at<float>(y,x) );
        marker_inradiuses[m[0]].push_back(dist_edge.at<float>(y,x) );
        marker_depths[m[0]].push_back( given_depth.at<float>(y,x) );
        // 자신과 다른 이웃 label숫자 세기.
        if(x < w1 && m[1] > 0 && *m != m[1])
          contact_before_rm_edge[ pairkey(*m, m[1]) ]++;
        else if(y < h1 && m[mstep] > 0 && *m != m[mstep])
          contact_before_rm_edge[ pairkey(*m, m[mstep]) ]++;
      }
    }
    marker += mstep;
  }

  std::priority_queue<Node> q1;
  cv::Mat costs = 999.*cv::Mat::ones(_marker.rows, _marker.cols, CV_32FC1);
  cv::Mat marker_before_extension = keep_boundary? _marker.clone() : cv::Mat();
  _marker.setTo(0, _marker < 0); 
  std::map< std::pair<int, int>, size_t > contact_after_rm_edge;
  marker = _marker.ptr<int>();
  for(int y = 0; y < size.height; y++ ) {
    for(int x = 0; x < size.width; x++ ) {
      int* m = marker+x;
      if( m[0] != 0 )
        continue;
      const uchar& e = given_edge.at<uchar>(y,x);
      if(x > 0 && m[-1] > 0 )
          q1.push( Node(x,y,0.,m[-1],e) );
      if(x < w1 && m[1] > 0)
        q1.push( Node(x,y,0.,m[1],e) );
      if(y > 0 && m[-mstep] > 0)
        q1.push( Node(x,y,0.,m[-mstep],e) );
      if(y < h1 && m[mstep] > 0)
        q1.push( Node(x,y,0.,m[mstep],e) );
    }
    marker += mstep;
  }

  // q1 : m==0 픽셀에 위차한 node의 집합.
  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    const int& x = k.x;
    const int& y = k.y;
    float& cost = costs.at<float>(k.y,k.x);
    int32_t& m = _marker.at<int32_t>(k.y,k.x);
    if(cost <= k.cost) // less 'eq' : 같은 cost에서 무한루프 발생하지 않게.
      continue;
    cost = k.cost;
    m = k.m;
    expanded_edge.at<uchar>(y,x) = k.e;
    if(x > 0 && _marker.at<int32_t>(y,x-1)==0){
      const uchar& e = given_edge.at<uchar>(y,x-1);
      q1.push( Node(x-1,y,k.cost+1.,k.m, k.e?k.e:e) );
    }
    if(x < w1 && _marker.at<int32_t>(y,x+1)==0){
      const uchar& e = given_edge.at<uchar>(y,x+1);
      q1.push( Node(x+1,y,k.cost+1.,k.m, k.e?k.e:e) );
    }
    if(y > 0 && _marker.at<int32_t>(y-1,x)==0){
      const uchar& e = given_edge.at<uchar>(y-1,x);
      q1.push( Node(x,y-1,k.cost+1.,k.m, k.e?k.e:e) );
    }
    if(y < h1 && _marker.at<int32_t>(y+1,x)==0){
      const uchar& e = given_edge.at<uchar>(y+1,x);
      q1.push( Node(x,y+1,k.cost+1.,k.m,k.e?k.e:e) );
    }
    if(x > 0 && y > 0 && _marker.at<int32_t>(y-1,x-1)==0){
      const uchar& e = given_edge.at<uchar>(y-1,x-1);
      q1.push( Node(x-1,y-1,k.cost+sqrt2,k.m,k.e?k.e:e) );
    }
    if(x < w1 && y > 0 && _marker.at<int32_t>(y-1,x+1)==0){
      const uchar& e = given_edge.at<uchar>(y-1,x+1);
      q1.push( Node(x+1,y-1,k.cost+sqrt2,k.m,k.e?k.e:e) );
    }
    if(x > 0 && y < h1 && _marker.at<int32_t>(y+1,x-1)==0){
      const uchar& e = given_edge.at<uchar>(y+1,x-1);
      q1.push( Node(x-1,y+1,k.cost+sqrt2,k.m,k.e?k.e:e) );
    }
    if(x < w1 && y < h1 && _marker.at<int32_t>(y+1,x+1)==0){
      const uchar& e = given_edge.at<uchar>(y+1,x+1);
      q1.push( Node(x+1,y+1,k.cost+sqrt2,k.m,k.e?k.e:e) );
    }
  } // while(!q1.empty())

  std::map<int, float> marker_phy_inradius;
  for(auto it : marker_inradius){
    const int& m = it.first;
    auto& it_depth = marker_depths[m];
    int n = it_depth.size();
    if(n < 2)
      continue;
    it_depth.sort();
    auto it2 = it_depth.begin();
    for(int i =0; i<n/2; i++)
      it2++;
    const float z = *it2;
    marker_phy_inradius[it.first] = z * it.second / sfx;
  }
  /*
  for(auto it : marker_inradiuses){
    const int& m = it.first;
    float z; {
      auto& it_depth = marker_depths[m];
      int n = it_depth.size();
      if(n < 2)
        continue;
      it_depth.sort();
      auto it2 = it_depth.begin();
      for(int i =0; i<n/2; i++)
        it2++;
      z = *it2;
    }
    float r; {
      int n = it.second.size();
      if(n < 2)
        continue;
      it.second.sort();
      auto it2 = it.second.begin();
      for(int i =0; i<int(.7*n); i++)
        it2++;
      r = *it2;
    }
    marker_phy_inradius[it.first] = z * r / sfx;
  }
  */

  cv::Mat bgmarker = _marker.clone();
  for(int y = 0; y < size.height; y++ ) {
    for(int x = 0; x < size.width; x++ ) {
      int32_t& m = bgmarker.at<int32_t>(y,x);
      if(m<1)
        continue;
      if( marker_phy_inradius[m] > .4) // [meter] TODO 파라미터화.
        continue;
      //if( marker_inradius[m] > 10.)
      //  continue;
      // TODO 가느다란 instance를 지운다음, occlusion이 발생한 주변 node를 q2에 추가.
      m = -1;
    }
  }

  {
    cv::imshow("given_edge", 255*(given_edge<1));
    cv::imshow("dist_edge", .05*dist_edge);
    cv::Mat dst_marker = GetColoredLabel(bgmarker);
    dst_marker.setTo(CV_RGB(255,255,255), GetBoundary(_marker));
    cv::imshow("expended_marker", dst_marker);
  }
  {
    cv::Mat dst_edge = cv::Mat::zeros(expanded_edge.size(), CV_8UC3);
    dst_edge.setTo(CV_RGB(255,0,0), expanded_edge==1); // FG
    dst_edge.setTo(CV_RGB(0,0,255), expanded_edge==2);
    dst_edge.setTo(CV_RGB(0,255,0), expanded_edge==3);
    cv::imshow("expanded edge", dst_edge);
  }

  #undef pairkey
  return;
}

void TotalSegmentor::Put(cv::Mat depth) {
  const uchar FG = 1; // foreground
  const uchar BG = 2; // background
  const uchar CE = 3; // concave edge
  const float dd_so = 1; // dd sample offset

  const float ce_so = 8.; // concave edges sample offset.
  const float min_obj_width = .5; // [meter]
  const float hessian_th = -20.;

  const float maxz = 1e+10;
  cv::Mat invmap = 1. / depth; {
    invmap.setTo(1./maxz, depth >= maxz);
  }

  // 1. Compute DD edges
  cv::Mat edge = GetEdge(depth,invmap,FG,BG,CE,dd_so,ce_so,min_obj_width,hessian_th);
  edge = FilterThinNoise(edge);

  cv::Mat edge32S;
  edge.convertTo(edge32S, CV_32SC1);
  /*
    2. Semgentation
    outline->segment는 대략적으로 segmentation하는거니까 기존 방식대로 edge type을 binary로 놓고 수행.
    TODO) 단, Dijkstra(이전에 BFS로 잘못처리)로 정정하면서, 이상한 under segmentation 해결.
  */
  cv::Mat outline = edge > 0;

  const int n_octave = 6;
  const int n_downsample = 1;
  const bool keep_boundary = true;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  std::vector<cv::Mat> pyr_edges, pyr_markers;
  pyr_edges.reserve(n_octave);
  pyr_edges.push_back(outline);
  while(pyr_edges.size() < n_octave){
    cv::Mat prev = *pyr_edges.rbegin();
    cv::Mat next;
    cv::resize( prev, next, cv::Size(int(prev.cols/2), int(prev.rows/2)), 0, 0);
    cv::dilate(next, next, kernel);
    pyr_edges.push_back(next>0);
  }
  std::reverse(pyr_edges.begin(), pyr_edges.end());

  std::vector<cv::Size> sizes;
  for(auto it : pyr_edges)
    sizes.push_back(it.size());

  cv::Mat label_curr;
  const int Lv = pyr_edges.size()-n_downsample;
  for(int lv=0; lv < Lv; lv++){
    const bool final_lv = lv+2 == Lv;
    const bool merge = final_lv; // 마지막 pyr 에서 merge 수행.
    cv::Mat edge_curr = pyr_edges.at(lv);
    if(lv==0)
      cv::connectedComponents(~edge_curr,label_curr,4, CV_32SC1); // else. get from prev.
    const cv::Mat& edge_next = pyr_edges.at(lv+1);
    cv::resize(label_curr, label_curr, edge_next.size(), 0, 0, cv::INTER_NEAREST);
    cv::resize(edge_curr, edge_curr, edge_next.size(),   0, 0, cv::INTER_NEAREST); // marker가 thin edge침범하는걸 막기위해서인데, 중복계산우려.
    label_curr.setTo( 0, edge_curr);
    label_curr.setTo(-1, edge_next);

    int l_max = Dijkstra(label_curr);
    cv::Mat zero_label = label_curr == 0;
    cv::Mat new_labels;
    cv::connectedComponents( zero_label, new_labels,4, CV_32SC1); // else. get from prev.
    new_labels += l_max;
    new_labels.setTo(0, ~zero_label);
    label_curr += new_labels;
    if(final_lv){
      cv::Mat boundary = GetBoundary(label_curr);
      label_curr.setTo(-1, edge_curr);
      label_curr.setTo(-1, boundary);
      cv::connectedComponents(label_curr>0,label_curr,4, CV_32SC1); // else. get from prev.

    }
    

    if(merge){
      // TODO 여기서 edge type을 참고해서 fg,bg관계 정리에 필요한 정보도 수집.
      // downsample일 경우, 병목지점의 직접접촉에 비해, edge가 상대적으로 두꺼운거 감안.
      Merge(depth, edge, label_curr, n_downsample > 0 ? .4 : .7, keep_boundary);
      //Merge(label_curr, .7, keep_boundary); // overseg 유발. 
    }
  }

  {
    cv::Mat dst = GetColoredLabel(label_curr,true)/2;
    dst.setTo(CV_RGB(255,0,0), edge==FG);
    dst.setTo(CV_RGB(0,0,255), edge==BG);
    dst.setTo(CV_RGB(120,120,120), edge==CE);
    cv::imshow("dst", dst);
  }
  /*
  // cv::imshow("depth0", .01*depth);
  // cv::imshow("depth1", .05*depth);
  {
    cv::Mat dst = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC3);
    dst.setTo(CV_RGB(255,0,0), edge==FG);
    dst.setTo(CV_RGB(0,0,255), edge==BG);
    dst.setTo(CV_RGB(120,120,120), edge==CE);
    cv::imshow("edge", dst);
  }
  */
  return;
}
