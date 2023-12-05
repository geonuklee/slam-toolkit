#include "../include/seg.h"
#include "../include/util.h"
#include <opencv2/highgui.hpp>


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

void Merge(cv::Mat& _marker, float min_direct_contact_ratio_for_merge, bool keep_boundary){
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

  cv::Mat marker_before_extension = keep_boundary?_marker.clone() : cv::Mat();

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

  if(keep_boundary)
    _marker = marker_before_extension; // outline edge를 남길 경우.

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
             bool keep_boundary,
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
    //cv::dilate(prev, next, kernel);
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

    if(merge){
      Merge(label_curr, n_downsample > 0 ? .4 : .7, keep_boundary); // downsample일 경우, 병목지점의 직접접촉에 비해, edge가 상대적으로 두꺼운거 감안.
      //Merge(label_curr, .7, keep_boundary); // overseg 유발. 
    }

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


}

