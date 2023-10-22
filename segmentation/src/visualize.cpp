#include "segslam.h"
#include "util.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace NEW_SEG {

/*
void PutInstanceInfoBox(const float& switch_threshold,
                        const std::map<Pth, float>& switch_states,
                        const std::map<Pth,float>& density_scores,
                        const std::map<Pth, std::pair<cv::Point2f, float> >& pth2center,
                        cv::Mat& dst){
  for(auto it : pth2center){
    const Pth& pth = it.first;
    const cv::Point2f& cp = it.second.first;
    std::string msg[3];
    cv::Scalar txt_colors[3];
    msg[0] = "#" + std::to_string(pth);
    for(int k=0; k<3;k++)
      txt_colors[k] = CV_RGB(0,0,0);
    if(switch_states.count(pth) ){
      std::ostringstream oss; oss << std::fixed << std::setprecision(2);
      oss<< switch_states.at(pth);
      msg[2] += oss.str();
      if(switch_states.at(pth) < switch_threshold)
        txt_colors[2] = CV_RGB(255,0,0);
    }
    else
      msg[2] += "-";
    int offset = 2;
    int w, h;
    w = h = 0;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .3; int fontThick = 1; int baseline = 0;
    for(int k=0; k<3; k++){
      auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
      h += size.height;
      w = std::max(size.width, w);
    }
    cv::Point2f dpt(.5*w, .5*h);
    cv::rectangle(dst, cp-dpt-cv::Point2f(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
    int x = cp.x - .5*w;
    int y = cp.y - .5*h;
    for(int k=0;k<3;k++){
      cv::putText(dst, msg[k], cv::Point(x,y),fontFace, fontScale, txt_colors[k], fontThick);
      auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
      y += size.height+offset;
    }
  }
  return;
}
*/


void Pipeline::Visualize(const cv::Mat rgb, const cv::Mat gt_dynamic_mask, cv::Mat& dst) {
  Qth qth = 0;
  RigidGroup* rig = qth2rig_groups_.at(qth);
  cv::Mat dst_frame = rgb.clone();
  std::map<Pth, std::pair<cv::Point2f, float> > pth2center;

  cv::Mat outline = vinfo_synced_marker_ < 1;
  cv::Mat dist_from_outline; {
    cv::Mat bb = ~outline;
    for (int i = 0; i < bb.cols; i++) {
      bb.at<uchar>(0, i) = bb.at<uchar>(bb.rows - 1, i) = 0;
    }
    for (int i = 0; i < bb.rows; i++) {
      bb.at<uchar>(i, 0) = bb.at<uchar>(i, bb.cols - 1) = 0;
    }
    cv::distanceTransform(bb,dist_from_outline, cv::DIST_L2, cv::DIST_MASK_3);
  }

  std::set<Pth> excluded;
  for(auto it : rig->GetExcludedInstances())
    excluded.insert(it.first);
  dst_frame.setTo(CV_RGB(0,0,0), outline);
  for(Pth pth : excluded)
    dst_frame.setTo(CV_RGB(255,0,0), vinfo_synced_marker_==pth);
  for(Pth pth : vinfo_outlier_detected_){
    if(excluded.count(pth))
      continue;
    dst_frame.setTo(CV_RGB(255,255,0), vinfo_synced_marker_==pth);
  }

  cv::addWeighted(rgb, .5, dst_frame, .5, 1., dst_frame);
  dst_frame.setTo(CV_RGB(0,0,0), outline>0);

  const auto& keypoints = prev_frame_->GetKeypoints();
  const auto& mappoints = prev_frame_->GetMappoints();
  const auto& instances = prev_frame_->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(mp)
      continue;
    cv::circle(dst_frame, pt, 3, CV_RGB(150,150,150), 1);
  }

  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(!mp)
      continue;
    const bool& supplied_pt = vinfo_supplied_mappoints_[n];
    if(supplied_pt){
      Instance* ins = instances[n];
      cv::Scalar color = CV_RGB(0,0,255);
      if(ins && ins->GetQth() != 0)
        color = CV_RGB(255,0,0);
      cv::circle(dst_frame, pt, 2, color, 1);
    }
  }
  std::list<Mappoint*> sorted_mappoints;
  for(size_t n=0; n<keypoints.size(); n++){
    cv::Point2f pt = keypoints[n].pt;
    Mappoint* mp = mappoints[n];
    if(!mp)
      continue;
    const bool& supplied_pt = vinfo_supplied_mappoints_[n];
    if(supplied_pt)
      continue;
    Instance* ins = instances[n];
    //Instance* ins = mp->GetInstance();
    sorted_mappoints.push_back(mp);
    bool is_dynamic = ins && ins->GetQth() != 0;
    cv::Scalar color = is_dynamic? CV_RGB(255,0,0) : CV_RGB(0,0,255);
    cv::circle(dst_frame, pt, 3, color, -1);
    std::list<Frame*> keyframes;
    for(Frame* kf : mp->GetKeyframes(qth))
      keyframes.push_back(kf);
    keyframes.sort([](Frame* a, Frame* b) { return a->GetId() < b->GetId(); } );
    int n_pt =0;
    for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
      Frame* kf = *it;
      int index = kf->GetIndex(mp);
      const cv::Point2f& pt_next = kf->GetKeypoint( index ).pt;
      cv::line(dst_frame, pt, pt_next, color, 1);
      //if(++n_pt > 1)
        break;
      //if(!is_dynamic)
      // break;
    }
  }

  sorted_mappoints.sort([](Mappoint* a, Mappoint* b) {
                        Pth pa = a->GetInstance()->GetId();
                        Pth pb = b->GetInstance()->GetId();
                        if(pa != pb)
                          return pa < pb;
                        return a->GetId() < b->GetId();
                        });

  // Draw patches.
  cv::Size patch_size(20,20);
  int n_rows = 5;
  int n_cols = 10;
  cv::Mat dst_patches = cv::Mat::zeros(patch_size.height*n_rows, patch_size.width*n_cols, CV_8UC3);
  dst_patches.setTo(CV_RGB(200,0,255));
  {
    static std::map<Jth, cv::Mat> gt_masks;
    gt_masks[prev_frame_->GetId()] = gt_dynamic_mask;
    int i_col = 0;
    for(Mappoint* mp : sorted_mappoints){
      Frame* ref = mp->GetRefFrame(0);
      cv::Point2f pt_ref = ref->GetKeypoint(ref->GetIndex(mp)).pt;
      if(!gt_masks[ref->GetId()].at<uchar>(pt_ref))
        continue;
      Pth pth = mp->GetInstance()->GetId();
      //if(mp->GetInstance()->GetId() != 101)
      //  continue;
      std::map<int, Frame*> sorted_keyframes;
      for(Frame* kf : mp->GetKeyframes(qth))
        sorted_keyframes[kf->GetId()] = kf;
      sorted_keyframes[prev_frame_->GetId()] = prev_frame_;
      int i_row = 0;
      for(auto it_kf : sorted_keyframes){
        Frame* kf = it_kf.second;
        const cv::Point2f& pt = kf->GetKeypoint( kf->GetIndex(mp) ).pt;
        const cv::Mat& rgb = kf->GetRgb();
        if(rgb.empty())
          continue;
        int x = static_cast<int>(pt.x - int(patch_size.width/2));
        int y = static_cast<int>(pt.y - int(patch_size.height/2));
        if (x < 0 || y < 0 || x + patch_size.width >= rgb.cols || y + patch_size.height >= rgb.rows) 
          continue;
        cv::Rect rgb_roi(x, y, patch_size.width, patch_size.height);
        cv::Rect dst_roi(patch_size.width*i_col, patch_size.height*i_row,patch_size.width, patch_size.height);
        rgb(rgb_roi).copyTo(dst_patches(dst_roi));
        {
          cv::Point pt(patch_size.width*i_col, patch_size.height*i_row);
          cv::putText(dst_patches, std::to_string(kf->GetId()%100),pt, cv::FONT_HERSHEY_SIMPLEX, .4, CV_RGB(200,0,255) );
        }

        if(i_row==0){
          cv::Point pt(patch_size.width*i_col, dst_patches.rows-2);
          cv::putText(dst_patches, std::to_string(pth%100),pt, cv::FONT_HERSHEY_SIMPLEX, .5, CV_RGB(0,0,0) );
        }

        if(++i_row >= n_rows)
          break;
      }
      if(++i_col >= n_cols)
        break;
    }
  }
  
  std::map<Jth, Frame*> keyframes = vinfo_neighbor_frames_[qth];
  keyframes[prev_frame_->GetId()] = prev_frame_;

  if(vinfo_match_filter_.empty())
    vinfo_match_filter_ = cv::Mat::zeros(dst_frame.rows, dst_frame.cols, CV_8UC3);
  if(vinfo_dynamic_detects_.empty())
    vinfo_dynamic_detects_ = cv::Mat::zeros(dst_frame.rows, dst_frame.cols, CV_8UC3);

  {
    int fid = prev_frame_->GetId();
    Frame* lkf = keyframes_.at(qth).rbegin()->second;
    std::string msg;
    msg += "Outlier matches F#" + std::to_string(fid);
    msg += prev_frame_->IsKeyframe() ? " is KF, " : " is not KF, ";
    msg += " / KF# " + std::to_string(lkf->GetKfId(qth));
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .6; int fontThick = 1; int baseline = 0;
    cv::Size size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
    cv::rectangle(vinfo_match_filter_, cv::Point(0,0), cv::Point(size.width+10,size.height+10), CV_RGB(150,150,150), -1);
    cv::putText(vinfo_match_filter_, msg, cv::Point(5,size.height+5),fontFace, fontScale, CV_RGB(255,0,0), fontThick);
  }

  {
    int fid = prev_frame_->GetId();
    Frame* lkf = keyframes_.at(qth).rbegin()->second;
    std::string msg;
    msg += "Dynamic detection F#" + std::to_string(fid);
    msg += prev_frame_->IsKeyframe() ? " is KF, " : " is not KF, ";
    msg += " / KF# " + std::to_string(lkf->GetKfId(qth));
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .6; int fontThick = 1; int baseline = 0;
    cv::Size size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
    cv::rectangle(vinfo_dynamic_detects_, cv::Point(0,0), cv::Point(size.width+10,size.height+10), CV_RGB(150,150,150), -1);
    cv::putText(vinfo_dynamic_detects_, msg, cv::Point(5,size.height+5),fontFace, fontScale, CV_RGB(255,0,0), fontThick);
  }

  cv::vconcat(vinfo_match_filter_, vinfo_dynamic_detects_, dst);
  cv::vconcat(dst, dst_frame, dst);
  cv::resize(dst, dst, cv::Size(900,900) );

  //cv::imshow("dst", dst);
  cv::imshow("dpatches", dst_patches);
  cv::moveWindow("dpatches", 1500, 100);
  return;
} // Pipeline::Visualize

} // namespace NEW_SEG
