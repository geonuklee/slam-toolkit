#include "segslam.h"
#include "util.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#if 0
cv::Mat Pipeline::VisualizeRigInfos(Frame* frame,
                                    const Qth& qth,
                                    const std::map<Jth, Frame* >& neighbor_frames,
                                    const std::set<Mappoint*>& neighbor_mappoints,
                                    std::set<Pth> given_instances,
                                    const float& switch_threshold,
                                    const std::map<Pth,float>& switch_states,
                                    const std::map<Pth,ShapePtr>& curr_shapes
                                   ) const {
  const cv::Mat rgb = frame->GetRgb();
  RigidGroup* rig = qth2rig_groups_.at(qth);
  cv::Mat dst_frame; {
    cv::Mat dst_fill  = rgb.clone();
    std::set<Pth> excluded;
    for(auto it : rig->GetExcludedInstances())
      excluded.insert(it.first);
    for(auto it : switch_states){
      if(it.second > switch_threshold)
        continue;
      excluded.insert(it.first);
    }
    for(auto pth : excluded)
      if(given_instances.count(pth))
        given_instances.erase(pth);

    for(Pth pth : given_instances){
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(0,255,0), 3);
    }
    for(Pth pth : excluded){
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(255,0,0), -1);
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(0,255,0), 3);
    }
    cv::addWeighted(rgb, .5, dst_fill, .5, 1., dst_frame);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; double fontScale = .3; int fontThick = 1; int baseline = 0;
    for(Pth pth : given_instances){
      if(!curr_shapes.count(pth))
        continue;
      auto msg = std::to_string(pth);
      const auto ptr = curr_shapes.at(pth);
      const auto& bb = ptr->outerior_bb_;
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      auto size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
      cv::Point dpt(.5*size.width, .5*size.height);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      cv::putText(dst_frame, msg, cp-dpt, fontFace, fontScale, CV_RGB(0,0,0));
    }
    for(Pth pth : excluded){
      if(!curr_shapes.count(pth))
        continue;
      auto msg = std::to_string(pth);
      const auto ptr = curr_shapes.at(pth);
      const auto& bb = ptr->outerior_bb_;
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      auto size = cv::getTextSize(msg, fontFace, fontScale, fontThick, &baseline);
      cv::Point dpt(.5*size.width, .5*size.height);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);
      cv::putText(dst_frame, msg, cp-dpt, fontFace, fontScale, CV_RGB(0,0,0));
    }
    for(Mappoint* mp : neighbor_mappoints){
      int n = frame->GetIndex(mp);
      if(n < 0)
        continue;
      cv::KeyPoint kpt = frame->GetKeypoint(n);
      cv::circle(dst_frame, kpt.pt, 3, CV_RGB(0,255,0) );
    }
  }
  cv::pyrDown(dst_frame, dst_frame);

  cv::Mat dst_texts = cv::Mat::zeros( dst_frame.rows, 400, CV_8UC3); {
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .4;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    /* Neighbor frame을 여기에 나열. */
    std::map<Jth, Frame*> keyframes = neighbor_frames;
    keyframes[frame->GetId()] = frame;
    for(auto it_kf : keyframes){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      auto t = kf->GetTcq(qth).inverse().translation();
      char text[100];
      sprintf(text,"est : F#%2d, KF#%2d, (%4.3f, %4.3f, %4.3f)", kf->GetId(), kf->GetKfId(qth), t[0], t[1], t[2] );
      auto size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }

    y+= 10;
    std::ostringstream buffer;
    buffer << "Include> ";
    for(auto it : rig->GetIncludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 3;
    buffer.str("");
    buffer << "Exclude> ";
    for(auto it : rig->GetExcludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 10;
    buffer.str("");
    buffer << "Pth:Switch state> ";
    for(auto it : switch_states){
      buffer << "(" << it.first <<  ":" << std::fixed << std::setprecision(3) << it.second << "), ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 100)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
  }

  cv::Mat dst = cv::Mat::zeros(std::max<int>(dst_frame.rows, dst_texts.rows),
                               dst_frame.cols + dst_texts.cols,
                               CV_8UC3);

  {
    cv::Rect rect(0, 0, dst_frame.cols, dst_frame.rows);
    cv::Mat roi(dst, rect);
    dst_frame.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  {
    cv::Rect rect(dst_frame.cols,0,
                  dst_texts.cols, dst_texts.rows);
    cv::Mat roi(dst, rect);
    dst_texts.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  return dst;
}


cv::Mat Pipeline::VisualizeStates(Frame* frame,
                        const std::map<Pth,float>& density_scores,
                        const std::map<Pth, float>& switch_states,
                        const float& switch_threshold,
                        const std::map<Jth, Frame* >& neighbor_frames,
                        const cv::Mat& synced_marker,
                        const EigenMap<int,g2o::SE3Quat>* gt_Tcws
                        ) const {
  const Qth dominant_qth = 0;
  const RigidGroup* dominant_rig = qth2rig_groups_.at(dominant_qth);
  const cv::Mat rgb = frame->GetRgb();
  cv::Mat dst_frame; {
    cv::Mat dst_fill  = rgb.clone();
    std::set<Pth> excluded;
    for(auto it : dominant_rig->GetExcludedInstances())
      excluded.insert(it.first);
    for(auto it : switch_states){
      if(it.second > switch_threshold)
        continue;
      excluded.insert(it.first);
    }

    for(auto it : excluded){
      const Pth& pth = it;
      if(!curr_shapes.count(pth))
        continue;
      ShapePtr s_ptr = curr_shapes.at(pth);
      std::vector< std::vector<cv::Point> > cnts;
      cnts.resize(1);
      cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
      cv::drawContours(dst_fill, cnts, 0, CV_RGB(255,0,0), -1);
    }
    cv::addWeighted(rgb, .5, dst_fill, .5, 1., dst_frame);
    cv::Mat dst_outline;
    cv::cvtColor(outline_mask, dst_outline, cv::COLOR_GRAY2BGR);
    cv::addWeighted(dst_frame, 1., dst_outline, .3, 1., dst_frame);
  }

  for(auto it : curr_shapes) {
    const Pth& pth = it.first;
    //bool sparse = density_scores.count(pth) ? density_scores.at(pth) < 1. : true;
    //bool dynamics = switch_states.count(pth) ? switch_states.at(pth) > switch_threshold : false;
    ShapePtr s_ptr = it.second;
    if(s_ptr->n_missing_ > 0)
      continue;
    if(s_ptr->n_belief_ < 2)
      continue;
    std::vector< std::vector<cv::Point> > cnts;
    cnts.resize(1);
    cnts[0].reserve(s_ptr->outerior_.size() );
      for( auto pt: s_ptr->outerior_)
        cnts[0].push_back(cv::Point(pt.x,pt.y));
    const auto& color = colors[pth % colors.size()];
    cv::drawContours(dst_frame, cnts, 0, color, 2);
  }

  const auto& keypoints = frame->GetKeypoints();
  const auto& mappoints = frame->GetMappoints();
  const auto& instances = frame->GetInstances();
  for(size_t n=0; n<keypoints.size(); n++){
    Instance* ins = instances[n];
    Mappoint* mp = mappoints[n];
    const cv::Point2f& pt = keypoints[n].pt;
    if(mp){
      std::stringstream ss;
      ss << std::hex << mp->GetId(); // Convert to hexadecimal
      std::string msg = ss.str();
      if(mp->HasEstimate4Rig(dominant_qth)) {
        const std::set<Frame*>& keyframes = mp->GetKeyframes(dominant_qth);
        for(auto it = keyframes.rbegin(); it!=keyframes.rend(); it++){
          const cv::Point2f& pt0 = (*it)->GetKeypoint( (*it)->GetIndex(mp) ).pt;
          cv::circle(dst_frame, pt, 3, CV_RGB(0,255,0), -1);
          cv::line(dst_frame, pt, pt0, CV_RGB(255,255,0),  1);
          break;
        }
      }
      else
        cv::circle(dst_frame, pt, 3, CV_RGB(255, -1,0), -1);
    }
    else {
      cv::circle(dst_frame, pt, 2, CV_RGB(150,150,150), 1);
    }
  }
  {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .3;
    int fontThick = 1;
    int baseline = 0;
    for(auto it : curr_shapes){
      const Pth& pth = it.first;
      ShapePtr ptr = it.second;
      const auto& bb = ptr->outerior_bb_;
      std::string msg[3];
      cv::Scalar txt_colors[3];
      msg[0] = "#" + std::to_string(pth);
      for(int k=0; k<3;k++)
        txt_colors[k] = CV_RGB(0,0,0);
      msg[1] = "d: "; msg[2] = "s: ";
      if(density_scores.count(pth) ){
        std::ostringstream oss; oss << std::fixed << std::setprecision(2);
        oss<< density_scores.at(pth);
        msg[1] += oss.str();
        if(density_scores.at(pth) < 1.)
          txt_colors[1] = CV_RGB(255,0,0);
      }
      else
        msg[1] += "-";
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
      for(int k=0; k<3; k++){
        auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
        h += size.height;
        w = std::max(size.width, w);
      }
      cv::Point cp(bb.x+.5*bb.width, bb.y+.5*bb.height);
      cv::Point dpt(.5*w, .5*h);
      cv::rectangle(dst_frame, cp-dpt-cv::Point(0,3*baseline), cp+dpt, CV_RGB(255,255,255), -1);

      int x = cp.x - .5*w;
      int y = cp.y - .5*h;
      for(int k=0;k<3;k++){
        cv::putText(dst_frame, msg[k], cv::Point(x,y),fontFace, fontScale, txt_colors[k], fontThick);
        auto size = cv::getTextSize(msg[k], fontFace, fontScale, fontThick, &baseline);
        y += size.height+offset;
      }

    }
  }

  cv::Mat dst_texts = cv::Mat::zeros( rgb.rows, 400, CV_8UC3); {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = .4;
    int thickness = 1;
    int baseline=0;
    int y = 0;
    cv::rectangle(dst_texts,cv::Rect(0,0,dst_texts.cols, dst_texts.rows),CV_RGB(255,255,255),-1);
    std::map<Jth, Frame*> keyframes = neighbor_frames;
    keyframes[frame->GetId()] = frame;
    Frame* prev_kf = nullptr;
    for(auto it_kf : keyframes){
      const Jth& jth = it_kf.first;
      Frame* kf = it_kf.second;
      auto t = kf->GetTcq(dominant_qth).inverse().translation();
      char text[100];

      sprintf(text,"est : F#%2d, KF#%2d  (%4.3f, %4.3f, %4.3f)", kf->GetId(), kf->GetKfId(dominant_qth), t[0], t[1], t[2] );
      auto size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);

      if(!gt_Tcws)
        continue;
      const g2o::SE3Quat& Tcw = gt_Tcws->at(jth);
      t = Tcw.inverse().translation();
      sprintf(text,"true: F#%2d,  (%4.3f, %4.3f, %4.3f)", kf->GetId(), t[0], t[1], t[2] );
      size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, text, cv::Point(5, y), fontFace, fontScale, CV_RGB(120,120,120), thickness);

      if(!prev_kf){
        prev_kf = kf;
        continue;
      }
      // dT =Tc0 c1 = Tc0 w * Tc1 w.inverse()
      g2o::SE3Quat est_dT = prev_kf->GetTcq(dominant_qth) * kf->GetTcq(dominant_qth).inverse();
      g2o::SE3Quat true_dT = gt_Tcws->at(prev_kf->GetId()) * gt_Tcws->at(kf->GetId()).inverse();
      g2o::SE3Quat err_dT = true_dT.inverse() * est_dT;
      if(true_dT.translation().norm() < 1e-2)
        continue;
      float norm_ratio = est_dT.translation().norm() / true_dT.translation().norm();
      float err_ratio = err_dT.translation().norm() / true_dT.translation().norm();
      sprintf(text,"norm ratio: %4.3f", norm_ratio);
      cv::putText(dst_texts, text, cv::Point(20+size.width, y), fontFace, fontScale, CV_RGB(255,0,0), thickness);

      y += 2;
      prev_kf = kf;
    }

    y+= 10;
    std::ostringstream buffer;
    buffer << "Include> ";
    for(auto it : dominant_rig->GetIncludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
    y+= 3;
    buffer.str("");
    buffer << "Exclude> ";
    for(auto it : dominant_rig->GetExcludedInstances()){
      const Pth& pth = it.first;
      if(!curr_shapes.count(pth))
        continue;
      buffer << pth << ", ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 10)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }

    y+= 10;
    buffer.str("");
    buffer << "Pth:Switch state> ";
    for(auto it : switch_states){
      buffer << "(" << it.first <<  ":" << std::fixed << std::setprecision(3) << it.second << "), ";
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      if(size.width < dst_texts.cols - 100)
        continue;
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
      buffer.str("");
    }
    if(!buffer.str().empty()){
      auto size = cv::getTextSize(buffer.str(), fontFace, fontScale, thickness, &baseline);
      y += size.height+2;
      cv::putText(dst_texts, buffer.str(), cv::Point(5, y), fontFace, fontScale, CV_RGB(0,0,0), thickness);
    }
  }

  cv::Mat dst = cv::Mat::zeros(std::max<int>(dst_frame.rows, dst_texts.rows),
                               dst_frame.cols + dst_texts.cols,
                               CV_8UC3);

  {
    cv::Rect rect(0, 0, dst_frame.cols, dst_frame.rows);
    cv::Mat roi(dst, rect);
    dst_frame.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }
  {
    cv::Rect rect(dst_frame.cols,0,
                  dst_texts.cols, dst_texts.rows);
    cv::Mat roi(dst, rect);
    dst_texts.copyTo(roi);
    cv::rectangle(dst, rect, CV_RGB(0,0,0), 2);
  }

  return dst;
}
#endif

namespace seg {

void Pipeline::Visualize() {
  cv::Mat rgb = vinfo_vis_rgb_;
  cv::Mat dst_frame = vinfo_vis_rgb_.clone();
  cv::Mat outline = vinfo_synced_marker_ < 1;
  //cv::Mat expanded_outline;
  //cv::distanceTransform(~outline,expanded_outline, cv::DIST_L2, cv::DIST_MASK_3);
  //expanded_outline = expanded_outline  < 10.;
  Qth dominant_qth = 0;
  if(prev_frame_->GetId() == 0){
    // 첫번째 frame이다.
    return;
  }

  Qth qth = dominant_qth;
  const auto& switch_states = vinfo_switch_states_.at(qth);
  for(size_t i = 0; i < dst_frame.rows; i++){
    for(size_t j = 0; j < dst_frame.cols; j++){
      cv::Vec3b& pixel = dst_frame.at<cv::Vec3b>(i,j);
      if(outline.at<uchar>(i,j)){
        pixel[0] = pixel[1] =pixel[2] = 0;
        continue;
      }
      
      const Pth& pth = vinfo_synced_marker_.at<int32_t>(i,j);
      bool other_rig = false;
      if(switch_states.count(pth) ){
        other_rig = switch_states.at(pth) < switch_threshold_;
      }
      else
        other_rig = true;
      if(other_rig){
        pixel[0] = pixel[1] = 0;
        pixel[2] = 255;
      }
      //const auto& color = colors[pth % colors.size() ];
      //pixel[0] = color[0];
      //pixel[1] = color[1];
      //pixel[2] = color[2];
    }
  }
  cv::addWeighted(rgb, .5, dst_frame, .5, 1., dst_frame);

  cv::imshow("dst", dst_frame);
  cv::imshow("sync_marker", GetColoredLabel(vinfo_synced_marker_));
  return;
}


}
