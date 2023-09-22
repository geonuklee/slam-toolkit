#ifndef SEG_DATASET_H_
#define SEG_DATASET_H_
#include "stdafx.h"
#include "camera.h"

void WriteKittiTrajectory(const g2o::SE3Quat& Tcw, std::ofstream& output_file);

class KittiTrackingDataset {
public:
  KittiTrackingDataset(std::string type, //  "training"|"testing"
                       std::string seq,
                       std::string dataset_path="");
  virtual ~KittiTrackingDataset();

  cv::Mat GetImage(int i, int flags=cv::IMREAD_COLOR) const;
  cv::Mat GetRightImage(int i, int flags=cv::IMREAD_COLOR) const; 
  cv::Mat GetDepthImage(int i) const;
  cv::Mat GetDynamicMask(int i) const;
  cv::Mat GetInstanceMask(int i) const;

  bool EixstCachedDepthImages() const;
  void ComputeCacheImages(); // depth image, dynamic instance mask.

  virtual int Size() const;
  virtual const EigenMap<int, g2o::SE3Quat>& GetTcws() const;
  virtual const Camera* GetCamera() const;
private:
  std::string dataset_path_;
  std::string type_;
  std::string seq_;
  std::string depth_path_, mask_path_;

  std::map<int, std::string> im0_filenames_;
  std::map<int, std::string> im1_filenames_;
  EigenMap<int, g2o::SE3Quat> Tcws_;
  Camera* camera_; // TODO Steroe, Depth, etc..
};


#endif