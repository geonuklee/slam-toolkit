#ifndef HITNET_
#define HITNET_
#include <memory>
#include <opencv2/opencv.hpp>

namespace pybind11{
  class object;
}

class HITNetStereoMatching{
public:
  HITNetStereoMatching();
  cv::Mat GetDisparity(cv::Mat gray, cv::Mat gray_r);
private:
  std::shared_ptr<pybind11::object> hitnet_;
};

cv::Mat Disparit2Depth(cv::Mat disp,
                       const float base_line,
                       const float fx,
                       const float min_disp);

#endif

