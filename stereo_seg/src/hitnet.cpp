#include "hitnet.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T>
cv::Mat GetCvMat(long rows, long cols, long dim);

template<>
cv::Mat GetCvMat<unsigned char>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_8UC3);
  return cv::Mat(rows, cols, CV_8UC1);
}

template<>
cv::Mat GetCvMat<float>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_32FC3);
  return cv::Mat(rows, cols, CV_32FC1);
}

template<>
cv::Mat GetCvMat<int32_t>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_32SC3);
  return cv::Mat(rows, cols, CV_32SC1);
}

template<typename T>
cv::Mat array2cvMat(py::array_t<T> array){
  // TODO Without copy
  cv::Mat mat;
  py::buffer_info buf_info = array.request();
  long rows = buf_info.shape[0];
  long cols = buf_info.shape[1];
  const T* ptr = (const T*) buf_info.ptr;
  long dim = buf_info.shape.size()==2 ? 1 : buf_info.shape[2];
  mat = GetCvMat<T>(rows, cols, dim);
  memcpy(mat.data, ptr, mat.total() * sizeof(T));
  return mat;
}

template<typename T>
py::array_t<T> cvMat2array(cv::Mat mat){
  // without copy memory
  size_t rows = mat.rows;
  size_t cols = mat.cols;
  size_t dims = mat.channels() > 1 ? 3 : 2;
  std::vector<size_t> shape;
  if(dims > 2)
    shape = std::vector{rows, cols , (size_t)mat.channels()};
  else
    shape = std::vector{rows, cols};
  // ref for stride : https://stackoverflow.com/questions/72702026/pybind11-cvmat-from-c-to-python
  std::vector<size_t> stride;
  if(dims > 2)
    stride = { sizeof(T) * cols * mat.channels(), sizeof(T) * mat.channels(), sizeof(T)};
  else
    stride = { sizeof(T) * cols, sizeof(T)};
  py::buffer_info info(mat.data, sizeof(T), py::format_descriptor<T>::format(),
                       shape.size(), shape, stride);
  return py::array_t<T>(info);
}


HITNetStereoMatching::HITNetStereoMatching() {
  std::string pkg_dir = PACKAGE_DIR;
  //py::exec("import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" );
  std::string pycode;
  pycode += "import sys; sys.path.append(\"" + pkg_dir + "\")\n";
  pycode += "from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig\n";
  pycode += "model_path=\"" + pkg_dir + "/models/eth3d.pb\"\n";
  pycode += "model_type=ModelType.eth3d\n";
#if 1
  pycode += "hitnet=HitNet(model_path, model_type)\n";
#else
  float base_line = 1.;
  float focal_length = 1.;
  pycode += "camera_config=CameraConfig("+ std::to_string(base_line)+"," +std::to_string(focal_length)+")\n";
  pycode += "hitnet=HitNet(model_path, model_type, camera_config)\n";
#endif
  // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  // std::cout << pycode << std::endl;
  // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  py::exec(pycode);
  hitnet_ = std::make_shared<py::object>(py::eval("hitnet"));
}

cv::Mat HITNetStereoMatching::GetDisparity(cv::Mat gray, cv::Mat gray_r) {
  py::array_t<uint8_t> _im = cvMat2array<uchar>(gray);
  py::array_t<uint8_t> _im_r = cvMat2array<uchar>(gray_r);
  py::object _disparity = hitnet_->attr("__call__")(_im,_im_r); // python> disparity_map = hitnet(im,im_r)
  return array2cvMat<float>(_disparity);
  //py::object _depthmap = hitnet_->attr("get_depth")();
  //return array2cvMat<float>(_depthmap);
}

cv::Mat Disparit2Depth(cv::Mat disp,
                       const float base_line,
                       const float fx,
                       const float min_disp) {
  cv::Mat depth;
  cv::Mat mask = disp < min_disp;
  disp.setTo(1., mask);
  depth = base_line*fx / disp;
  return depth;
}


