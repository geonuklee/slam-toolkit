#include "dataset.h"
#include "orb_extractor.h"
#include "frame.h"
#include "camera.h"
#include "pybind11/pytypes.h"
#include <exception>
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>

#if 1
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

class HITNetStereoMatching{
public:
  HITNetStereoMatching(float base_line, float focal_length)
  {
    int max_distance = 50;
    std::string pkg_dir = PACKAGE_DIR;
    //py::exec("import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" );
    std::string pycode;
    pycode += "import sys; sys.path.append(\"" + pkg_dir + "\")\n";
    pycode += "from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig\n";
    pycode += "model_path=\"" + pkg_dir + "/models/eth3d.pb\"\n";
    pycode += "model_type=ModelType.eth3d\n";
    pycode += "camera_config=CameraConfig("+ std::to_string(base_line)+"," +std::to_string(focal_length)+")\n";
    pycode += "max_distance="+std::to_string(max_distance)+"\n";
    pycode += "hitnet=HitNet(model_path, model_type, camera_config)\n";
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << pycode << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    py::exec(pycode);
    hitnet_ = py::eval("hitnet");
  }

  cv::Mat Put(cv::Mat gray, cv::Mat gray_r){
    py::array_t<uint8_t> _im = cvMat2array<uchar>(gray);
    py::array_t<uint8_t> _im_r = cvMat2array<uchar>(gray_r);
    hitnet_.attr("__call__")(_im,_im_r); // python> disparity_map = hitnet(im,im_r)
    py::object _depthmap = hitnet_.attr("get_depth")();
    return array2cvMat<float>(_depthmap);
  }

private:
  py::object hitnet_;
};

#else
cv::Mat GetDisparity(cv::cuda::GpuMat g_gray, cv::cuda::GpuMat g_gray_r){
  cv::cuda::GpuMat  g_disp;
  cv::Mat disparity;
  /*
  {
    static auto bm = cv::cuda::createStereoBM();
    bm->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disp); // 8UC
  }
  */
  if(false){
    static auto sbp = cv::cuda::createStereoBeliefPropagation();
    sbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  else{
    int ndisp=128;
    int iters=4;
    int levels=4;
    int nr_plane=4;
    static auto csbp = cv::cuda::createStereoConstantSpaceBP(ndisp,iters,levels,nr_plane);
    csbp->compute(g_gray, g_gray_r, g_disp);
    g_disp.download(disparity); // 32FC1
  }
  disparity.convertTo(disparity, CV_32FC1);
  return disparity;
}
#endif


int TestKitti(int argc, char** argv) {
  std::string seq(argv[1]);
  KittiDataset dataset(seq);

  const auto& Tcws = dataset.GetTcws();
  if(Tcws.empty()){
    std::cout << "Seq" << seq << " with no ground truth trajectory." << std::endl;
  }
  const auto& D = dataset.GetCamera()->GetD();
  std::cout << "Distortion = " << D.transpose() << std::endl;
  const StereoCamera* camera = dynamic_cast<const StereoCamera*>(dataset.GetCamera());
  assert(camera);
  const auto Trl_ = camera->GetTrl();
  const float base_line = -Trl_.translation().x();
  const float fx = camera->GetK()(0,0);

  py::scoped_interpreter python; // 이 인스턴스가 파괴되면 인터프리터 종료.
  HITNetStereoMatching hitnet(base_line, fx);

  bool stop = false;
  for(int i=0; i<dataset.Size(); i+=1){
    cv::Mat rgb   = dataset.GetImage(i, cv::IMREAD_COLOR);
    cv::Mat rgb_r = dataset.GetRightImage(i, cv::IMREAD_COLOR);

    cv::Mat gray, gray_r;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb_r, gray_r, cv::COLOR_BGR2GRAY);
 
    // small input to reserve time.
    cv::pyrDown(gray, gray);
    cv::pyrDown(gray_r, gray_r);

    auto t0 = std::chrono::steady_clock::now();
    cv::Mat depth = hitnet.Put(gray, gray_r);
    depth *= 2.; // Restore scale for pyrDown.
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "etime = " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "[msec]" << std::endl;

    cv::resize(depth, depth, rgb.size());

    cv::Mat ndisp = 0.01*depth;
    //cv::normalize(depth, ndisp, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("depth", ndisp);
    cv::imshow("rgb", rgb);

    char c = cv::waitKey(stop?0:1);
    if(c == 'q')
      break;
    else if (c == 's')
      stop = !stop;
  }

  std::cout << "Done. The end of the sequence" << std::endl;
  return 1;
}

int main(int argc, char** argv){
  if(argc < 1){
    std::cout << "No arguments for dataset name." << std::endl;
    exit(-1);
  }
  return TestKitti(argc, argv);
}
