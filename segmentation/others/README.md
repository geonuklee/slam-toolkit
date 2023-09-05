# TUM dataset
```
svn checout https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools

```

# Installation

## 1. CUDA 10.2 + cuDNN7.6.5
* ref : https://medium.com/@anarmammadli/how-to-install-cuda-10-2-cudnn-7-6-5-and-samples-on-ubuntu-18-04-2493124478ca
```
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
#sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub # <필요한가?
#sudo apt-get update
#sudo apt-get -y install cuda-10-2

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# https://developer.nvidia.com/rdp/cudnn-archive 에서 cudnn v7.6.5 runtime, developer library 다운로드 후,
sudo dpkg -i libcudnn7*.deb

#  .zshrc 마지막에, 기존 cuda는 잠시 비활성화하고, cuda-10.2를 설정. opencv와 버전 충돌을 피해야한다.
#export PATH=/usr/local/cuda-11.1/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

## OpenCV with CUDA
* opencv 3.4.15 with CUDA 10.20

```bash
git clone https://github.com/opencv/opencv opencv_3.4.15_cuda10.2 # make install은 생략해, 기존 lib과 충돌은 피한다.
cd opencv_3.4.15_cuda10.2
git checkout 3.4.15
# optical flow DIP
git submodule add https://github.com/opencv/opencv_contrib
cd opencv_contrib; git checkout 3.4.15; cd ..

mkdir build; cd build;
# No cudacodec for https://github.com/opencv/opencv_contrib/issues/1786
# CUDA_LEGACY는 BroxOpticalFlow 때문에필요.
# 나머지 off는 빌드시간 절약용.
* [ ] WITH_DEBUGINFO
ccmake .. -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF\
  -DBUILD_opencv_dnn=ON -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DBUILD_opencv_python_bindings_g=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_opencv_stitching=OFF\
  -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_face=OFF  -DBUILD_opencv_objdetect=OFF \
  -DBUILD_opencv_cudalegacy=ON -DBUILD_opencv_xphoto=OFF -DBUILD_opencv_xobjdetect=OFF\
  -DENABLE_CXX11=ON \
  -DBUILD_opencv_cudacodec=ON \
  -D NVCUVID=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules
make -j4
```

## Pangolin
```bash
  sudo apt-get install cmake build-essential libgl1-mesa-dev libglew-dev
  cd Pangolin;
  git checkout v0.6
  cmake .. -DBUILD_PANGOLIN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j4 \
    && sudo make install
```

## DynaSLAM
```bash
  sudo apt-get -y install libboost-all-dev; \
  pip2 install --user keras
  cd Thirdpart/DBoW2
  # cmake에서 find_package(OpenCV 의 경로수정.
  mkdir build; cd build;
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && make -j4 
  cd ../../..

  cd Thirdpart/g2o
  mkdir build; cd build;
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && make -j4 
  cd ../../..
```

CMakeLists.txt에서 opencv 경로 바꾸고, 마지막 3줄 아래와같이 주석처리.
```
find_package(OpenCV 3.4.15 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.15_cuda10.2/build
  NO_DEFAULT_PATH)

#add_executable(mono_carla
#Examples/Monocular/mono_carla.cc)
#target_link_libraries(mono_carla ${PROJECT_NAME})
```
[다음](https://github.com/raulmur/ORB_SLAM2/pull/585#issuecomment-834243996)과 같이, include/LoopClosing.h 수정.
```
- Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;
+ Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3> > > KeyFrameAndPose;
```


## ClusterVO
* Repo를 공개하지않고, 소스코드만 공개된 application
  * [출처](https://huangjh-pub.github.io/)

CMakeLists.txt, lib/darknet/CMakeLists.txt 에서 opencv 경로 바꾸고,
```
find_package(OpenCV 3.4.15 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.15_cuda10.2/build
  NO_DEFAULT_PATH)
```


```bash
sudo apt install libgoogle-glog-dev libceres-dev libsuitesparse-dev
git clone https://github.com/strasdat/Sophus;
cd Sophus; mkdir build; cd build;
ccmake .. -DCMAKE_BUILD_TYPE=Release
make -j8sudo make install

git clone https://github.com/mpark/variant; cd variant
mkdir build; cd build; cmake ..; sudo make install

cd $ClusterVO
cd third_party/darknet
# nvim src/image_opencv.cpp +63
# ipl = m -> ipl = cvIplImage(m); // https://github.com/pjreddie/darknet/issues/1997

mkdir build; cd build;
ccmake .. -DOpenCV_DIR=$OPENCV_WITH_CUDA10.2_PATH -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17
make -j4

# For ubuntu 20.04, 
cd $ClusterVO
nvim ClusterVO/include/extractor.hpp # include <vector> 밑에 include <list> 추가
nvim ClusterVO/optimization/marginalization_factor.hpp #pragma once 밑에 #include<unordered_map> 추가

nvim ClusterVO/lib/exptools/CMakeLists.txt
# find_package(glog REQUIRED) -> 
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(glog REQUIRED libglog) # ref : https://github.com/google/glog/issues/519#issuecomment-1014378289

```


## StaticFusion
* [repo](https://github.com/raluca-scona/staticfusion)
CMakeLists.txt에서 opencv 경로 바꾸고,
```
find_package(OpenCV 3.4.15 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.15_cuda10.2/build
  NO_DEFAULT_PATH)
```

``` bash
sudo apt-get install cmake libmrpt-dev freeglut3-dev libglew-dev libopencv-dev libopenni2-dev git

mkdir build; cd build;
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 -DOpenCV_DIR=$ENV{HOME}/ws/opencv_3.2.0/build \
  && make -j4 
```

## VDO SLAM
CMakeLists.txt에서 opencv 경로 바꾸고,
```
find_package(OpenCV 3.4.15 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.15_cuda10.2/build
  NO_DEFAULT_PATH)
```

* VDO SLAM repo에서 빌드.
```bash
./build.sh
```

## supersurfel fusion
* Paper : Speed and memory efficient dense RGB-D SLAM in dynamic scenes
CMakeLists.txt에서 opencv 경로 바꾸고,
```
find_package(OpenCV 3.4.15 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.15_cuda10.2/build
  NO_DEFAULT_PATH)
```

* `core/src/motion_detection.cu,MotionDetection::detectMotionYoloOnly 제일 앞에 아래 변수선언 추가.
    * [ ] 정상작동여부 확인 필요.
'''cpp
void MotionDetection::detectMotionYoloOnly(const cv::Mat& rgb,
                                            const thrust::device_vector<SuperpixelRGBD>& superpixels,
                                            const cv::cuda::GpuMat& index_mat)
 {
    std::vector<std::vector<int>> adjacency_list_all; // TODO Geonuk Missing declare?
```


* third_party/darknet/CMakeLists.txt의 opencv 경로도 수정.
```bash
# Thirdpart build.
cd third_party/darknet
mkdir build
ccmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$OPENCV_WITH_CUDA10.2 -DENABLE_CUDNN=OFF
make -j8
sudo make install

# catkin build
cd ~/catkin_ws/src/
ln -s ${SUPERSURFEL_FUSION_REPO} .
cd ..

sudo apt-get install -y ros-melodic-pcl-ros
catkin build -j2 supersurfel_fusion --cmake-args -DCMAKE_BUILD_TYPE=Release 

```
