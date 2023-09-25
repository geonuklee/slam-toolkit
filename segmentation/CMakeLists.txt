# Copyright (c) 2023 Geonuk Lee
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.1)
project(segmentation)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0072 OLD)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake_modules)

#find_package(OpenCV 3.4 REQUIRED PATHS $ENV{HOME}/ws/opencv_3.4.14/build)
find_package(OpenCV 3.4.20 REQUIRED
  PATHS $ENV{HOME}/ws/opencv_3.4.20_cuda10.2/build
  NO_DEFAULT_PATH)

include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )

# After compile, install g2o from source
# Add below command at bashrc
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS system filesystem regex)
include_directories(${Boost_INCLUDE_DIRS})

find_package(FLANN REQUIRED)
include_directories(${FLANN_INCLUDE_DIR})

find_package(Pangolin REQUIRED EXACT 0.8
  PATHS ${CMAKE_SOURCE_DIR}/Pangolin-0.8/build
  NO_DEFAULT_PATH
)
include_directories(${Pangolin_LIBRARIES})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g") # Debug info for perf profile.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # Forbid it with direct method.

add_definitions(-DPACKAGE_DIR="${CMAKE_SOURCE_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/..")
include_directories("${CMAKE_SOURCE_DIR}/../pipeline/include")


file(GLOB src src/*.cpp)
add_executable(example_segmentation ${src}
  ../pipeline/src/camera.cpp
  ../pipeline/src/common.cpp
  ../pipeline/src/dataset.cpp
  ../pipeline/src/orb_extractor.cpp
)

# If use HITNET for stereo dataset such as KITTI..
set(PYBIND11_PYTHON_VERSION 3.8)
find_package(PythonLibs ${PYBIND11_PYTHON_VERSION} EXACT REQUIRED)
find_package(pybind11 REQUIRED PATHS
  $ENV{HOME}/.local/lib/python${PYBIND11_PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11
  NO_DEFAULT_PATH)
target_include_directories(example_segmentation PRIVATE
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/../stereo_seg/include)
target_link_libraries(example_segmentation ${CMAKE_SOURCE_DIR}/../stereo_seg/lib/libhitnetstereo.so
  ${PYTHON_LIBRARIES}
)

set_source_files_properties(src/edge.cpp PROPERTIES COMPILE_FLAGS "-mavx2") # No optmization

#set_target_properties(example_segmentation PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 10)
#set_source_files_properties(src/debug.cpp PROPERTIES COMPILE_FLAGS "-O0 -g") # No optmization
#set_source_files_properties(src/optimizer.cpp PROPERTIES COMPILE_FLAGS "-O0 -g") # No optmization
set_source_files_properties(src/segslam_new.cpp PROPERTIES COMPILE_FLAGS "-O0 -g") # No optmization
set_source_files_properties(src/visualize.cpp PROPERTIES COMPILE_FLAGS "-O0 -g") # No optmization

set_source_files_properties(src/seg_viewer.cpp PROPERTIES COMPILE_FLAGS "-O3") # Too slow for non optimization.
set_source_files_properties(../pipeline/src/common.cpp PROPERTIES COMPILE_FLAGS "-O3") # Too slow for non optimization.

target_link_libraries(example_segmentation
${OpenCV_LIBS}
${Boost_LIBRARIES}
${FLANN_LIBRARY}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../thirdparty/DBoW2/lib/libDBoW2.so
optimized g2o_core
optimized g2o_stuff
optimized g2o_types_sba
#optimized g2o_types_slam3d

debug g2o_core_d
debug g2o_stuff_d
debug g2o_types_sba_d
#optimized g2o_types_slam3d_d
stdc++fs
)


