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

find_package(OpenCV 3.4 REQUIRED PATHS $ENV{HOME}/ws/opencv_3.4.14/build)
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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fdiagnostics-color=always")
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
)

set_target_properties(example_segmentation PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 10)

target_link_libraries(example_segmentation
${OpenCV_LIBS}
${Boost_LIBRARIES}
${FLANN_LIBRARY}
${PROJECT_SOURCE_DIR}/../thirdparty/DBoW2/lib/libDBoW2.so
optimized g2o_core
optimized g2o_stuff
optimized g2o_types_sba
debug g2o_core_d
debug g2o_stuff_d
debug g2o_types_sba_d
)


