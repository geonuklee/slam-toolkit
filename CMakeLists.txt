# Copyright (c) 2020 Geonuk Lee
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
project(pipeline)
cmake_policy(SET CMP0042 NEW)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

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

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt5 COMPONENTS REQUIRED Core Gui Widgets)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fdiagnostics-color=always")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # Forbid it with direct method.

add_definitions(-DPACKAGE_DIR="${CMAKE_SOURCE_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}")

file(GLOB src src/*.cpp)
add_library(pipeline ${src})
set_target_properties(pipeline PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 5)

target_link_libraries(pipeline 
${OpenCV_LIBS}
${Boost_LIBRARIES}
${FLANN_LIBRARY}
${VTK_LIBRARIES}
Qt5::Core
Qt5::Gui 
Qt5::Widgets
${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib/libDBoW2.so
optimized g2o_core
optimized g2o_stuff
optimized g2o_types_sba
debug g2o_core_d
debug g2o_stuff_d
debug g2o_types_sba_d
)


#file(GLOB unit_src unittest/*.cpp)
#add_executable(unittest ${unit_src})
#set_target_properties(unittest PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 5)
#target_link_libraries(unittest pipeline ${GTEST_LIBRARIES})
#add_dependencies(unittest pipeline)

add_subdirectory(examples/kitti)
#add_subdirectory(examples/epip_cluster)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
