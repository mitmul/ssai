cmake_minimum_required(VERSION 2.6)
project(ssai)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")

add_definitions(-DUSE_MKL)
set(inc_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include
    /usr/local/include/opencv
    /usr/local/include/opencv2
    /usr/local/cuda/include
    /opt/intel/mkl/include
    $ENV{HOME}/Libraries/caffe/build/src
    $ENV{HOME}/Libraries/caffe/include
    $ENV{HOME}/anaconda/include/python2.7
)
set(lib_dirs
    $ENV{HOME}/Libraries/caffe/build/lib
)
set(libs
    caffe
    glog
    leveldb
    lmdb
    gflags
    protobuf
    opencv_core
    opencv_highgui
    opencv_imgproc
    boost_system
    boost_filesystem
    boost_python
    boost_numpy
    python2.7
)
include_directories(
    ${inc_dirs}
)
link_directories(
    ${lib_dirs}
)

add_library(ssai SHARED
    src/ssai_tools.cpp
)
target_link_libraries(ssai
    ${libs}
)
set_target_properties(ssai PROPERTIES PREFIX "")