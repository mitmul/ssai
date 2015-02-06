cmake_minimum_required(VERSION 2.6)
project(ssai)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-Wall -O3")

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
)
include_directories(
    ${inc_dirs}
)
link_directories(
    ${lib_dirs}
)

add_executable(compute_image_mean
    src/compute_image_mean.cpp
)
target_link_libraries(compute_image_mean
    ${libs}
)
