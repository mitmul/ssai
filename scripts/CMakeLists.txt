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
    $ENV{HOME}/Libraries/boost_1_57_0/install/include
    $ENV{HOME}/Libraries/opencv-2.4.10/install/include
    $ENV{HOME}/Libraries/opencv-2.4.10/install/include/opencv
    $ENV{HOME}/Libraries/opencv-2.4.10/install/include/opencv2
    $ENV{HOME}/Libraries/Boost.NumPy/install/include
    $ENV{HOME}/anaconda/include
    $ENV{HOME}/anaconda/include/python2.7
)
set(lib_dirs
    $ENV{HOME}/anaconda/lib
    $ENV{HOME}/Libraries/caffe/build/lib
    $ENV{HOME}/Libraries/boost_1_57_0/install/lib
    $ENV{HOME}/Libraries/opencv-2.4.10/install/lib
    $ENV{HOME}/Libraries/Boost.NumPy/install/lib
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

add_executable(calc_mean_stddev
    src/calc_mean_stddev.cpp
)
target_link_libraries(calc_mean_stddev
    ${libs}
)
