cmake_minimum_required(VERSION 2.6)
project(ssai)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")

# add_definitions(-DUSE_MKL)
set(inc_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include
    /usr/local/include/opencv
    /usr/local/include/opencv2
    /usr/local/cuda/include
    /opt/intel/mkl/include
    $ENV{HOME}/lib/caffe/build/install/include
    $ENV{HOME}/lib/boost_1_58_0/install/include
    $ENV{HOME}/lib/opencv-2.4.10/install/include
    $ENV{HOME}/lib/opencv-2.4.10/install/include/opencv
    $ENV{HOME}/lib/opencv-2.4.10/install/include/opencv2
    $ENV{HOME}/lib/Boost.NumPy/install/include
    $ENV{HOME}/anaconda/include/python2.7
)
set(lib_dirs
    /usr/local/lib
    $ENV{HOME}/lib/caffe/build/install/lib
    $ENV{HOME}/anaconda/lib
    $ENV{HOME}/lib/opencv-2.4.10/install/lib
    $ENV{HOME}/lib/Boost.NumPy/install/lib
    $ENV{HOME}/lib/boost_1_58_0/install/lib
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
