#! /bin/bash
cd $1
mkdir snapshots
caffe_dir=$HOME/Libraries/caffe
$caffe_dir/python/draw_net.py train_test.prototxt net.png
export GLOG_log_dir=$PWD
echo 'start learning' $1
nohup $caffe_dir/build/tools/caffe train \
    -solver=$PWD/solver.prototxt &