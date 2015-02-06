#! /bin/bash
fn=`date +"%Y-%m-%d_%I-%M-%S"`
if [ ! -d results ]; then
    mkdir results
fi
cd results
mkdir $1_$fn
cd $1_$fn
cp ../../models/$1/*.prototxt ./
mkdir snapshots
caffe_dir=$HOME/Libraries/caffe
$caffe_dir/python/draw_net.py train_test.prototxt net.png
export GLOG_log_dir=$PWD
echo 'start learning' $1_$fn
nohup $caffe_dir/build/tools/caffe train \
    -solver=../../models/$1/solver.prototxt &