#! /bin/bash
fn=`date +"%Y-%m-%d_%I-%M-%S"`
if [ ! -d results ]; then
    mkdir results
fi
cd results
fpath=$1
modelname="${fpath##*/}"
echo $modelname
dname=$modelname'_'$fn
echo $dname
mkdir $dname
cd $dname
mkdir snapshots
cp ../../models/$modelname/*.prototxt ./
caffe_dir=$HOME/Libraries/caffe/build/install
$caffe_dir/python/draw_net.py train_test.prototxt net.png
# export GLOG_log_dir=$PWD
echo 'start learning' $1
nohup $caffe_dir/bin/caffe train \
    -solver=$PWD/solver.prototxt &
