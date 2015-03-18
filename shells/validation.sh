#! /bin/bash

cd results/Mnih_CNN_Asym_2015-03-17_05-36-05
echo $PWD
for ((i=10000; i <= 1000000; i+=10000)); do
    python ../../scripts/test_prediction.py \
    --model predict.prototxt \
    --weight snapshots/Mnih_CNN_Asym_iter_$i.caffemodel \
    --img_dir ../../data/mass_merged/valid/sat
done