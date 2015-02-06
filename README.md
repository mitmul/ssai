Semantic Segmentation of Aerial Imagery
========================================

Extract building and road from aerial imagery

# Requirements
- OpenCV 2.4.10
- Caffe (modified caffe: https://github.com/mitmul/caffe)
    - NOTE: Build the `dev` branch of the above repository

# Data preparation

    $ bash shells/donwload.sh
    $ python scripts/create_dataset.py

# Start training

    $ bash shells/train.sh Multi_Plain_Mnih_NN_S_ReLU

will create a directory named `results/Multi_Plain_Mnih_NN_S_ReLU_{started date}`.

# Prediction

    $ cd results/Multi_Plain_Mnih_NN_S_ReLU_{started date}
    $ python ../../scripts/test_prediction.py --model predict.prototxt --weight snapshots/Multi_Plain_Mnih_NN_S_ReLU_iter_250000.caffemodel

