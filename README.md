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

it will create
- train: 3849568 patches
    - mean: [73.5661, 82.3104, 80.6461]
    - stddev: [48.524406, 46.994726, 49.018684]
- valid: 123904 patches
- test: 309654 patches

# Start training

    $ bash shells/train.sh Multi_Plain_Mnih_NN_S_ReLU

will create a directory named `results/Multi_Plain_Mnih_NN_S_ReLU_{started date}`.

# Prediction

    $ cd results/Multi_Plain_Mnih_NN_S_ReLU_{started date}
    $ python ../../scripts/test_prediction.py --model predict.prototxt --weight snapshots/Multi_Plain_Mnih_NN_S_ReLU_iter_250000.caffemodel

