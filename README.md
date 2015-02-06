Semantic Segmentation of Aerial Imagery
========================================

Extract building and road from aerial imagery

# Requirements

    - OpenCV 2.4.10
    - Caffe (modified caffe: https://github.com/mitmul/caffe)
        - NOTE: Build the `dev` branch

# Data preparation

    $ bash shells/donwload.sh
    $ python scripts/create_dataset.py

# Start training

    $ bash shells/train.sh Multi_Plain_Mnih_NN_S_ReLU
    