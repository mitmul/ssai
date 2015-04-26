Semantic Segmentation of Aerial Imagery
========================================

Extract building and road from aerial imagery

# Requirements

- OpenCV 2.4.10
- Boost 1.57.0
- Boost.NumPy
- Caffe (modified caffe: https://github.com/mitmul/caffe)
    - NOTE: Build the `ssai` branch of the above repository

# Data preparation

    $ bash shells/donwload.sh
    $ python scripts/create_dataset.py --dataset multi
    $ python scripts/create_dataset.py --dataset single
    $ python scripts/create_dataset.py --dataset roads_mini
    $ python scripts/create_dataset.py --dataset roads
    $ python scripts/create_dataset.py --dataset buildings
    $ python scripts/create_dataset.py --dataset merged

## Massatusetts Building & Road dataset

- train: 3849568 patches
- valid: 36100 patches
- test: 89968 patches

# Create Models

    $ python scripts/create_models.py --seed seeds/model_seeds.json

# Start training

    $ bash shells/train.sh models/Mnih_CNN

will create a directory named `results/Mnih_CNN_{started date}`.

# Prediction

    $ cd results/Mnih_CNN_{started date}
    $ python ../../scripts/test_prediction.py --model predict.prototxt --weight snapshots/Mnih_CNN_iter_1000000.caffemodel --img_dir ../../data/mass_merged/test/sat --channel 3

# Build Library for Evaluation

    $ cd lib
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

# Evaluation

    $ cd results/Mnih_CNN_{started date}
    $ python ../../scripts/test_evaluation.py --map_dir ../../data/mass_merged/test/map --result_dir prediction_1000000 --channel 3
