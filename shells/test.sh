# bash create_result_ReLU.sh [image_id] [iteration] [device_id]
if [ ! -d results/VGG_Roads_2015-10-06_01-27-32/prediction_150000 ]; then
  mkdir results/VGG_Roads_2015-10-06_01-27-32/prediction_150000
fi

function single_prediction() {
    nohup python scripts/create_result_origcaffe.py \
    --input_img data/test/sat/Google$1.png \
    --caffemodel results/VGG_Roads_2015-10-06_01-27-32/snapshots/VGG_Roads_iter_150000.caffemodel \
    --prototxt results/VGG_Roads_2015-10-06_01-27-32/predict_orig.prototxt \
    --output_fn results/VGG_Roads_2015-10-06_01-27-32/prediction_150000/Google$1.png \
    --device_id $2 --n_dups 16 > $1.log 2>&1 &
}

# single_prediction 0 150000 0
single_prediction 1 2
single_prediction 2 3
single_prediction 3 4
single_prediction 4 5
single_prediction 5 6
single_prediction 6 7
single_prediction 7 8
single_prediction 8 0
