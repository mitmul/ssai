#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import glob
import subprocess
import argparse
import os
from multiprocessing import Process
from os.path import basename

get_sat_dir = lambda t: '../../data/mass_%s/test/sat' % t.lower()
get_map_dir = lambda t: '../../data/mass_%s/test/map' % t.lower()
channel = None


def parallel_pred_eval(pred, evaluate, model_dir, n_iter, snapshot_fn, test_sat_dir, test_map_dir, channel):
    worker = Process(
        target=pred, args=(model_dir, snapshot_fn, test_sat_dir, channel))
    worker.start()
    worker.join()

    result_dir = 'prediction_%d' % n_iter
    worker = Process(
        target=evaluate, args=(model_dir, test_map_dir, result_dir, channel))
    worker.start()
    worker.join()


def predict(model_dir, snapshot_fn, test_sat_dir, channel):
    os.chdir(model_dir)
    print subprocess.check_output(['ls'])
    cmd = [
        'python', '../../scripts/test_prediction.py',
        '--model', 'predict.prototxt',
        '--weight', 'snapshots/%s' % basename(snapshot_fn),
        '--img_dir', test_sat_dir,
        '--channel', str(channel)
    ]
    subprocess.check_output(cmd)


def evaluate(model_dir, test_map_dir, result_dir, channel):
    os.chdir(model_dir)
    cmd = [
        'python', '../../scripts/test_evaluation.py',
        '--map_dir', test_map_dir,
        '--result_dir', result_dir,
        '--channel', str(channel)
    ]
    subprocess.check_output(cmd)

get_iter = lambda fn: int(re.search(ur'_([0-9]+)\.', basename(fn)).groups()[0])
for model_dir in glob.glob('results/*'):
    for snapshot_fn in glob.glob('%s/snapshots/*.caffemodel' % model_dir):
        n_iter = get_iter(snapshot_fn)
        if n_iter % 100000 == 0:
            dname = os.path.dirname(snapshot_fn)
            pred_dname = '%s/prediction_%d' % (model_dir, n_iter)
            if os.path.exists(pred_dname):
                continue

            if 'Buildings_2015' in model_dir:
                channel = 1
                test_sat_dir = get_sat_dir('buildings')
                test_map_dir = get_map_dir('buildings')
            elif 'Roads_2015' in model_dir:
                channel = 1
                test_sat_dir = get_sat_dir('roads')
                test_map_dir = get_map_dir('roads')
            elif 'Roads_Mini_2015' in model_dir:
                channel = 1
                test_sat_dir = get_sat_dir('roads')
                test_map_dir = get_map_dir('roads')
            else:
                channel = 3
                test_sat_dir = get_sat_dir('merged')
                test_map_dir = get_map_dir('merged')

            worker = Process(target=parallel_pred_eval,
                             args=(predict, evaluate, model_dir, n_iter, snapshot_fn, test_sat_dir, test_map_dir, channel))
            worker.start()

            print pred_dname, channel, test_sat_dir, test_map_dir
