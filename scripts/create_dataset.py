#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
import glob
import lmdb
import caffe
import numpy as np
import cv2 as cv
import json


def create_merged_map():
    # copy sat images
    for data_type in ['train', 'test', 'valid']:
        out_dir = 'data/mass_merged/%s/sat' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('data/mass_buildings/%s/sat/*.tiff' % data_type):
            shutil.copy(fn, '%s/%s' % (out_dir, os.path.basename(fn)))

    road_maps = dict([(os.path.basename(fn).split('.')[0], fn)
                      for fn in glob.glob('data/mass_roads/*/map/*.tif')])

    # combine map images
    for data_type in ['train', 'test', 'valid']:
        out_dir = 'data/mass_merged/%s/map' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('data/mass_buildings/%s/map/*.tif' % data_type):
            base = os.path.basename(fn).split('.')[0]
            building_map = cv.imread(fn, cv.IMREAD_GRAYSCALE)
            road_map = cv.imread(road_maps[base], cv.IMREAD_GRAYSCALE)
            _, building_map = cv.threshold(
                building_map, 0, 1, cv.THRESH_BINARY)
            _, road_map = cv.threshold(road_map, 0, 1, cv.THRESH_BINARY)
            h, w = road_map.shape
            merged_map = np.zeros((h, w))
            merged_map += building_map
            merged_map += road_map * 2
            merged_map = np.where(merged_map > 2, 0, merged_map)
            cv.imwrite('data/mass_merged/%s/map/%s.tif' % (data_type, base),
                       merged_map)
            print merged_map.shape, fn
            merged_map = np.array([np.where(merged_map == 0, 1, 0),
                                   np.where(merged_map == 1, 1, 0),
                                   np.where(merged_map == 2, 1, 0)])
            merged_map = merged_map.swapaxes(0, 2).swapaxes(0, 1)
            cv.imwrite('data/mass_merged/%s/map/%s.png' % (data_type, base),
                       merged_map * 255)


def create_dataset(sat_data_dir, map_data_dir, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # db
    env = lmdb.Environment(out_dir, map_size=1099511627776)
    txn = env.begin(write=True, buffers=True)

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))

    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((os.path.basename(sat_fn).split('.')[0])
                != (os.path.basename(map_fn).split('.')[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return

        sat_fn = os.path.abspath(sat_fn)
        map_fn = os.path.abspath(map_fn)

        key = '%010d' % file_i

        data = json.dumps([sat_fn, map_fn])
        txn.put(key, data)

        if file_i % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True, buffers=True)

        print file_i

    txn.commit()
    env.close()

if __name__ == '__main__':
    # create_merged_map()
    create_dataset('data/mass_merged/valid/sat',
                   'data/mass_merged/valid/map',
                   'data/mass_merged/lmdb/valid.lmdb')
    create_dataset('data/mass_merged/test/sat',
                   'data/mass_merged/test/map',
                   'data/mass_merged/lmdb/test.lmdb')
    create_dataset('data/mass_merged/train/sat',
                   'data/mass_merged/train/map',
                   'data/mass_merged/lmdb/train.lmdb')
