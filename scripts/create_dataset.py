#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
import glob
import lmdb
import caffe
import numpy as np
import cv2 as cv


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


def create_patches(patch_size, sat_data_dir, map_data_dir, sat_out_dir,
                   map_out_dir, map_ch):
    if os.path.exists(sat_out_dir):
        shutil.rmtree(sat_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    os.makedirs(sat_out_dir)
    os.makedirs(map_out_dir)

    # db
    sat_env = lmdb.Environment(sat_out_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=True)
    map_env = lmdb.Environment(map_out_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=True)

    # patch size
    sat_size = int(patch_size * 1.5)
    map_size = int(16 * 1.5)
    stride = int(16)
    print 'patch size:', sat_size, map_size, stride

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))
    index = np.arange(len(sat_fns))
    np.random.shuffle(index)
    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    # create keys
    keys = np.arange(5000000)
    np.random.shuffle(keys)

    n_all_files = len(sat_fns)
    print 'n_all_files:', n_all_files

    n_patches = 0
    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((os.path.basename(sat_fn).split('.')[0])
                != (os.path.basename(map_fn).split('.')[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return

        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)

        for y in range(0, sat_im.shape[0], stride):
            for x in range(0, sat_im.shape[1], stride):
                if (y + sat_size > sat_im.shape[0] or
                        x + sat_size > sat_im.shape[1]):
                    continue

                sat_patch = np.copy(sat_im[y:y + sat_size, x:x + sat_size])
                map_patch = np.copy(map_im[y + sat_size / 2 - map_size / 2:
                                           y + sat_size / 2 + map_size / 2,
                                           x + sat_size / 2 - map_size / 2:
                                           x + sat_size / 2 + map_size / 2])

                # exclude patch including big white region
                if np.sum(np.sum(sat_patch, axis=2) == 255 * 3) > 16:
                    continue

                key = '%010d' % keys[n_patches]

                # sat db
                sat_patch = sat_patch.swapaxes(0, 2).swapaxes(1, 2)
                datum = caffe.io.array_to_datum(sat_patch, 0)
                value = datum.SerializeToString()
                sat_txn.put(key, value)

                # map db
                map_patch_multi = []
                for ch in range(map_ch):
                    map_patch_multi.append(np.asarray(map_patch == ch,
                                                      dtype=np.uint8))
                map_patch = np.asarray(map_patch_multi, dtype=np.uint8)
                datum = caffe.io.array_to_datum(map_patch, 0)
                value = datum.SerializeToString()
                map_txn.put(key, value)

                n_patches += 1

                if n_patches % 10000 == 0:
                    sat_txn.commit()
                    sat_txn = sat_env.begin(write=True, buffers=True)
                    map_txn.commit()
                    map_txn = map_env.begin(write=True, buffers=True)

        print file_i, '/', n_all_files, 'n_patches:', n_patches

    sat_txn.commit()
    sat_env.close()
    map_txn.commit()
    map_env.close()
    print 'patches:\t', n_patches

if __name__ == '__main__':
    create_merged_map()
    create_patches(64,
                   'data/mass_merged/valid/sat',
                   'data/mass_merged/valid/map',
                   'data/mass_merged/lmdb/valid_sat',
                   'data/mass_merged/lmdb/valid_map', 3)
    create_patches(64,
                   'data/mass_merged/test/sat',
                   'data/mass_merged/test/map',
                   'data/mass_merged/lmdb/test_sat',
                   'data/mass_merged/lmdb/test_map', 3)
    create_patches(64,
                   'data/mass_merged/train/sat',
                   'data/mass_merged/train/map',
                   'data/mass_merged/lmdb/train_sat',
                   'data/mass_merged/lmdb/train_map', 3)
