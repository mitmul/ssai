#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import re
import os
import glob
import lmdb
import numpy as np
import cv2 as cv
import caffe


def create_single_maps(map_data_dir):
    for map_fn in glob.glob('%s/*.png' % map_data_dir):
        map = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        _, ext = os.path.splitext(map_fn)
        png_fn = map_fn.replace(ext, '_0-1.png')
        if not os.path.exists(png_fn):
            _, map = cv.threshold(map, 1, 1, cv.THRESH_BINARY)
            cv.imwrite(png_fn, map)


def create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                   sat_data_dir, map_data_dir,
                   sat_out_dir, map_out_dir):
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
    sat_size = sat_patch_size
    map_size = map_patch_size
    print 'patch size:', sat_size, map_size, stride

    # get filenames
    sat_fns = np.asarray(
        [f[1] for f in
         sorted([(int(re.search(ur'Google([0-9]+)', f).groups()[0]), f)
                 for f in glob.glob('%s/*.png' % sat_data_dir)])])
    map_fns = np.asarray(
        [f[1] for f in
         sorted([(int(re.search(ur'Google([0-9]+)', f).groups()[0]), f)
                 for f in glob.glob('%s/*0-1.png' % map_data_dir)])])
    index = np.arange(len(sat_fns))
    np.random.shuffle(index)
    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    # create keys
    keys = np.arange(15000000)
    np.random.shuffle(keys)

    n_all_files = len(sat_fns)
    print 'n_all_files:', n_all_files

    n_patches = 0
    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((re.search(ur'(Google[0-9]+)', sat_fn).groups()[0])
                != (re.search(ur'(Google[0-9]+)', map_fn).groups()[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return
        print sat_fn
        print map_fn
        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)

        for y in range(0, sat_im.shape[0] + stride, stride):
            for x in range(0, sat_im.shape[1] + stride, stride):
                if (y + sat_size) > sat_im.shape[0]:
                    y = sat_im.shape[0] - sat_size
                if (x + sat_size) > sat_im.shape[1]:
                    x = sat_im.shape[1] - sat_size

                sat_patch = np.copy(sat_im[y:y + sat_size, x:x + sat_size])
                map_patch = np.copy(map_im[y + sat_size / 2 - map_size / 2:
                                           y + sat_size / 2 + map_size / 2,
                                           x + sat_size / 2 - map_size / 2:
                                           x + sat_size / 2 + map_size / 2])

                key = '%010d' % keys[n_patches]

                # sat db
                sat_patch = sat_patch.swapaxes(0, 2).swapaxes(1, 2)
                datum = caffe.io.array_to_datum(sat_patch, 0)
                value = datum.SerializeToString()
                sat_txn.put(key, value)

                # map db
                map_patch = map_patch.reshape((1, map_patch.shape[0],
                                               map_patch.shape[1]))
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
    # create_single_maps('data/train/map')
    # create_single_maps('data/valid/map')
    # create_single_maps('data/test/map')

    # create_patches(192, 48, 32, 1,
    #                'data/train/sat',
    #                'data/train/map',
    #                'data/train/sat.lmdb',
    #                'data/train/map.lmdb')
    # create_patches(192, 48, 32, 1,
    #                'data/valid/sat',
    #                'data/valid/map',
    #                'data/valid/sat.lmdb',
    #                'data/valid/map.lmdb')
    # create_patches(192, 48, 32, 1,
    #                'data/test/sat',
    #                'data/test/map',
    #                'data/test/sat.lmdb',
    #                'data/test/map.lmdb')
    create_patches(192, 48, 32, 1,
                   'data/trainval/sat',
                   'data/trainval/map',
                   'data/trainval/sat.lmdb',
                   'data/trainval/map.lmdb')
