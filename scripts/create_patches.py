import numpy as np
import cv2 as cv
import glob
import os
from os.path import basename
from os.path import splitext


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)


for db_type in ['train', 'test', 'valid']:
    split_num = 5
    sat_dname = 'data/mass_merged/%s/sat' % db_type
    map_dname = 'data/mass_merged/%s/map' % db_type

    sat_out = 'data/mass_merged_split/%s/sat' % db_type
    map_out = 'data/mass_merged_split/%s/map' % db_type
    mkdir(sat_out)
    mkdir(map_out)

    sat_fnames = sorted(glob.glob('%s/*.tif*' % sat_dname))
    map_fnames = sorted(glob.glob('%s/*.tif*' % map_dname))

    for sat_fn, map_fn in zip(sat_fnames, map_fnames):
        sat = cv.imread(sat_fn)
        map = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        h, w, c = sat.shape
        h_side = 92
        w_side = 92
        for st_h in range(0, h, 16):
            for st_w in range(0, w, 16):
                st_h = st_h if (st_h + h_side) < h else h - h_side
                st_w = st_w if (st_w + w_side) < w else w - w_side
                en_h = st_h + h_side if (st_h + h_side) < h else h
                en_w = st_w + w_side if (st_w + w_side) < w else w
                sp_sat = sat[st_h:en_h, st_w:en_w]
                sp_map = map[st_h:en_h, st_w:en_w]

                if np.sum(sp_sat) > h_side * w_side * 255 * 0.6:
                    continue

                fn, _ = splitext(basename(map_fn))
                cv.imwrite('%s/%s_%d-%d.tif' %
                           (sat_out, fn, st_h, st_w), sp_sat)
                cv.imwrite('%s/%s_%d-%d.tif' %
                           (map_out, fn, st_h, st_w), sp_map)
                cv.imwrite('%s/%s_%d_%d.png' %
                           (map_out, fn, st_h, st_w), sp_map * 125)

        print sat_fn, map_fn
