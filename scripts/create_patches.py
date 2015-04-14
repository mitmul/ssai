import cv2 as cv
import glob
import os
from os.path import basename
from os.path import splitext


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)


split_num = 10
sat_dname = 'data/mass_merged/train/sat'
map_dname = 'data/mass_merged/train/map'

sat_out = 'data/mass_maerged_split/train/sat'
map_out = 'data/mass_maerged_split/train/map'
mkdir(sat_out)
mkdir(map_out)

sat_fnames = sorted(glob.glob('%s/*.tif*' % sat_dname))
map_fnames = sorted(glob.glob('%s/*.tif*' % map_dname))

for sat_fn, map_fn in zip(sat_fnames, map_fnames):
    sat = cv.imread(sat_fn)
    map = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
    h, w, c = sat.shape
    h_side = h / split_num
    w_side = w / split_num
    for st_h in range(0, h - h_side, h_side / 2):
        for st_w in range(0, w - w_side, w_side / 2):
            sp_sat = sat[st_h:st_h + h_side,
                         st_w:st_w + w_side]
            sp_map = map[st_h:st_h + h_side,
                         st_w:st_w + w_side]
            cv.imwrite('%s/%s' % (sat_out, basename(sat_fn)), sp_sat)
            cv.imwrite('%s/%s' % (map_out, basename(map_fn)), sp_map)
            fn, _ = splitext(basename(map_fn))
            cv.imwrite('%s/%s.png' % (map_out, fn), sp_map * 125)

    print sat_fn, map_fn
