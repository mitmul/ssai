#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    print(args)

    if not os.path.exists('{}/map_1m'.format(args.data)):
        os.mkdir('{}/map_1m'.format(args.data))
    if not os.path.exists('{}/sat_1m'.format(args.data)):
        os.mkdir('{}/sat_1m'.format(args.data))

    for dtype in ['sat', 'map']:
        for fn in glob.glob('{}/{}/*.png'.format(args.data, dtype)):
            print fn,
            img = cv.imread(fn)
            img = cv.resize(
                img, (int(img.shape[1] * 0.58), int(img.shape[0] * 0.58)))
            sfn = fn.replace(dtype, '{}_1m'.format(dtype))
            print sfn
            cv.imwrite(sfn, img)
