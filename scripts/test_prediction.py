#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import basename
import glob
import numpy as np
import cv2 as cv
import caffe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--weight', '-w', type=str)
args = parser.parse_args()
print args

caffe.set_mode_gpu()
caffe.set_phase_test()
caffe.set_device(0)


def get_predict(ortho, net, num,
                l_ch, l_height, l_width,
                d_ch, d_height, d_width):
    h_limit = ortho.shape[0]
    w_limit = ortho.shape[1]

    # create input, label patches
    rects = []  # input data region
    o_patches = []
    l_patches = []
    for y in range(0, h_limit, l_height):
        for x in range(0, w_limit, l_width):
            if (y + d_height > h_limit) or (x + d_width > w_limit):
                break
            rects.append((y, x, y + d_height, x + d_width))

            # ortho patch
            o_patch = ortho[y:y + d_height, x:x + d_width, :]
            o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)
            o_patches.append(o_patch)

    o_patches = np.asarray(o_patches, dtype=np.float32)
    l_patches = np.asarray(l_patches, dtype=np.int32)

    # the number of patches
    n_patches = len(o_patches)

    # create predict, label patches
    pred_patches = np.zeros(
        (n_patches, l_ch, l_height, l_width), dtype=np.float32)
    for i in range(n_patches / num + 1):
        orthos = None
        if (i + 1) * num > n_patches:
            orthos = o_patches[n_patches - num:]
        else:
            orthos = o_patches[i * num:(i + 1) * num]
        net.blobs['input_data'].data[:, :, :, :] = orthos
        predicts = net.forward().values()[0]

        for j, predict in enumerate(predicts):
            if (i + 1) * num <= n_patches:
                pred_patches[i * num + j] = predict
            else:
                pred_patches[n_patches - num + j] = predict

    pred_img = np.zeros((h_limit, w_limit, l_ch), dtype=np.float32)
    for i, (rect, predict) in enumerate(
            zip(rects, pred_patches)):
        predict = predict.swapaxes(0, 2).swapaxes(0, 1)
        pred_img[rect[0] + d_height / 2 - l_height / 2:
                 rect[0] + d_height / 2 + l_height / 2,
                 rect[1] + d_width / 2 - l_width / 2:
                 rect[1] + d_width / 2 + l_width / 2, :] = predict

    out_h = pred_img.shape[0] - (d_height - l_height)
    out_w = pred_img.shape[1] - (d_width - l_width)
    pred_img = pred_img[d_height / 2 - l_height / 2:out_h,
                        d_width / 2 - l_width / 2:out_w, :]
    ortho_img = ortho[d_height / 2 - l_height / 2:out_h,
                      d_width / 2 - l_width / 2:out_w, :]

    return pred_img, ortho_img

if __name__ == '__main__':
    model_fn = args.model
    weight_fn = args.weight
    result_dir = args.model.replace('/%s' % basename(args.model), '')

    net = caffe.Net(model_fn, weight_fn)
    net.set_raw_scale('input_data', 255)
    net.set_channel_swap('input_data', (2, 1, 0))
    num = 64
    l_ch, l_height, l_width = 3, 16, 16
    d_ch, d_height, d_width = 3, 64, 64

    for img_fname in glob.glob('data/mass_merged/test/sat/*.tiff'):
        ortho = cv.imread(img_fname)
        pred_img, ortho_img = get_predict(ortho, net, num,
                                          l_ch, l_height, l_width,
                                          d_ch, d_height, d_width)
        cv.imwrite('%s/pred_%s.png' % (result_dir, basename(img_fname)),
                   pred_img * 125)
        cv.imwrite('%s/ortho_%s.png' % (result_dir, basename(img_fname)),
                   ortho_img)
        np.save('%s/pred_%s' % (result_dir, basename(img_fname)),
                pred_img)

        print img_fname
