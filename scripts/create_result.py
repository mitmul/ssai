#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import caffe
import argparse
import cv2 as cv
import numpy as np


def get_predict(ortho, net, num,
                l_ch, l_height, l_width,
                d_ch, d_height, d_width, offset=0):
    h_limit = ortho.shape[0]
    w_limit = ortho.shape[1]

    # create input, label patches
    rects = []  # input data region
    o_patches = []
    for y in range(offset, h_limit, l_height):
        for x in range(offset, w_limit, l_width):
            if (y + d_height > h_limit) or (x + d_width > w_limit):
                break
            rects.append((y - offset, x - offset,
                          y - offset + d_height, x - offset + d_width))

            # ortho patch
            o_patch = ortho[y:y + d_height, x:x + d_width, :]
            o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)
            o_patches.append(o_patch)

    o_patches = np.asarray(o_patches, dtype=np.float32)

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

    return pred_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', type=str)
    parser.add_argument('--output_fn', type=str)
    parser.add_argument('--caffemodel', type=str)
    parser.add_argument('--prototxt', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--n_dups', type=int, default=8)
    parser.add_argument('--input_side', type=int, default=128)
    parser.add_argument('--output_side', type=int, default=32)
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.device_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    ortho = cv.imread(args.input_img)
    preds = []
    for offset in range(args.n_dups):
        p = get_predict(ortho, net, 64, 1, args.output_side, args.output_side,
                        3, args.input_side, args.input_side, offset)
        preds.append(p)
        print 'offset:', offset

    shape = preds[0].shape
    canvas = np.zeros((shape[0] + 2 * (args.n_dups - 1),
                       shape[1] + 2 * (args.n_dups - 1), shape[2]))
    for offset, pred in enumerate(preds):
        h, w, c = pred.shape
        canvas[offset:offset + h, offset:offset + w, :] += pred
    canvas /= args.n_dups

    out_h = args.n_dups + shape[0] - 2 * (args.n_dups - 1)
    out_w = args.n_dups + shape[1] - 2 * (args.n_dups - 1)
    canvas = canvas[args.n_dups - 1:out_h, args.n_dups - 1:out_w, :]

    sat_fn = '{}_sat.png'.format(os.path.splitext(args.output_fn)[0])
    pad = (args.input_side - args.output_side) / 2 + (args.n_dups - 1)
    cv.imwrite(sat_fn,
               ortho[pad:pad + canvas.shape[0], pad:pad + canvas.shape[1], :])
    cv.imwrite(args.output_fn, canvas * 255)
    npy_fn = '{}_arr.npy'.format(os.path.splitext(args.output_fn)[0])
    np.save(open(npy_fn, 'wb'), canvas)
