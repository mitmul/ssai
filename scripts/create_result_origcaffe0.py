#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import caffe
import argparse
import cv2 as cv
import numpy as np


def get_predict(ortho, net, num,
                l_ch, l_height, l_width,
                d_ch, d_height, d_width, rects, offset=0):
    h_limit = ortho.shape[0]
    w_limit = ortho.shape[1]

    # create input, label patches
    o_patches = []
    for y in range(offset, h_limit, l_height):
        for x in range(offset, w_limit, l_width):
            if (y + d_height > h_limit) or (x + d_width > w_limit):
                break

            # ortho patch
            o_patch = ortho[
                y:y + d_height, x:x + d_width, :].astype(np.float32)
            for ch in range(o_patch.shape[2]):
                o_patch[:, :, ch] -= np.mean(o_patch[:, :, ch])
                o_patch[:, :, ch] /= np.std(o_patch[:, :, ch])
            o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)
            o_patches.append(o_patch)

    o_patches = np.asarray(o_patches, dtype=np.float32)

    # the number of patches
    n_patches = len(o_patches)

    # create predict, label patches
    pred_patches = np.zeros(
        (n_patches, l_ch, l_height, l_width), dtype=np.float32)
    for i in range(n_patches / num + 1):
        orthos = o_patches[i * num:(i + 1) * num]
        net.blobs['input_data'].reshape(*(orthos.shape))
        pred_patches[i * num:i * num + orthos.shape[0]] = net.forward(
            input_data=orthos.astype(np.float32, copy=False)).values()[0]
        print 'pred_patches:{:4d}/{:4d}'.format(i, n_patches / num)

    lt_offset = d_height / 2 - l_height / 2
    rb_offset = d_height / 2 + l_height / 2
    pred_img = np.zeros((h_limit, w_limit, l_ch), dtype=np.float32)
    for i, (rect, predict) in enumerate(zip(rects, pred_patches)):
        predict = predict.swapaxes(0, 2).swapaxes(0, 1)
        pred_img[rect[0] + lt_offset:rect[0] + rb_offset,
                 rect[1] + lt_offset:rect[1] + rb_offset, :] = predict

    out_h = pred_img.shape[0] - (d_height / 2 - l_height / 2)
    out_w = pred_img.shape[1] - (d_width / 2 - l_width / 2)
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
    rects = []
    for y in range(0, ortho.shape[0], args.output_side):
        for x in range(0, ortho.shape[1], args.output_side):
            if ((y + args.input_side > ortho.shape[0])
                    or (x + args.input_side > ortho.shape[1])):
                break
            rects.append((y, x, y + args.input_side, x + args.input_side))

    for offset in range(args.n_dups):
        p = get_predict(ortho, net, 64, 1, args.output_side, args.output_side,
                        3, args.input_side, args.input_side, rects, offset)
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
