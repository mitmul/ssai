#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import caffe
import argparse
import ctypes
import time
import cv2 as cv
import numpy as np
from multiprocessing import Process, Queue, Array


def create_minibatch(n_dups, l_height, l_width, d_height, d_width, h_limit,
                     w_limit, ortho, num, queue):
    minibatch = []
    for d in range(n_dups):
        for y in range(d, h_limit, l_height):
            for x in range(d, w_limit, l_width):
                if (y + d_height > h_limit) or (x + d_width > w_limit):
                    break
                # ortho patch
                o_patch = ortho[
                    y:y + d_height, x:x + d_width, :].astype(np.float32, copy=False)
                for ch in range(o_patch.shape[2]):
                    o_patch[:, :, ch] -= np.mean(o_patch[:, :, ch])
                    o_patch[:, :, ch] /= np.std(o_patch[:, :, ch])
                o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)

                minibatch.append(o_patch)
                if len(minibatch) == num:
                    queue.put(np.asarray(minibatch, dtype=np.float32))
                    minibatch = []
    queue.put(None)


def tile_patches(sat_shape, queue, n_dups, h_limit, w_limit, l_height, l_width,
                 d_height, d_width, canvas):
    for d in range(n_dups):
        st = time.time()
        for y in range(d, h_limit, l_height):
            for x in range(d, w_limit, l_width):
                if (y + d_height > h_limit) or (x + d_width > w_limit):
                    break
                pred = queue.get()
                if pred is None:
                    break
                pred = pred.swapaxes(0, 2).swapaxes(0, 1)
                canvas[y:y + l_height, x:x + l_width, :] += pred
        print 'offset:{} ({} sec)'.format(d, time.time() - st)

def get_predict(ortho, net, num,
                l_ch, l_height, l_width,
                d_ch, d_height, d_width, n_dups):
    h_limit = ortho.shape[0]
    w_limit = ortho.shape[1]
    canvas_h = h_limit - (d_height - l_height) / 2
    canvas_w = w_limit - (d_width - l_width) / 2

    # to share 'canvas' between different threads
    canvas_ = Array(ctypes.c_float, canvas_h * canvas_w * l_ch)
    canvas = np.ctypeslib.as_array(canvas_.get_obj())
    canvas = canvas.reshape((canvas_h, canvas_w, l_ch))

    # prepare queues and threads
    patch_queue = Queue(maxsize=1)
    preds_queue = Queue()
    patch_worker = Process(target=create_minibatch,
                           args=(n_dups, l_height, l_width, d_height, d_width,
                                 h_limit, w_limit, ortho, num, patch_queue))
    canvas_worker = Process(target=tile_patches,
                            args=(ortho.shape, preds_queue, n_dups, h_limit,
                                  w_limit, l_height, l_width, d_height,
                                  d_width, canvas))
    patch_worker.start()
    canvas_worker.start()

    while True:
        minibatch = patch_queue.get()
        if minibatch is None:
            break
        net.blobs['input_data'].reshape(*(minibatch.shape))
        preds = net.forward(input_data=minibatch).values()[0]
        [preds_queue.put(pred) for pred in preds]
    preds_queue.put(None)
    patch_worker.join()
    canvas_worker.join()

    return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', type=str)
    parser.add_argument('--output_fn', type=str)
    parser.add_argument('--caffemodel', type=str)
    parser.add_argument('--prototxt', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--n_dups', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--input_side', type=int, default=128)
    parser.add_argument('--output_side', type=int, default=32)
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.device_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    ortho = cv.imread(args.input_img)
    output = get_predict(
        ortho, net, args.batchsize, 1, args.output_side, args.output_side, 3,
        args.input_side, args.input_side, args.n_dups)
    output /= args.n_dups

    # save results
    cv.imwrite(args.output_fn, output * 255)
    npy_fn = '{}_arr.npy'.format(os.path.splitext(args.output_fn)[0])
    np.save(open(npy_fn, 'wb'), output)
