#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
import caffe

caffe.set_mode_gpu()
caffe.set_phase_test()
caffe.set_device(0)

model_fn = '../../models/Multi_Plain_Mnih_NN_S_ReLU/solver.prototxt'
solver = caffe.SGDSolver(model_fn)

if not os.path.exists('../../tmp'):
    os.mkdir('../../tmp')


k = 0
print solver.net.forward()
for i in range(5):
    input_data = \
        solver.net.blobs['input_data'].data[
            i].swapaxes(0, 2).swapaxes(0, 1)
    cv.imwrite('../../tmp/%d_%d_input.png' % (k, i), input_data)

    augment_data = \
        solver.net.blobs['augment1'].data[i].swapaxes(0, 2).swapaxes(0, 1)
    print np.mean(augment_data[:, :, 0]),
    print np.mean(augment_data[:, :, 1]),
    print np.mean(augment_data[:, :, 2])
    print np.std(augment_data[:, :, 0]),
    print np.std(augment_data[:, :, 1]),
    print np.std(augment_data[:, :, 2])
    augment_data -= np.min(augment_data)
    augment_data /= np.max(augment_data)
    cv.imwrite('../../tmp/%d_%d_aug.png' % (k, i), augment_data * 255)

    label = solver.net.blobs['label'].data[
        i].swapaxes(0, 2).swapaxes(0, 1)
    print label.shape
    cv.imwrite('../../tmp/%d_%d_label.png' % (k, i), label * 255)
