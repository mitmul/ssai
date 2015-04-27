import time
import shutil
import lmdb
import os
import glob
import argparse
import caffe
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--weight', '-w', type=str)
parser.add_argument('--sat_dir', '-s', type=str)
parser.add_argument('--map_dir', '-p', type=str)
parser.add_argument('--out_dir', '-o', type=str)
parser.add_argument('--channel', '-c', type=int, default=3)
parser.add_argument('--device', '-d', type=int, default=0)
parser.add_argument('--mode', '-e', type=str)
parser.add_argument('--pred_data_dir', type=str)
parser.add_argument('--map_data_dir', type=str)
parser.add_argument('--pred_db_dir', type=str)
parser.add_argument('--map_db_dir', type=str)
args = parser.parse_args()
print args

caffe.set_mode_gpu()
caffe.set_device(args.device)


def get_predict(ortho, map, net, num,
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
    map_img = map[d_height / 2 - l_height / 2:out_h,
                  d_width / 2 - l_width / 2:out_w]

    return pred_img, ortho_img, map_img


def create_patches(pred_patch_size, map_patch_size, stride, map_ch,
                   pred_data_dir, map_data_dir,
                   pred_out_dir, map_out_dir):
    if os.path.exists(pred_out_dir):
        shutil.rmtree(pred_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    os.makedirs(pred_out_dir)
    os.makedirs(map_out_dir)

    # db
    sat_env = lmdb.Environment(pred_out_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=True)
    map_env = lmdb.Environment(map_out_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=True)

    # patch size
    sat_size = pred_patch_size
    map_size = map_patch_size
    print 'patch size:', sat_size, map_size, stride

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.npy' % pred_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.npy' % map_data_dir)))
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
        if ((os.path.basename(sat_fn).split('.')[0])
                != (os.path.basename(map_fn).split('.')[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return

        sat_im = np.load(sat_fn)
        map_im = np.load(map_fn)

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
                # convert to 8bit data
                sat_patch = np.asarray(sat_patch * 255, dtype=np.uint8)
                datum = caffe.io.array_to_datum(sat_patch)

                # too slow!
                # datum.float_data.extend(sat_patch.flatten().tolist())

                value = datum.SerializeToString()
                sat_txn.put(key, value)

                # map db
                if map_ch == 3:
                    map_patch_multi = []
                    for ch in range(map_ch):
                        map_patch_multi.append(np.asarray(map_patch == ch,
                                                          dtype=np.uint8))
                    map_patch = np.asarray(map_patch_multi, dtype=np.uint8)
                elif map_ch == 1:
                    map_patch = map_patch.reshape((1, map_patch.shape[0],
                                                   map_patch.shape[1]))

                datum = caffe.io.array_to_datum(map_patch)
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
    num = 64
    l_ch, l_height, l_width = args.channel, 16, 16
    d_ch, d_height, d_width = 3, 64, 64

    if args.mode == 'create_predictions':
        net = caffe.Net(args.model, args.weight, caffe.TEST)
        pred_out_dir = '%s/pred' % args.out_dir
        map_out_dir = '%s/map' % args.out_dir
        if not os.path.exists(pred_out_dir):
            os.makedirs(pred_out_dir)
        if not os.path.exists(map_out_dir):
            os.makedirs(map_out_dir)

        sat_fns = sorted(glob.glob('%s/*.tif*' % args.sat_dir))
        map_fns = sorted(glob.glob('%s/*.tif*' % args.map_dir))
        for sat_fn, map_fn in zip(sat_fns, map_fns):
            ortho = cv.imread(sat_fn)
            label = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
            pred_img, ortho_img, map_img = get_predict(ortho, label, net, num,
                                                       l_ch, l_height, l_width,
                                                       d_ch, d_height, d_width)
            base, ext = os.path.splitext(sat_fn)
            base = os.path.basename(base)
            np.save('%s/%s' % (pred_out_dir, base), pred_img)
            np.save('%s/%s' % (map_out_dir, base), map_img)
            print pred_img.shape, ortho_img.shape, map_img.shape

    if args.mode == 'create_dataset':
        create_patches(92, 24, 16, 3,
                       args.pred_data_dir, args.map_data_dir,
                       args.pred_db_dir, args.map_db_dir)
