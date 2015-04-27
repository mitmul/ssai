import numpy as np
import glob
import os

for dname in sorted(glob.glob('results/*')):
    if not os.path.isdir(dname):
        continue
    model_name = os.path.basename(dname)
    for pred_dir in sorted(glob.glob('%s/prediction_*' % dname)):
        iter = int(pred_dir.split('_')[-1].strip())
        recalls = []
        for pre_rec_fn in sorted(glob.glob(
                '%s/evaluation_%d/*.npy' % (pred_dir, iter))):
            pre_rec = np.load(pre_rec_fn)
            pre_rec = np.array([[pre, rec] for pre, rec in pre_rec
                                if pre != 0 and rec != 0])
            be_pt = np.argmin(np.abs(pre_rec[:, 0] - pre_rec[:, 1]))
            ch = int(pre_rec_fn.split('_')[-1].split('.')[0])
            pre, rec = pre_rec[be_pt]
            recalls.append(rec)

        if len(recalls) == 3:
            msg = model_name + ','
            msg += str(iter) + ','
            msg += ','.join([str(r) for r in recalls])
            print msg
