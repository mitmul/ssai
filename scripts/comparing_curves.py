import numpy as np
import matplotlib.pyplot as plt


models = [
    ['VGG_PReLU_2015-04-21_09-52-02_2', 150000],
    ['VGG_2015-04-27_02-18-32', 50000],
    ['Mnih_CNN_Asym_Zero_2015-04-24_12-12-51', 2000000],
    ['Mnih_CNN_2015-04-23_08-07-50', 1500000],
    ['Mnih_CNN_Asym_2015-04-24_12-12-55', 1500000],
    ['Mnih_CNN_Zero_2015-04-24_12-12-49', 1000000],
    ['Mnih_CNN_Euclid_2015-04-10_06-28-31', 2000000]
]


def breakeven(pre_rec):
    pre_rec = np.array([[pre, rec] for pre, rec in pre_rec
                        if pre != 0 and rec != 0])
    be_pt = np.argmin(np.abs(pre_rec[:, 0] - pre_rec[:, 1]))
    pre, rec = pre_rec[be_pt]

    return pre, rec


def compare_channel(ch):
    plt.ylim([0.65, 1.0])
    plt.xlim([0.65, 1.0])
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--')
    for model, iter in models:
        model_name = model.split('_2015')[0]
        dname = 'results/%s' % model
        dname += '/prediction_%d' % iter
        dname += '/evaluation_%d' % iter

        pre_rec = np.load('%s/pre_rec_%d.npy' % (dname, ch))
        pre, rec = breakeven(pre_rec)
        plt.plot(pre_rec[:, 0], pre_rec[:, 1],
                 label='%s(%.3f)' % (model_name, rec))

        print model_name, pre, rec

    plt.legend(loc='lower left')
    plt.savefig('comparing_%d.pdf' % ch, dpi=300, bbox_inches='tight')

compare_channel(1)
compare_channel(2)
