#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Process, Queue

n_thread = 4
data_queue = Queue()


def worker_thread():
    while True:
        data = data_queue.get()
        if data is None:
            break
        for _ in range(100):
            data += 0.01
        print data

workers = [Process(target=worker_thread) for _ in range(n_thread)]
map(lambda w: w.start(), workers)
[data_queue.put(np.random.rand()) for _ in range(n_thread * 100)]
[data_queue.put(None) for _ in range(n_thread)]
map(lambda w: w.join(), workers)
print 'finished'
