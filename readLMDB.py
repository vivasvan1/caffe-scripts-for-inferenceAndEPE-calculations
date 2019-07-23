import caffe
import numpy as np
import matplotlib.pyplot as plt

import lmdb

lmdb_env = lmdb.open('./../data/KLensDataset_TEST_relese_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    print(key)
    # print(value)
    datum.ParseFromString(value)
    label = datum.label
    print(len(datum.data))
    # print(datum.height)
    # datum.width = 900
    data = caffe.io.datum_to_array(datum)
    data = np.rollaxis(data, 0, 3)
    plt.imshow(data[:,:,0:3])
    plt.show()
    break
    