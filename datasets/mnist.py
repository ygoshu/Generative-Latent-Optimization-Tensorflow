from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pprint import pprint
from collections import Counter

import os.path
import numpy as np
import h5py
import tensorflow as tf

from util import log

__PATH__ = '/scratch//datasets/mnist'

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = 'data.hdf5'

        file = os.path.join(__PATH__, filename)
        log.info("Reading %s ...", file)

        try:
            self.data = h5py.File(file, 'r+')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

    def get_data(self, id):
        # preprocessing and data augmentation
        m = self.data[id]['image'].value/255. * 2 - 1
        try:
            l = self.data[id]['update'].value.astype(np.float32)
        except:
            l = self.data[id]['code'].value.astype(np.float32)
        return m, l

    def set_data(self, id, z):
        try:
            self.data[id]['update'] = z
        except:
            np.allclose(self.data[id]['update'].value, z)
        return

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def get_conv_info():
    return np.array([32, 64, 128])


def get_deconv_info():
    return np.array([[100, 4, 2], [50, 4, 2], [25, 4, 2], [6, 4, 2], [1, 4, 2]])

def create_default_splits(is_train=True, is_few_shot=False, few_shot_class=2):
    ids = all_ids()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    total_y = np.concatenate((y_train,y_test))
    num_trains = 60000
    few_shot_filtered_ids = []
    count = 0
    class_sample_count = Counter()
    if (is_few_shot):
        y_train = y_train
        train_ids = ids[:num_trains]
        f = open("img_ids_for_small_sample_testtxt","w+")
        for train_id in train_ids:
           class_sample_count[total_y[int(train_id)]] += 1 
           is_few_shot_class = total_y[int(train_id)] == few_shot_class
           if (class_sample_count[total_y[int(train_id)]] > 500):
             continue
           if (is_few_shot_class and class_sample_count[total_y[int(train_id)]] < 11): 
             f.write("img id for class: " + train_id) 
           if (is_few_shot_class and class_sample_count[total_y[int(train_id)]] > 10):
             continue 
           few_shot_filtered_ids.append(train_id)
        f.close() 
        with open('relativesize.txt', 'w+') as out:
           pprint(class_sample_count, stream=out)
        dataset_train = Dataset(few_shot_filtered_ids , name='train', is_train=False)
        dataset_test = Dataset(ids[num_trains:], name='test', is_train=False) 
    else:
        dataset_train = Dataset(ids[:num_trains], name='train', is_train=False)
        dataset_test = Dataset(ids[num_trains:], name='test', is_train=False)
    return dataset_train, dataset_test



def all_ids():
    id_filename = 'id.txt'

    id_txt = os.path.join(__PATH__, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

    rs.shuffle(_ids)
    return _ids

