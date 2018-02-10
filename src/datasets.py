import os
import urllib.request
import gzip
import struct
import numpy as np
from scipy.misc import imresize
from scipy.io import loadmat

# Where does mnist live
MNIST_TRAIN_IMG = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LAB = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

# Where does Caltech101Sillhou live
CTECH_MAT = 'http://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28.mat'

# Some .npz files will be stored here! Watch out...
DATA_PATH = os.path.expanduser('~/data')


def _download_mnist():
    '''
    Don't call this, `mnist` will handle it.
    '''
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    urllib.request.urlretrieve(MNIST_TRAIN_IMG, DATA_PATH + '/.tmp.gz')
    with gzip.open(DATA_PATH + '/.tmp.gz', 'rb') as image_data:
        image_data.read(16)
        images = np.empty((60000, 784), dtype=np.float32)
        for i in range(60000):
            bytes = image_data.read(784)
            image = np.asarray(struct.unpack('784B', bytes), dtype=np.float32)
            images[i] = image
    os.remove(DATA_PATH + '/.tmp.gz')
    urllib.request.urlretrieve(MNIST_TRAIN_LAB, DATA_PATH + '/.tmp.gz')
    with gzip.open(DATA_PATH + '/.tmp.gz', 'rb') as label_data:
        label_data.read(8)
        labels = np.empty((10000,), dtype=np.int)
        for i in range(10000):
            byte = label_data.read(1)
            labels[i] = struct.unpack('1B', byte)[0]
    os.remove(DATA_PATH + '/.tmp.gz')
    np.savez_compressed(DATA_PATH + '/mnist.npz', i=images, l=labels)


def _download_caltech():
    '''
    Don't call this, caltech functions will do it. "Just a cache."
    '''
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    urllib.request.urlretrieve(CTECH_MAT, DATA_PATH + '/ctech28.mat')
    mat = loadmat(DATA_PATH + '/ctech28.mat')
    images = mat['X']
    int_labels = mat['Y'][0]
    labels_dict = {i + 1: lab[0] for i, lab in enumerate(mat['classnames'][0])}
    np.savez_compressed(DATA_PATH + '/caltech.npz',
                        i=images, l=int_labels, d=labels_dict)
    os.remove(DATA_PATH + '/ctech28.mat')


def mnist(label, img_height=28, not_mod=1):
    '''
    Make a generator yielding all of the images in the
    MNIST dataset with label `label`.
    '''
    if not os.path.isdir(DATA_PATH) or not os.path.isfile(DATA_PATH + '/mnist.npz'):
        _download_mnist()
    dat = np.load(DATA_PATH + '/mnist.npz')
    images, labels = dat['i'], dat['l']
    skip_counter = 0
    for (i, l) in zip(images, labels):
        if l == label:
            if not (skip_counter % not_mod):
                if img_height != 28:
                    # resize
                    im = imresize(i.reshape((28, 28)), (img_height, img_height))
                    i = im.reshape(img_height * img_height)
                    yield i
            skip_counter += 1


def mnist_class(class_num, extra='', img_height=28, not_mod=1):
    np.savez_compressed(DATA_PATH + '/mnist%d%s.npz' % (class_num, extra),
        np.asarray(list(mnist(class_num, img_height=img_height))))


def caltech(int_label=False, string_label=False):
    '''
    Produce a generator running through all the images in the Caltech 101
    Silhouettes datasets. Generator will yield the requested labels in
    that order.
    '''
    if not os.path.isdir(DATA_PATH) or not os.path.isfile(DATA_PATH + '/caltech.npz'):
        _download_caltech()
    dat = np.load(DATA_PATH + '/caltech.npz')
    ims, ils, key = dat['i'], dat['l'], dat['d']
    for i, l in zip(ims, ils):
        if int_label and string_label:
            yield i, l, key[l]
        elif int_label:
            yield i, l
        elif string_label:
            yield i, key[l]
        else:
            yield i
