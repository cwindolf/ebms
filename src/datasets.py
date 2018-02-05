import os
import urllib
import gzip
import struct
import numpy as np
from scipy.misc import imresize


MNIST_TRAIN_IMG = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LAB = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

# Some .npz files will be stored here! Watch out...
DATA_PATH = os.path.expanduser('~/data')


def _download_mnist():
    '''
    Don't call this, `mnist` will handle it.
    '''
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    urllib.request.urlretrieve(MNIST_TRAIN_IMG, '/.tmp.gz')
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
