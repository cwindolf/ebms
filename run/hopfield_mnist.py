from src.hopfield import HopfieldNet
from src.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from random import choice
import seaborn as sns
sns.set_style('white')

# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Try it out on mnist.
    # We'll train by just storing the average 0 and the average 1 as states,
    # and seeing what the thing does with that.

    # Load data
    zeros = list(mnist(0, img_height=14, not_mod=2))
    ones = list(mnist(1, img_height=14, not_mod=2))

    # Representative states. could try mean squares?
    avg_zero = np.mean(zeros, axis=0)
    avg_one = np.mean(ones, axis=0)

    # threshold to binary
    mean = (np.mean(avg_zero) + np.mean(avg_one)) / 2
    Z = np.zeros_like(avg_zero)
    O = np.ones_like(avg_one)
    zeros = [np.where(z > mean, 1, 0) for z in zeros]
    ones = [np.where(o > mean, 1, 0) for o in ones]
    avg_zero = np.where(avg_zero > mean, 1, 0)
    avg_one = np.where(avg_one > mean, 1, 0)

    # *********************************************************************** #
    # init model

    hopnet = HopfieldNet([avg_zero, avg_one])

    plt.axis('off')
    plt.imshow(hopnet.T)
    plt.title('the Hopfield net\'s weights matrix')
    plt.show()


    # *********************************************************************** #
    # run on a couple of random examples and plot them with the net's result.

    fig, axes = plt.subplots(7, 2, figsize=(8, 14), sharex=True, sharey=True)
    for a in axes.flat:
        a.set_xticklabels([])
        a.set_yticklabels([])

    for (a1, a2), v in zip(axes, (choice(zeros) if i % 2 else choice(ones)
                                  for i in range(6))):
        a1.imshow(v.reshape((14, 14)), interpolation='none')
        a1.set_xlabel('Input')
        pred = hopnet.sample(v)
        a2.imshow(pred.reshape((14, 14)), interpolation='none')
        a2.set_xlabel('Result')

    a1, a2 = axes[-1]
    a1.imshow(avg_zero.reshape((14, 14)), interpolation='none')
    a2.imshow(avg_one.reshape((14, 14)), interpolation='none')
    a1.set_xlabel('Stored Representation of 0')
    a2.set_xlabel('Stored Representation of 1')

    plt.show()