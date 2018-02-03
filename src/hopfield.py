'''
This is a Numpy implementation of the network described in Hopfield's 1982
paper titled `Neural networks and physical systems with emergent collective
computational abilities`.
'''
import numpy as np
from itertools import product


class HopfieldNet:

    def __init__(self, S):
        '''
        Initialize a new Hopfield net, using Hopfield's information storage
        algorithm to compute the necessary weights for storing `S` in
        memory.

        Note we are taking the biases to be 0 like Hopfield did.

        @param  S  List-like of 1-d np.arrays
        '''
        # dumb check.
        S = np.asarray(S)
        assert len(S.shape) == 2

        # init weights
        self.n = S[0].shape[0]
        self.V = np.zeros(self.n).astype(np.float64)
        self._V = np.zeros(self.n).astype(np.float64)
        self.T = np.zeros((self.n, self.n)).astype(np.float64)

        # train
        for i in range(self.n):
            for j in range(self.n):
                self.T[i, j] = sum((2. * S[s, i] - 1.) * (2. * S[s, j] - 1.)
                                   for s in range(S.shape[0]))

    def sample(self, x, max_iters=1000):
        '''
        Iterate on input `x` until convergence or `max_iters`.
        '''
        self.V[:] = x
        self._V[:] = x
        order = list(range(self.n))
        for _ in range(max_iters):
            self._V = self.V
            np.random.shuffle(order)
            for i in order:
                s = sum(self.T[i, j] * self.V[j] for j in range(self.n))
                if s >= 0:
                    self.V[i] = 1.
                else:
                    self.V[i] = 0.
            if (self.V == self._V).all():
                break
        return self.V


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    from datasets import mnist
    import matplotlib.pyplot as plt
    from random import choice
    import seaborn as sns
    sns.set_style('white')

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
