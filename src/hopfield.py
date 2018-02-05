'''
This is a Numpy implementation of the network described in Hopfield's 1982
paper titled "Neural networks and physical systems with emergent collective
computational abilities".

This network is structured as a content-addressible memory, and so its "API"
was not what I, at least, was used to. I.e., it does not directly attempt to
approximate a function between two Euclidean spaces. More interestingly, it
differs from the interface of the Boltzmann machine, even though the Boltzmann
machine is in some ways the `stochastic version` of this net. Anyway, maybe
they can be made compatible, I'm not sure.
'''
import numpy as np
from itertools import product


class HopfieldNet:

    def __init__(self, V):
        '''
        Initialize a new Hopfield net, using Hopfield's information storage
        algorithm to compute the necessary weights for storing `V` in
        memory.

        Note we are taking the biases to be 0 like Hopfield did.

        @param  V  List-like of 1-d np.arrays
        '''
        # dumb check.
        V = np.asarray(V)
        assert len(V.shape) == 2

        # init weights
        self.n = V[0].shape[0]
        self.T = np.zeros((self.n, self.n)).astype(np.float64)

        # Some buffers for the current state of the net, and for scratch
        self.X = np.zeros(self.n).astype(np.float64)
        self._X = np.zeros(self.n).astype(np.float64)

        # train. Hopfield's rule
        # $T_{i,j} = \sum_s (2 V^s_i - 1)(2 V^s_j - 1)$
        for i in range(self.n):
            for j in range(self.n):
                self.T[i, j] = sum((2. * V[s, i] - 1.) * (2. * V[s, j] - 1.)
                                   for s in range(V.shape[0]))

    def sample(self, x, max_iters=100):
        '''
        Iterate on input `x` until convergence or `max_iters`.
        Each step, we go through all of the states in a (new) random order,
        and update them according to the rule
        $X_i=\operatorname{ciel}(\operatorname{relu}(\sum_{j=1}^n T_{ij} X_j))$
        where $n$ is the number of nodes, and this mess of ciels and relus is
        just meant to say that if $W_i\dot X > 0$, the result is 1, and 0
        otherwise.

        @param x            np.array    the input at which to start iterating
        @param max_iters    int         if we don't converge, when to stop?
        '''
        self.X[:] = x
        self._X[:] = x
        order = list(range(self.n))
        for _ in range(max_iters):
            self._X[:] = self.X
            np.random.shuffle(order)
            for i in order:
                s = sum(self.T[i, j] * self.X[j] for j in range(self.n))
                if s >= 0:
                    self.X[i] = 1.
                else:
                    self.X[i] = 0.
            if (self.X == self._X).all():
                break
        return self.X


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
