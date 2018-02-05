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
