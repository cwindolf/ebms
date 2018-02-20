'''
Here we implement a Deep Belief Net trained by single-step contrastive
divergence. We follow Hinton, 2002 (`Training Products of Experts by Minimizing
Contrastive Divergence') and Hinton et al., 2006 (`A Fast Learning Algorithm
for Deep Belief Nets') for reference. Another good thing to look at is Hinton's
Matlab RBM impl `http://www.cs.toronto.edu/~hinton/code/rbm.m`.

We note that this yields an RBM when instantiated with only one hidden layer.
'''
import numpy as np


class DeepBeliefNet:

    def __init__(self, *dims):
        '''
        Instantiate a Deep Belief Network. Usage:

            DeepBeliefNet(num_visible, num_hidden_1, num_hidden_2, ...)

        I.e., the args are just a list of layer dimensions, starting with
        visible and going deeper.
        '''
        assert(len(dims) > 1)
        # Store dimensions
        self.dims = dims
        self.num_visible, self.hidden_dims = dims
        
        # Model state
        self.units = [np.zeros(d) for d in dims]
        # We need fresh biases for each part of the stack.
        self.biases = [(np.zeros(d), np.zeros(dd))
                       for d, dd in zip(dims[:-1], dims[1:])]
        # Connections.
        self.weights = [np.zeros(d, dd) for d, dd in zip(dims[:-1], dims[1:])]


    def gibbs_in(self, k, beta):
        '''
        Sample the `k`th layer given the state of the `k-1`th layer. I.e.,
        sampling layer `k=1` given state for layer `k=0` is equivalent to
        sampling the first hidden layer given a visible state.

        This step is performed with inverse temperature `beta`.
        '''
        dE = self.biases[k - 1][1] + self.weights[k - 1] @ self.units[k - 1]
        P = 1 / (1 + np.exp(-beta * dE))
        self.units[k] = np.where(np.random.rand(self.dims[k]) > P, 1, 0)


    def gibbs_out(self, k, beta):
        '''
        Sample the `k`th layer given the state of the `k+1`th layer. I.e.,
        sampling layer `k=0` given state for layer `k=1` is equivalent to
        sampling a new visible state given a configuration for the first
        hidden layer.

        This step is performed with inverse temperature `beta`.
        '''
        dE = self.biases[k][0] +  self.units[k + 1] @ self.weights[k]
        P = 1 / (1 + np.exp(-beta * dE))
        self.units[k] = np.where(np.random.rand(self.dims[k]) > P, 1, 0)


    def forward_pass(self, x, stop_k, beta):
        '''
        Sample all hidden layers, up to and including layer `stop_k`, after
        clamping stimulus `x` to the visible units.

        This runs at inverse temperature `beta`. Make it large for nice
        inference.
        '''
        # I feel like having this clarifies things.
        assert(0 < stop_k and stop_k < len(self.dims))
        # Clamp stimulus
        self.units[0][:] = x
        # Do the pass
        for k in range(1, stop_k + 1):
            self.gibbs_in(k, beta)


    def backward_pass(self, beta):
        '''
        Use the deep hidden state of the net to propagate through alll of the
        layers and eventually get a new visible state. Useful for sampling.
        '''
        for k in reversed(range(0, len(self.dims) - 1)):
            self.gibbs_out(k, beta)


    def train_layer(self, X, k, beta, learning_rate=0.1):
        '''
        One training epoch for just the weights between the `k-1`th layer and
        the `k`th layer, where the 0th layer is the visible one, on input `X`
        (a list-like of training examples).
        
        This just does one loop over `X`, doing a single step of contrastive
        divergence training `for x in X`. Call it a few times if you want.

        Uses learning rate `learning_rate` and inverse temperature `beta`. One
        could do annealing by nicely changing beta each time they call this
        function.
        '''
        # Scratch space
        si0 = np.empty_like(self.unit[k])
        sj0 = np.empty_like(self.unit[k])
        si1 = np.empty_like(self.unit[k])
        sj1 = np.empty_like(self.unit[k])
        sisj0 = np.empty_like(self.weights[k - 1])
        sisj1 = np.empty_like(self.weights[k - 1])
        for x in X:
            # Clamping `x`:
            if k > 1:
                # We are training a deep layer, so propagate the training
                # example.
                self.forward_pass(x, k - 1, beta)
            else:
                # Otherwise we can just clamp the stimulus.
                self.units[0][:] = x

            # Gibbs visible->hidden
            self.gibbs_in(k, beta)

            # Sample $\langle s_i s_j\rangle_{Q^0}$
            sisj0[:] = np.outer(self.units[k - 1], self.units[k])
            # And this is similar, for the biases
            si0[:] = self.units[k - 1]
            sj0[:] = self.units[k]

            # Gibbs reconstruction, and another visible->hidden
            self.gibbs_out(k - 1, beta)
            self.gibbs_in(k, beta)

            # Sample $\langle s_i s_j\rangle_{Q^1}$
            sisj1[:] = np.outer(self.units[k - 1], self.units[k])
            si1[:] = self.units[k - 1]
            sj1[:] = self.units[k]

            # Weight update for this example.
            self.weights[k - 1]   += learning_rate * (sisj0 - sisj1)
            self.biases[k - 1][0] += learning_rate * (si0 - si1)
            self.biases[k - 1][1] += learning_rate * (sj0 - sj1)





