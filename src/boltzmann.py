'''
Implementing the machine and training procedure described by Ackley, Hinton,
and Sejnowski in their 1985 paper 'A Learning Algorithm for Boltzmann
Machines'.

Here we allow the API of a Hopfield net, i.e. the content-addressible memory
API of initializing the visible units to some content, but not clamping them,
so that the network will run to an equilibrium "remembered state".

We also allow the "image completion" API of clamping some units and iterating
the rest of the network until it reconciles the rest of its units with the
clamped ones.

This second interface can also be understood in an input-output fashion by
considering the clamped units as inputs and the remaining visible units as
the outputs of the model, and being careful to always clamp inputs and train
appropriately.

To toggle between these two APIs, look for `clamp_indices` arguments to
sampling functions. The visible units at these indices will be held fixed.
'''
import numpy as np
import os.path
from tqdm import tqdm


class BoltzmannMachine:

    def __init__(self, n_h, n_v):
        '''
        Initialize a Boltzmann machine with `n_h` hidden units and `n_v`
        visible units. This does not train the machine, call the appropriate
        method for that when you want.

        @param n_h  int     number of hidden units
        @param n_v  int     number of visible units
        '''
        # Store values
        self.n_h = n_h
        self.n_v = n_v

        # Placeholders for the units. Set the `true unit` (see pp. 150-151)
        self.units = np.zeros(n_h + n_v + 1)
        self.h = self.units[:n_h]
        self.v = self.units[n_h:n_h + n_v]
        self.units[-1] = 1.

        # Weights matrix. Having a true unit means this absorbs the thresholds.
        # Dims chosen so that right-multiplication by `self.units` spits out
        # the energy difference of each visible and hidden unit.
        self.W = np.zeros((n_h + n_v, n_h + n_v + 1))

        # Get a view over the thresholds (last column)
        self.theta = self.W[:, -1]

        # Implementation details.
        self.cache = np.zeros_like(self.units)
        self.hidden_inds = list(range(self.n_h))
        self.all_inds = list(range(self.n_h + self.n_v))



    @staticmethod
    def default_annealing_schedule(N=100):
        '''
        An annealing is an iterable of inverse temperatures a.k.a.
        thermodynamic betas. Training algorithms can just loop through these
        betas.

        @param N    int     how many betas to yield
        '''
        for k in range(N):
            yield 0.01 * k


    def hopfield_sampling(self, x, clamp_indices=(), max_iters=100):
        '''
        This is a deterministic sampling rule that just goes straight down the
        energy gradient. This is the rule described in section 2.1 of the paper
        and it's basically the rule used to sample in Hopfield nets.

        So, don't use it lol. Just putting it here to help me work through
        things.

        This does not have a restricted flavor and does everything in a fixed
        order.

        @param x
               np.array     visible units will start at these values
        @param clamp_indices
               Array-like   don't change the values of these visible units
        @param max_iters
               int          how long to run for if we don't converge
        '''

        # Start hidden units with Bernoulli IC. is this best practice?
        self.h[:] = np.random.binomial(1, 0.5, self.n_h)

        # Set visible units to input x
        self.v[:] = x

        # What units are changing? Not the clamped visible ones, and
        # not the true unit.
        update_inds = list(j for j in range(self.n_h + self.n_v)
                           if j - self.n_h not in clamp_indices)

        # Start stepping.
        for _ in range(max_iters):
            self.cache[:] = self.units

            # See where energy is positive after thresholding
            activations = np.where(
                self.W @ self.units > 0.,
                1., 0.)

            # Deterministically update unclamped units
            self.units[update_inds] = activations[update_inds]

            # Convergence?
            if (self.cache == self.units).all():
                break

        return self.v


    def _boltzmann_update(self, beta, update_inds):
        '''
        Helper for `boltzmann_sampling`. Go call that. This handles one
        stochastic step down the information gain gradient.
        '''
        # First we need to compute our probabilities:
        # $p_k = \frac{1}{1 + \exp(-beta \Delta E_k)}$
        # where, recall that since we have a true unit, if $S_i$ is the
        # state of the $i$th unit,
        # $\Delta E_k = \sum_i W_{ki} S_i$.
        dE = self.W @ self.units
        P = 1. / (1 + np.exp(-beta * dE))

        # Now update each unit ~ Bernoulli($p_k$).
        activations = np.where(
            np.random.rand(self.n_h + self.n_v) < P,
            1, 0)
        self.units[update_inds] = activations[update_inds]


    def boltzmann_sampling(self, x=None, clamp_indices=(), N=100,
                           betas=None):
        '''
        Metropolis-style sampling algorithm with annealing as described in
        section 2.2. This does not enforce any kind of restricted structure.
        Note that `betas` is the annealing schedule, and should be a function
        taking in an integer representing how many steps to take and spitting
        out some iterable of inverse temperatures.

        @param x
            np.array            the input to the visible units.
        @param clamp_indices
            Array-like of int   visible units to hold fixed.
        @param N
            int                 number of training steps to take
        @param betas
            function int -> iterable of inverse temperatures
            the annealing schedule.
        '''
        # Set visible units to input x if present, and set
        # other units with Bernoulli IC. Shouldn't matter for suitable
        # annealing schedule, since infinite temperature gives uniform distro.
        if x:
            self.v[:] = x
            self.h[:] = np.random.binomial(1, 0.5, self.n_h)
        else:
            self.units[:-1] = np.random.binomial(1, 0.5, self.n_h + self.n_v)

        # Annealing sched
        if not betas:
            betas = BoltzmannMachine.default_annealing_schedule

        # As above...
        update_inds = list(j for j in range(self.n_h + self.n_v)
                           if j - self.n_h not in clamp_indices)

        # Run sampler. Skip the convergence checks because it's stochastic.
        for beta in betas(N):
            self._boltzmann_update(beta, update_inds)

        # That's it!
        return self.v

    def train_gen(self, X, epochs, chains=5, burn_in=5, lr=0.5, save_dir=None,
                  name='boltzmann'):
        for e in range(epochs):
            print('epoch', e)
            # We do annealing in the learning rate per the authors.
            epsilon = lr * epochs / (e + 1)
            print(epsilon)

            # Approximate $p_{ij}$ using `chains` different chains at each
            # example x to sample these mean probabilities.
            # "It's just Monte Carlo"^{TM}
            # Recall that if $y$ indexes over possible hidden states,
            # $p_{ij} = \sum_{x,y} P(x, y) S^{x,y}_i S^{x,y}_j$,
            # where the $S$ terms give states of units at $x$ and $y$.
            np.random.shuffle(X)
            for x in tqdm(X, mininterval=1):
                # These store the main ingredients for computing weight updates
                # Look at the appendix of the paper to see definitions. They
                # are called $p_{ij}$ and $p_{ij}'$ in there.
                p_clamped = np.zeros_like(self.W)
                p_free = np.zeros_like(self.W)

                # Run the model clamped to x a bunch of times. Doing some
                # extra annealing here to shake up the chain between
                # observations, but not sure if that's advisable, or what
                # sort of schedule is best for that.
                for _ in range(chains):
                    self.v[:] = x
                    for b in range(burn_in):
                        beta = epsilon + burn_in - b
                        self._boltzmann_update(beta, self.hidden_inds)
                    for b in range(burn_in):
                        self._boltzmann_update(epsilon, self.hidden_inds)
                    p_clamped += np.outer(self.units, self.units)[:-1,:]
                p_clamped /= chains

                # Same but running free.
                for _ in range(chains):
                    self.v[:] = x
                    for b in range(burn_in):
                        beta = epsilon + burn_in - b
                        self._boltzmann_update(beta, self.all_inds)
                    for b in range(burn_in):
                        self._boltzmann_update(epsilon, self.all_inds)
                    p_free += np.outer(self.units, self.units)[:-1,:]
                p_free /= chains

                # Do the update for this example
                self.W += epsilon * (p_clamped - p_free)

            # Save this epoch's new weights.
            if save_dir:
                np.savez_compressed(os.path.join(save_dir, name + '.npz'))
            yield e


    def train(self, X, epochs, chains=5, burn_in=5, lr=0.5, save_dir=None,
              name='boltzmann'):
        '''
        We implement an extrapolation of the algorithm described in section
        3 of the paper. This is similar to contrastive divergence, but not
        really the same I don't think? Hmm.

        '''
        for _ in self.train_gen(X, epochs, chains, burn_in, lr, save_dir, name):
            pass

