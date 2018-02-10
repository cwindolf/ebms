from src.datasets import caltech
from src.boltzmann import BoltzmannMachine
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Try to get the Boltzmann machine to hallucinate Caltech 101 Silhouettes.
    # http://people.cs.umass.edu/~marlin/data.shtml
    
    data = np.array(list(caltech()))

    # *********************************************************************** #
    # Init and train machine

    machine = BoltzmannMachine(20, 28 * 28)
    for e in machine.train_gen(data, 100):
        fig, ax = plt.subplots()
        ax.imshow(machine.boltzmann_sampling().reshape((28, 28)),
                  interpolation=None)
        ax.set_title('epoch %d' % e)
        plt.show()
        plt.close('all')
