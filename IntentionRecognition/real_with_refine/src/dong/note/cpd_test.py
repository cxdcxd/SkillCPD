from functools import partial
import matplotlib.pyplot as plt
from pycpd import AffineRegistration
import numpy as np


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    X = np.loadtxt('data_half.txt')      # target dataset
    Y = np.loadtxt('repro_half.txt')     # source dataset (moving)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    reg.register()
    plt.show()
    ####
    res_B, res_t = reg.get_registration_parameters()
    print(res_B)
    print(res_t)
    ####


if __name__ == '__main__':
    main()
