import matplotlib.pyplot as plt
import numpy as np

class Analysis(object):

    def __init__(self, fname, env, agent):

        self.fname = fname
        self.env = env
        self.agent = agent

    def cumulative_reward(self, reward):
        """
        Plot cumulative reward throughout the experiment

        :param fname: file name
        :param reward: n_time x 1 ndarray
        """

        t = range(len(reward))

        plt.clf()
        plt.plot(t, np.cumsum(reward), 'k')
        plt.xlabel('Time')
        plt.ylabel('Cumulative reward')
        plt.savefig('figures/' + self.fname + '_cumulative_reward.png')

    def weight_matrix(self, W):
        """
        Plot weight matrix

        :param fname: file name
        :param W: N x M weight matrix
        """

        plt.clf()
        plt.pcolor(W)
        plt.title('Weight matrix')
        plt.savefig('figures/' + self.fname + '_weight_matrix.png')

    def functional_connectivity(self, x):
        """
        Plot functional connectivity matrix (full correlation)

        :param fname: file name
        :param x: T x M timeseries data
        """

        M = np.corrcoef(x.transpose())

        plt.clf()
        plt.pcolor(M)
        plt.title('Functional connectivity')
        plt.savefig('figures/' + self.fname + '_functional_connectivity.png')

    def spike_rate(self, fname, neurons, terminal):
        """
        Create a spike rate plot for a bunch of neurons. trial boundaries are determined by terminal

        Requires time constant interpretation!

        :param fname: file name
        :param neurons: n_time x n_neurons ndarray
        :param terminal: n_time x 1 ndarray
        """

        pass