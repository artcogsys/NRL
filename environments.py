import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
from scipy.io import savemat

###
# Base class for an environment

class Environment(object):

    def reset(self):
        """
        Reset environment and return observation

        Returns: observation

        """

        pass

    def step(self, action):
        """
        Take one step based on an agent's action

        Args:
            action:

        Returns: observation

        """

        pass

    def get_state(self):
        """
        Returns: ground truth state of the environment
        """

        return self.state

    def set_state(self, state):
        """
        :param: state : sets ground truth state of the environment
        """

        self.state = state

###
# Specific environments

class ProbabilisticCategorization(Environment):
    """

    Let odds be a vector determining the odds ratio for emitting a certain symbol x = i given state k = j:

        odds = [ P(x = 0 | k = 1) / P(x = 0 | k = 0) ... P(x = n | k = 1) / P(x = n | k = 0) ]

    We define

        p = odds / sum(odds)
        q = (1/odds) / sum(1/odds)

    Let P(x = i | p, q, k) define the probability that the emitted symbol is i given that we have probability vector p and q and
    the true state can be either k = 0 or k = 1. Then

        P(x = i | p, k) = p^k * q^(k-1)

    Note: vector of zeros indicates absence of evidence (starting state)

    """

    def __init__(self, odds = [0.25, 0.75, 1.5, 2.5]):
        """

        :param: odds : determines odds ratio

        """

        super(ProbabilisticCategorization, self).__init__()

        self.odds = np.array(odds)

        self.p = self.odds/float(np.sum(self.odds))
        self.q = (1.0/self.odds)/float(np.sum(1.0/self.odds))

        self.ninput = len(self.p)
        self.noutput = 3 # number of output variables

        self.rewards = [-1, 15, -100]

        # normalize rewards
        self.rewards = np.array(self.rewards, dtype='float32') / np.max(np.abs(self.rewards)).astype('float32')

        self.reset()

    def reset(self):
        """

        Returns: observation

        """

        self.state = np.random.randint(1, 3)  # 1 = left, 2 = right

        obs = np.zeros([1, self.ninput], dtype='float32')

        return obs


    def render(self):

        print self.state


    def step(self, action):

        # convert 1-hot encoding to discrete action
        action = np.argmax(action)

        if action == 0:  # wait to get new evidence

            reward = self.rewards[0]

            # choose piece of evidence
            if self.state == 1:
                evidence = np.random.choice(self.ninput, p = self.p)
            else:
                evidence = np.random.choice(self.ninput, p = self.q)

            obs = np.zeros([1, self.ninput], dtype='float32')
            obs[0, evidence] = 1

            terminal = np.float32(0)

        else:  # left or right was chosen

            if action == self.state:
                reward = self.rewards[1]
            else:
                reward = self.rewards[2]

            terminal = np.float32(1)

            obs = self.reset()

        return obs, reward, terminal