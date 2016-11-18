import numpy as np
from chainer import Variable
import chainer.functions as F

###
# Base class for a task

class Task(object):

    def reset(self):
        """
        Reset task and return observation

        Returns: observation

        """

        pass

    def step(self, action):
        """
        Take one step based on an agent's action

        Args:
            action:

        Returns: observation, reward, terminal, target

        Note:
            target is used to train a model using a supervised agent. It depends on the task whether or
            not it is available. The representation should be acceptable by the task's loss function

        """

        pass

    def get_state(self):
        """
        Returns: ground truth state of the task
        """

        return self.state

    def set_state(self, state):
        """
        :param: state : sets ground truth state of the task
        """

        self.state = state

    def loss(self, x, t):
        """

        Loss function in case of SupervisedAgent

        :param x: predicted action
        :param t: target action
        :return: loss
        """
        pass

###
# Specific environments

class ProbabilisticCategorizationTask(Task):
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

    Note: a nicer way to generalize this to 2D input is to have the 2D input be a very noisy version of the underlying stimulus
          this makes it an object categorization task. We then have one knob to turn (noise level)

    Note: This code has now been generalized so it can also be used in supervised training. The parameter nsteps when not none indicates
          how many steps are taken in each trial

    """

    def __init__(self, odds = [0.25, 0.75, 1.5, 2.5], nsteps = None):
        """

        :param: odds : determines odds ratio

        """

        super(ProbabilisticCategorizationTask, self).__init__()

        self.odds = np.array(odds)

        # nr of steps taken in each trial (if specified)
        self.nsteps = nsteps

        # keep track of the iteration
        self.iter = 1

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

        Returns: observation and target value

        """

        self.state = np.int32(np.random.randint(1, 3))  # 1 = left, 2 = right

        obs = np.zeros([1, self.ninput], dtype='float32')

        return obs, self.state


    def render(self):

        print self.state


    def step(self, action):
        """

        :param action:
        :return: obs, reward, terminal, target
        """

        # convert 1-hot encoding to discrete action
        action = np.argmax(action)

        if action == 0 or self.iter < self.nsteps:  # wait to get new evidence

            reward = self.rewards[0]

            # choose piece of evidence
            if self.state == 1:
                evidence = np.random.choice(self.ninput, p = self.p)
            else:
                evidence = np.random.choice(self.ninput, p = self.q)

            obs = np.zeros([1, self.ninput], dtype='float32')
            obs[0, evidence] = 1

            terminal = np.float32(0)

            target = self.state

            self.iter += 1

        else:  # left or right was chosen

            self.iter = 1

            if action == self.state:
                reward = self.rewards[1]
            else:
                reward = self.rewards[2]

            terminal = np.float32(1)

            obs, target = self.reset()

        return obs, reward, terminal, target

    def loss(self, x, t):
        return F.softmax_cross_entropy(x, np.array([t]))