import math
import numpy as np
from chainer import optimizers, Variable
import chainer.functions as F
from chainer import serializers

###
# Buffer to store results

class Buffer(object):
    """
    This buffer either stores numpy arrays or chainer variables. In the latter case, it creates a dictionary {1: .., 2: ...}
    which stores the chainer variables using an index variable as key
    """

    def __init__(self, n):

        self.idx = 0
        self.n = n
        self.data = {}

    def add(self, *args):
        """
        Add key, value pairs where values are numpy scalar, numpy array or chainer Variable
        """

        for i in range(0,len(args),2):

            key = args[i]
            value = args[i+1]

            if not self.data.has_key(key):
                if isinstance(value, Variable):
                    self.data[key] = {}
                else:
                    self.data[key] = np.zeros([self.n, value.size], dtype=value.dtype)

            self.data[key][self.idx] = value

    def get(self, key, i):
        """
        Return value from buffer

        :param key: property to return
        :param i: index to return counting backwards from last filled index
        :return: value
        """

        return self.data[key][self.idx-1-i]

    def increment(self):
        self.idx += 1

    def reset(self):

        self.idx = 0
        self.data = {}


###
# Base class for an RL agent

class Agent(object):

    def __init__(self, model, **kwargs):

        # discounting factor
        self.gamma = kwargs.get('gamma', 0.99)

        # the 'brain' of an agent
        self.model = model

        # define optimizer
        self.optimizer = kwargs.get('optimizer', optimizers.Adam())

        # setup optimizer
        self.optimizer.setup(self.model)

    def act(self, obs):
        """
        :param obs: observation
        :return: action
        """

        return self.model.action(obs)

    def run(self, env, niter):
        """
        Run agent on environment for niter iterations

        :param env: Environment
        :param niter: Number of iterations
        :return: dictionary of results
        """

        # initialize results
        result = Buffer(niter)

        # initialize environment and get first observation
        obs = env.reset()

        # reset agent's brain
        self.model.reset()

        # run agent on environment
        for t in xrange(niter):

            result.add('observation',obs)

            # generate action using actor model
            action, pi, v, internal = self.act(obs)

            # get feedback from environment
            obs, reward, terminal = env.step(action)

            # add to results
            result.add('action',action,'policy',pi.data,'value',v.data,'reward',reward,'terminal',terminal)
            for key in internal.keys():
                result.add(key,internal[key])

            if terminal:
                self.model.reset()

            result.increment()

        return result.data

    def save(self, name):
        """
        Save model
        """

        serializers.save_npz('models/{0}.model'.format(name), self.model)

    def load(self, name):
        """
        Load model
        """

        serializers.load_npz('models/{0}.model'.format(name), self.model)


class Advantage_Actor_Critic(Agent):

    def __init__(self, model, **kwargs):
        super(Advantage_Actor_Critic, self).__init__(model, **kwargs)

        # maximal number of steps to accumulate information
        self.t_max = kwargs.get('t_max', 10)

        # contribution of the entropy term
        self.beta = 1e-2

    def learn(self, env, niter):
        """
        Train agent on environment for niter iterations

        :param env: Environment
        :param niter: Number of iterations
        :return: dictionary of results
        """

        # initialize results buffer
        result = Buffer(niter)

        # initialize buffer for backpropagation
        past = Buffer(self.t_max)

        # initialize environment and get first observation
        obs = env.reset()

        # reset agent's brain
        self.model.reset()

        # run agent on environment
        counter = 0
        for t in xrange(niter):

            if np.mod(t,niter/100) == 0:
                print str(t) + '/' + str(niter)

            self.model.unchain_backward()

            result.add('observation',obs)

            # generate action using actor model
            action, pi, v, internal = self.act(obs)

            # store log policy data
            _score_function = self.score_function(action, pi)

            # compute entropy
            _entropy = self.entropy(pi)

            # get feedback from environment
            obs, reward, terminal = env.step(action)

            # add to results
            result.add('score_function',_score_function.data,'entropy',_entropy.data,
                       'value',v.data,'action',action,'policy',pi.data,'reward',reward,'terminal',terminal)
            for key in internal.keys():
                result.add(key,internal[key])

            # add to past buffer
            past.add('score_function',_score_function,'entropy',_entropy,'value',v,'reward',reward)

            result.increment()
            past.increment()

            if terminal:
                self.model.reset()

            # initiate learning based on accumulated information
            if terminal or counter == (self.t_max - 1) or t == niter-1:

                if terminal:
                    R = 0
                else:
                    x = self.model.get_persistent()
                    _, vout, _ = self.model(obs)
                    self.model.set_persistent(x)
                    R = float(vout.data)

                pi_loss = v_loss = 0
                for i in range(0, counter+1):

                    R = past.get('reward',i) + self.gamma * R

                    v = past.get('value',i)

                    # Compute advantage (difference between approximation of action-value and value)
                    advantage = R - v

                    # get log probability
                    score_function = past.get('score_function',i)

                    # Compute entropy
                    entropy = past.get('entropy', i)

                    # Log policy (probability of action given observations) is increased proportionally to advantage
                    pi_loss -= score_function * float(advantage.data)

                    # loss is reduced by high entropy (stochastic) policies
                    pi_loss -= self.beta * entropy

                    # Get squared difference between accumulated reward and value function
                    v_loss += advantage ** 2

                v_loss = F.reshape(v_loss, pi_loss.data.shape)

                # Compute total loss; 0.5 supposedly used by Mnih et al
                loss = pi_loss + 0.5 * v_loss

                # Compute gradients
                self.model.zerograds()
                loss.backward()

                # update the shared model
                self.optimizer.update()

                # reset buffer
                past.reset()

                counter = 0

            else:

                counter += 1

        return result.data

    def score_function(self, action, pi):

        # convert 1-hot encoding to discrete action
        action = np.argmax(action)

        logp = F.log_softmax(pi)
        return F.select_item(logp, Variable(np.asarray([action], dtype=np.int32)))

    def entropy(self, pi):

        p = F.softmax(pi)
        logp = F.log_softmax(pi)

        return - F.sum(p * logp, axis=1)