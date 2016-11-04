from chainer import Chain, Variable
import chainer.initializers as init
import chainer.functions as F
import chainer.links as L
from elman import Elman
import numpy as np

class RNN_Elman(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN_Elman, self).__init__(
            l1_pi=Elman(ninput, nhidden),
            l1_v=Elman(ninput, nhidden),
            pi=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
            v=L.Linear(nhidden, 1, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

    def __call__(self, x):

        h_pi = self.l1_pi(x)
        h_v = self.l1_v(x)

        pi = self.pi(h_pi)
        v = self.v(h_v)

        return pi, v, {'hidden_pi': h_pi.data[0], 'hidden_v': h_v.data[0]}

    def reset(self):
        self.l1_pi.reset_state()
        self.l1_v.reset_state()

    def get_persistent(self):
        return [self.l1_pi.h, self.l1_v.h]

    def set_persistent(self, x):
        self.l1_pi.h = x[0]
        self.l1_v.h = x[1]

    def action(self, obs):
        """
        Generate action

        :param obs:
        :return: action, pi, v, internal
        """

        pi, v, internal = self(obs)

        # generate action according to policy
        p = F.softmax(pi).data[0]

        # normalize p in case tiny floating precision problems occur
        p = p.astype('float32')
        p /= p.sum()

        # one-out-of-K representation
        idx = np.random.choice(self.noutput, None, True, p)
        action = np.zeros(self.noutput)
        action[idx] = 1.0

        return action, pi, v, internal

    def unchain_backward(self):
        if not self.l1_pi.h is None:
            self.l1_pi.h.unchain_backward()
        if not self.l1_v.h is None:
            self.l1_v.h.unchain_backward()

    def policy_weights(self):
        """
        Return weights of the policy network
        :return: input weights U and hidden weights W
        """
        return [self.l1_pi.U.W.data, self.l1_pi.W.W.data]

    def value_weights(self):
        """
        Return weights of the value network
        :return: input weights U and hidden weights W
        """
        return [self.l1_v.U.W.data, self.l1_v.W.W.data]

class RNN_GRU(Chain):
    """
    Implements a GRU-based RNN
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN_GRU, self).__init__(
            l1_pi=L.StatefulGRU(ninput, nhidden),
            l1_v=L.StatefulGRU(ninput, nhidden),
            pi = L.Linear(nhidden, noutput, initialW=init.HeNormal()),
            v = L.Linear(nhidden, 1, initialW=init.HeNormal()),
       )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

    def __call__(self, x):
        """
        http://www.felixgers.de/papers/phd.pdf
        http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/lstm.html

        """

        h_pi = self.l1_pi(x)
        h_v = self.l1_v(x)

        pi = self.pi(h_pi)
        v = self.v(h_v)

        return pi, v, {'hidden_pi': h_pi.data[0], 'hidden_v': h_v.data[0]}

    def reset(self):
        self.l1_pi.reset_state()
        self.l1_v.reset_state()

    def get_persistent(self):
        return [self.l1_pi.h, self.l1_v.h]

    def set_persistent(self, x):
        self.l1_pi.h = x[0]
        self.l1_v.h = x[1]

    def action(self, obs):
        """
        Generate action

        :param obs:
        :return: action, pi, v, internal
        """

        pi, v, internal = self(obs)

        # generate action according to policy
        p = F.softmax(pi).data[0]

        # normalize p in case tiny floating precision problems occur
        assert (np.sum(p) > 0.999999)
        p = p.astype('float64')
        p /= p.sum()

        # one-out-of-K representation
        idx = np.random.choice(self.noutput, None, True, p)
        action = np.zeros(self.noutput)
        action[idx] = 1.0

        return action, pi, v, internal

    def unchain_backward(self):
        if not self.l1_pi.h is None:
            self.l1_pi.h.unchain_backward()
        if not self.l1_v.h is None:
            self.l1_v.h.unchain_backward()



# class Elman(Chain):
#     """
#     Elman network
#
#     We use separate policy and value networks
#
#     """
#
#     def __init__(self, ninput, nhidden, noutput):
#         super(Elman, self).__init__(
#             l1_pi=L.Linear(ninput+nhidden, nhidden, wscale=np.sqrt(2)),
#             l1_v=L.Linear(ninput+nhidden, nhidden, wscale=np.sqrt(2)),
#             pi = L.Linear(nhidden, noutput, initialW=init.HeNormal()),
#             v = L.Linear(nhidden, 1, initialW=init.HeNormal()),
#         )
#
#         self.ninput = ninput
#         self.nhidden = nhidden
#         self.noutput = noutput
#
#         self.topdown_pi = Variable(np.zeros([1,nhidden]).astype(np.float32), volatile='auto')
#         self.topdown_v = Variable(np.zeros([1,nhidden]).astype(np.float32), volatile='auto')
#
#         # Required? => WE SHOULD IMPLEMENT A SIMPLE ELMAN LAYER IF CHAINER DOES NOT
#         #self.add_persistent('topdown1', self.topdown1)
#
#     def __call__(self, x, persistent=False):
#
#         h_pi = F.relu(self.l1_pi(F.concat([x, self.topdown_pi])))
#         h_v = F.relu(self.l1_v(F.concat([x, self.topdown_v])))
#
#         pi = self.pi(h_pi)
#         v = self.v(h_v)
#
#         self.topdown_pi = h_pi
#         self.topdown_v = h_v
#
#         return pi, v, {'hidden_pi': h_pi.data[0], 'hidden_v': h_v.data[0]}
#
#     def reset(self):
#         """
#         Reset persistent state
#         """
#         self.topdown_pi = Variable(np.zeros(self.topdown_pi.data.shape).astype(np.float32), volatile='auto')
#         self.topdown_v = Variable(np.zeros(self.topdown_v.data.shape).astype(np.float32), volatile='auto')
#
#     def get_persistent(self):
#         return [self.topdown_pi, self.topdown_v]
#
#     def set_persistent(self, x):
#         self.topdown_pi = x[0]
#         self.topdown_v = x[1]
#
#     def action(self, obs):
#         """
#         Generate action
#
#         :param obs:
#         :return: action, pi, v, internal
#         """
#
#         pi, v, internal = self(obs)
#
#         # generate action according to policy
#         p = F.softmax(pi).data[0]
#
#         # normalize p in case tiny floating precision problems occur
#         p = p.astype('float32')
#         p /= p.sum()
#
#         # one-out-of-K representation
#         idx = np.random.choice(self.noutput, None, True, p)
#         action = np.zeros(self.noutput)
#         action[idx] = 1.0
#
#         return action, pi, v, internal
#
#     def unchain_backward(self):
#         """
#         Do we need to implement something here?
#         """
#
#         if not self.topdown_pi is None:
#             self.topdown_pi.unchain_backward()
#         if not self.topdown_v is None:
#             self.topdown_v.unchain_backward()
#         # if not self.l1_pi is None:
#         #     self.l1_pi.unchain_backward()
#         # if not self.l1_v is None:
#         #     self.l1_v.unchain_backward()