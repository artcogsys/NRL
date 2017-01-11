import numpy as np
import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear
from chainer import Chain, Variable
import chainer.initializers as init
import chainer.functions as F
import chainer.links as L

###
# Implementation of custom layers

class ElmanBase(link.Chain):

    def __init__(self, n_units, n_inputs=None, initU=None,
                 initW=None, bias_init=0):
        """

        :param n_units: Number of hidden units
        :param n_inputs: Number of input units
        :param initU: Input-to-hidden weight matrix initialization
        :param initW: Hidden-to-hidden weight matrix initialization
        :param bias_init: Bias initialization
        """
        if n_inputs is None:
            n_inputs = n_units
        super(ElmanBase, self).__init__(
            U=linear.Linear(n_inputs, n_units,
                            initialW=initU, initial_bias=bias_init),
            W=linear.Linear(n_units, n_units,
                            initialW=initW, initial_bias=bias_init, nobias=True),
        )

class Elman(ElmanBase):
    """
    Implementation of simple Elman layer
    """

    def __init__(self, in_size, out_size, initU=None,
                 initW=None, bias_init=0):
        super(Elman, self).__init__(
            out_size, in_size, initU, initW, bias_init)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(Elman, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Elman, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.U(x)
        if self.h is not None:
            z += self.W(self.h)

        self.h = relu.relu(z)

        return self.h

###
# Implementation of neural networks

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