import numpy

import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear

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
                            initialW=initW, initial_bias=bias_init),
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
        if self.xp == numpy:
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