from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        ### unstable because exp may equals to 0
        ### make result -inf
        # maxn = array_api.max(Z, axis=1, keepdims=True)
        # Z = Z - maxn
        # exp = array_api.exp(Z)
        # print(array_api.sum(exp, axis=1, keepdims=True))
        # return array_api.log(exp / array_api.sum(exp, axis=1, keepdims=True))
    
        ### stable
        self.maxn = array_api.max(Z, axis=1, keepdims=True)
        maxn_shrink = array_api.max(Z, axis=1, keepdims=True)
        # print(self.axes)
        return Z - (array_api.log(array_api.sum(
            array_api.exp(Z - self.maxn), axis=1, keepdims=True)) 
            + maxn_shrink)
        ### END YOUR SOLUTION

    # what the hell I can't do it correctly
    # ohohoh fuck you fuck you!!!! I know
    # in the process it go through a broadcast,
    # so it need to sum the out_grad
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        maxn = array_api.max(lhs.realize_cached_data(), axis=1, keepdims=True)
        pre_shape = list(lhs.shape)
        pre_shape[1] = 1
        exp_lhs = exp(lhs - Tensor(maxn))
        sum_exp = summation(exp_lhs, 1).reshape(pre_shape).broadcast_to(lhs.shape)
        out_grad_sum = summation(out_grad,1).reshape(pre_shape).broadcast_to(lhs.shape)
        return out_grad - out_grad_sum * exp_lhs / sum_exp
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            self.axes = (self.axes,)
        self.maxn = array_api.max(Z, axis=self.axes, keepdims=True)
        maxn_shrink = array_api.max(Z, axis=self.axes, keepdims=False)
        # print(self.axes)
        return (array_api.log(array_api.sum(
            array_api.exp(Z - self.maxn), axis=self.axes, keepdims=False)) 
            + maxn_shrink)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        pre_shape = list(lhs.shape)
        for i in self.axes:
            pre_shape[i] = 1
        exp_lhs = exp(lhs - Tensor(self.maxn))
        sum_exp = summation(exp_lhs, self.axes)
        grad_middle = broadcast_to(reshape(out_grad / sum_exp, pre_shape), lhs.shape)
        return grad_middle * exp_lhs
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

