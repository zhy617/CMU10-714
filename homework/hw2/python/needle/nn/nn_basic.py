"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = init.kaiming_uniform(
            self.in_features, self.out_features, device=self.device, dtype=self.dtype
        )
        print(self.weight.shape)
        if bias:
            self.bias:Tensor = (init.kaiming_uniform(
                self.out_features, 1, device=self.device, dtype=self.dtype
            )).transpose()
        else:
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias is not None:
            shape = (X.shape[0], self.out_features)
            return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias,shape) 
        else:
            return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        shape = list(X.shape)
        last_shape = 1
        for i in range(1, len(shape)):
            last_shape *= shape[i]
        return X.reshape((shape[0], last_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        one_hots = ops.broadcast_to(one_hot, logits.shape)
        # print(one_hots)
        return (ops.summation(ops.logsumexp(logits, axes = -1)) - ops.summation(logits * one_hots)) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(self.dim, device=device, dtype=dtype)
        self.bias = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_mean:Tensor = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var:Tensor = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean: Tensor = ops.summation(x, axes=0) / x.shape[0]
        self.running_mean = ((1 - self.momentum) * self.running_mean 
                             + mean * self.momentum)
        mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)

        var: Tensor = ops.summation((x-mean) ** 2, axes=0) / x.shape[0]
        self.running_var = ((1 - self.momentum) * self.running_var 
                            + var * self.momentum)
        var = var.reshape((1, self.dim)).broadcast_to(x.shape)

        if self.training == False:
            running_mean = (self.running_mean.reshape((1, self.dim))
                            .broadcast_to(x.shape))

            running_var = (self.running_var.reshape((1, self.dim))
                           .broadcast_to(x.shape))
            x = (x - running_mean) / ((running_var + self.eps)**0.5)
        else:
            x = (x - mean) / ((var + self.eps)**0.5)
        
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return x * weight + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(dim, device=device, dtype=dtype)
        self.bias = init.zeros(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print(x)
        mean: Tensor = ops.summation(x, axes=1) / x.shape[1]
        mean = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var: Tensor = ops.summation((x-mean) ** 2, axes=1) / x.shape[1]
        var = var.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # print(mean)
        # print(var)
        x = (x - mean) / ((var + self.eps)**0.5)
        # print(x)
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return x * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*(x.shape), p=(1-self.p), device=x.device, dtype="bool") / (1 - self.p)
            x = x * mask
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
