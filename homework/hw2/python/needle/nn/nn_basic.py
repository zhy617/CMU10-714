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
        self.weight = Parameter(init.kaiming_uniform(
            self.in_features, self.out_features, 
            device=self.device, dtype=self.dtype,
            requires_grad=True
        ))
        # print(self.weight.shape)
        if bias:
            # NOTE: notice that we must make sure
            # that the parameters' type is Parameter
            # rather than Tensor
            self.bias = Parameter(((init.kaiming_uniform(
                self.out_features, 1, 
                device=self.device, dtype=self.dtype,
                requires_grad=True
            ))).transpose())
            # print(type(self.bias))
            # self.bias = Parameter(self.bias.realize_cached_data(),dtype=dtype)
        else:
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # print("LINEAR", X.dtype)
        # assert X.dtype == "float32"
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
        # print("RELU", x.dtype)
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("SEQUENTIAL", x.dtype)
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
        self.weight = Parameter(init.ones(self.dim, 
                                          device=device, 
                                          dtype=dtype,
                                          requires_grad=True))
        
        self.bias = Parameter(init.zeros(self.dim, 
                                         device=device, 
                                         dtype=dtype,
                                         requires_grad=True))
        
        self.running_mean:Tensor = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var:Tensor = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("BATCHNORM", x.dtype)
        batch_size = np.float32(x.shape[0])
        mean: Tensor = ops.summation(x, axes=0) 
        # print("fuck", mean.dtype)
        # print("fuck", x.dtype)
        mean = mean / batch_size
        x_minus_mean = (x - mean.broadcast_to(x.shape))
        var: Tensor = ops.summation(x_minus_mean ** 2, axes=0) / batch_size
        # print("mean.dtype",mean.dtype)
        # NOTE: eval mode doesn't need momentum
        if self.training == False:
            x_minus_mean = (x - self.running_mean.broadcast_to(x.shape))
            x_std = ((self.running_var + self.eps)**0.5).broadcast_to(x.shape)
            x = x_minus_mean / x_std
        else:
            # NOTE: when encountering the iterative variable,
            # you should use .data or .detach() to disconnect the graph
            self.running_mean.data = ((1.0 - self.momentum) * self.running_mean 
                                 + mean * self.momentum)
            self.running_var.data = ((1.0 - self.momentum) * self.running_var
                         + var * self.momentum)
            x_std = ((var + self.eps)**0.5).broadcast_to(x.shape)
            x = x_minus_mean / x_std
        
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return x * weight + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, 
                                          device=device, 
                                          dtype=dtype,
                                          requires_grad=True))

        self.bias = Parameter(init.zeros(dim, 
                                         device=device, 
                                         dtype=dtype,
                                         requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("LAYERNORM", x.dtype)
        # print(x)
        feature_size = np.float32(x.shape[1])
        # print(x)
        mean: Tensor = ops.summation(x, axes=1) / feature_size
        mean = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var: Tensor = ops.summation((x-mean) ** 2, axes=1) / feature_size
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
        # print("DROPOUT", x.dtype)
        if self.training:
            # NOTE: make sure that all Tensor is the same dtype
            mask = init.randb(*(x.shape), p=(1-self.p), device=x.device, dtype=x.dtype) / (1 - self.p)
            x = x * mask
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("RESIDUAL", x.dtype)
        return self.fn(x) + x
        ### END YOUR SOLUTION
