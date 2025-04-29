"""Optimization module"""
import needle as ndl
import numpy as np
# import needle.init as init


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p not in self.u:
                self.u[p] = ndl.init.zeros_like(p)
            grad = p.grad

            # NOTE: lambda / 2 * p.data * p.data is L2 decay
            # but the grad is lambda * p.data
            if self.weight_decay != 0:
                grad = p.grad + self.weight_decay * p.data
            self.u[p] = (self.momentum * self.u[p] 
                         - (1 - self.momentum) * grad.data)
            p.data = p.data + self.lr * self.u[p]
        ### END YOUR SOLUTION

        # ### BEGIN YOUR SOLUTION
        # for theta in self.params:
        #     if theta not in self.u:
        #         self.u[theta]= ndl.init.zeros_like(theta.data)
        #     grad = theta.grad.data + self.weight_decay * theta.data
        #     self.u[theta]= self.momentum * self.u[theta]+(1- self.momentum)* grad
        #     theta.data -= self.lr * self.u[theta]
        # ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for p in self.params:
            if p not in self.m:
                self.m[p] = ndl.init.zeros_like(p)
                self.v[p] = ndl.init.zeros_like(p)
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0:
                grad = grad.data + self.weight_decay * p.data
            self.m[p] = (self.beta1 * self.m[p].data
                         + (1 - self.beta1) * grad.data)
            self.v[p] = (self.beta2 * self.v[p].data
                         + (1 - self.beta2) * (grad.data ** 2))
            
            m_hat = self.m[p].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[p].data / (1 - self.beta2 ** self.t)
            p.data = (p.data - self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps))
        
        # self.t += 1
        # for i, param in enumerate(self.params):
        #     if i not in self.m:
        #         self.m[i] = ndl.init.zeros(*param.shape)
        #         self.v[i] = ndl.init.zeros(*param.shape)
            
        #     # NOTE: param.grad maybe None
        #     if param.grad is None:
        #         continue
        #     grad_data = ndl.Tensor(param.grad.numpy(), dtype='float32').data \
        #          + param.data * self.weight_decay
        #     self.m[i] = self.beta1 * self.m[i] \
        #         + (1 - self.beta1) * grad_data
        #     self.v[i] = self.beta2 * self.v[i] \
        #         + (1 - self.beta2) * grad_data**2
        #     # NOTE: bias correction
        #     u_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
        #     v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
        #     param.data = param.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
