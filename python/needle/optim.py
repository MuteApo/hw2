"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

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
        for param in self.params:
            grad = ndl.Tensor(param.grad + self.weight_decay * param.data, dtype=param.dtype)
            u_next = (self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad).detach()
            self.u[param] = u_next
            param.data = param - self.lr * u_next


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

        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            grad = ndl.Tensor(param.grad + self.weight_decay * param.data, dtype=param.dtype)
            u_next = (self.beta1 * self.u.get(param, 0) + (1 - self.beta1) * grad).detach()
            v_next = (self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad ** 2).detach()
            self.u[param] = u_next
            self.v[param] = v_next
            u_next = (u_next / (1 - self.beta1 ** self.t)).detach()
            v_next = (v_next / (1 - self.beta2 ** self.t)).detach()
            param.data = param - self.lr * u_next / (v_next ** 0.5 + self.eps)
