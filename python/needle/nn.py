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
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        init_weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        init_bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
        self.weight = Parameter(init_weight)
        self.bias = Parameter(ops.transpose(init_bias))

    def forward(self, X: Tensor) -> Tensor:
        y = ops.matmul(X, self.weight)
        if self.has_bias:
            y = ops.add(y, ops.broadcast_to(self.bias, y.shape))
        return y


class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.modules:
            x = layer(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        lse = ops.logsumexp(logits, 1)
        y_one_hot = init.one_hot(logits.shape[-1], y)
        zy = ops.summation(logits * y_one_hot, 1)
        return ops.summation(lse - zy) / logits.shape[0]    


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)

    def forward(self, x: Tensor) -> Tensor:
        ex, dx = self.norm_train(x) if self.training else self.norm_test(x)
        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return w * (x - ex) / (dx + self.eps) ** 0.5 + b

    def norm_train(self, x: Tensor) -> Tensor:
        ex = ops.summation(x, 0) / x.shape[0]
        self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * ex).detach()
        ex = ops.broadcast_to(ops.reshape(ex, (1, self.dim)), x.shape)
        dx = ops.summation((x - ex) ** 2, 0) / x.shape[0]
        self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * dx).detach()
        dx = ops.broadcast_to(ops.reshape(dx, (1, self.dim)), x.shape)
        return ex, dx

    def norm_test(self, x: Tensor) -> Tensor:
        ex = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
        dx = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape)
        return ex, dx


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        ex = ops.summation(x, 1) / x.shape[1]
        ex = ops.broadcast_to(ops.reshape(ex, (x.shape[0], 1)), x.shape)
        dx = ops.summation((x - ex) ** 2, 1) / x.shape[1]
        dx = ops.broadcast_to(ops.reshape(dx, (x.shape[0], 1)), x.shape)
        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return w * (x - ex) / (dx + self.eps) ** 0.5 + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.flatten = Flatten()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            y = init.randb(*self.flatten(x).shape, p=1 - self.p)
            x = x * ops.reshape(y, x.shape) / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
