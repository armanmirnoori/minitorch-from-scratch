import numpy as np

class Tensor:
    def __init__(self, data, requires_grad = False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )

        def _backward():
            if self.requires_grad:
                grad = sum_to_shape(out.grad, self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = sum_to_shape(out.grad, other.data.shape)
                other.grad = grad if other.grad is None else other.grad + grad


    def __mul__(self, other):
        out = Tensor(
            result = self.data * other.data,
            requires_grad= self.requires_grad or other.requires_grad,
            _children = (self, other)
        )
        
        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward

    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()



    def __matmul__(self, other):
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out
        

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

def sum_to_shape(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad
        