import numpy as np


def to_tensor(x):
    # let us write things like  t + 1  or  t * 2  (wrap plain numbers in a Tensor)
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def sum_to_shape(grad, shape):
    # numpy broadcasting can make the output bigger than the input
    # (for example bias (1, 3) + x (64, 3) -> (64, 3)).
    # in backward we must sum the gradient back to the original shape.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    def _add_grad(self, grad):
        # gradients from different paths are added together (chain rule)
        if self.requires_grad:
            grad = sum_to_shape(grad, self.data.shape)
            self.grad = grad if self.grad is None else self.grad + grad

    # ---------------- basic math ----------------

    def __add__(self, other):
        other = to_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )

        def _backward():
            # d(a+b)/da = 1 , d(a+b)/db = 1
            self._add_grad(out.grad)
            other._add_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = to_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )

        def _backward():
            # d(a*b)/da = b , d(a*b)/db = a
            self._add_grad(other.data * out.grad)
            other._add_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __pow__(self, power):
        # only numbers as power (like x ** 2)
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,)
        )

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self._add_grad(power * self.data ** (power - 1) * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )

        def _backward():
            # out = A @ B  ->  dA = dout @ B.T , dB = A.T @ dout
            self._add_grad(out.grad @ other.data.T)
            other._add_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    # these are built from the ops above, so they need no backward of their own
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-to_tensor(other))

    def __radd__(self, other):  # 1 + t
        return self + other

    def __rmul__(self, other):  # 2 * t
        return self * other

    def __rsub__(self, other):  # 1 - t
        return to_tensor(other) + (-self)

    def __truediv__(self, other):
        return self * to_tensor(other) ** -1

    # ---------------- activations ----------------

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,)
        )

        def _backward():
            # gradient passes only where input was positive
            self._add_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            # d(sigmoid)/dx = s * (1 - s)
            self._add_grad(s * (1 - s) * out.grad)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,)
        )

        def _backward():
            # d(log x)/dx = 1 / x
            self._add_grad(out.grad / self.data)

        out._backward = _backward
        return out

    # ---------------- reductions ----------------

    def sum(self):
        out = Tensor(
            self.data.sum(),
            requires_grad=self.requires_grad,
            _children=(self,)
        )

        def _backward():
            # every element gets the same gradient
            self._add_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self):
        return self.sum() / self.data.size

    # ---------------- backprop ----------------

    def backward(self):
        # 1) sort the graph so every node comes after its children
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2) start from d(loss)/d(loss) = 1 and go backwards
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.grad is not None:  # constants (like the labels) have no grad
                node._backward()
