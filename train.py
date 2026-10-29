from tensor import Tensor
from nn import Linear

layer = Linear(2, 3)

x = Tensor([[1, 2]])

z = layer.forward(x)

print(z.data)
print(z.data.shape)