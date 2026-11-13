class SGD:
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def step(self):
        # move each weight a small step against its gradient
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        # gradients add up in backward, so clear them before the next step
        for p in self.params:
            p.grad = None
