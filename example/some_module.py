from torch import nn


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 10, bias=False)
        self.l2 = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class A:
    pass
