from torch import nn


class MyLargeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100, bias=False)
        self.l2 = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        for _ in range(10):
            x = self.l1(x)
        x = self.l2(x)
        return x


class MySmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        x = self.l1(x)
        return x
