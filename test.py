from time import time

import torch
import torchcde

st = time()
device = 'cuda:1'
# Create some data
batch, length, input_channels = 64, 200, 5
hidden_channels = 128
t = torch.linspace(0, 1, length).to(device)
t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
x_ = torch.rand(batch, length, input_channels - 1).to(device)
x = torch.cat([t_, x_], dim=2)  # include time as a channel

# Interpolate it
coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
X = torchcde.CubicSpline(coeffs)

# Create the Neural CDE system
class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.linear = torch.nn.Linear(hidden_channels,
                                      hidden_channels * input_channels)

    def forward(self, t, z):
        return self.linear(z).view(batch, hidden_channels, input_channels)

func = F().to(device)
z0 = torch.rand(batch, hidden_channels).to(device)

# Integrate it
torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval)

et = time()
print(et - st)