"""
ODE-based models.
"""

import torchcde
from torchdiffeq import odeint

from model.base import *
from model.induced_att import IAEncoderLayer, PositionalEncoding


class CDEEncoder(Encoder):
    def __init__(self, input_cols, hidden_size, output_size, sampler):
        super().__init__(sampler, 'CDE')

        self.input_cols = input_cols
        self.input_size = len(input_cols)
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Linear(hidden_size, self.input_size * hidden_size)
        self.z_start = nn.Parameter(torch.rand(hidden_size).float(), requires_grad=True)
        self.out_linear = nn.Linear(hidden_size * 2, output_size)

    def cde_neural(self, t, z):
        return rearrange(self.net(z), 'B (H I) -> B H I', H=self.hidden_size, I=self.input_size)

    def forward(self, trip, valid_len):
        B, L, E_in = trip.shape

        x, = self.sampler(trip)
        x = x[:, :, self.input_cols]

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.z_start.unsqueeze(0).repeat(B, 1)
        z = torchcde.cdeint(X=X, func=self.cde_neural, z0=z0, t=X.interval)  # (B, 2, hidden_size)

        z = rearrange(z, 'B Z H -> B (Z H)')
        z = self.out_linear(z)
        return z