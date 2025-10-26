from torch import nn


class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.lin(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
