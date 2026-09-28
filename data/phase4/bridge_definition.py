import torch
from torch import nn

class ResidualBottleneckBridge(nn.Module):
    """Trainable tokenwise residual approximation; pipeline integration is a separate wrapper."""
    def __init__(self, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        # Identity initialization is a starting point for learning, not the final replacement.
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        return hidden_states + self.up(self.activation(self.down(hidden_states)))
