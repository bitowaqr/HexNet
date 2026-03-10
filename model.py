import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ResBlock(nn.Module):
    """Simple residual block: conv -> bn -> relu -> conv -> bn -> skip connection -> relu"""

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class HexNet(nn.Module):
    def __init__(self, size=5, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.size = size
        self.action_size = size * size

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * size * size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(size * size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args: x of shape (batch, 3, size, size)
        Returns: (policy, value)
          - policy: (batch, size*size) -- log probabilities
          - value: (batch, 1) -- in range [-1, 1]
        """
        # Initial conv block
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
