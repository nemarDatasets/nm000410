"""
Enhanced Multi-Scale ResNet with Dilated Convolutions and Attention

This module implements an enhanced version of the ResNet architecture incorporating:
- Dilated convolutions for expanded receptive fields
- Squeeze-and-Excitation (SE) attention blocks
- LSTM layers for temporal modeling
- Multi-stage processing with increasing dilation rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation attention layer

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.

    Args:
        channel: Number of input channels
        reduction: Reduction ratio for squeeze operation (default: 16)
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DilatedConv1d(nn.Module):
    """
    1D Dilated Convolution Layer

    Dilated convolutions provide exponentially expanding receptive fields
    without increasing the number of parameters.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: If True, adds a learnable bias (default: False)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super(DilatedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=dilation * (kernel_size - 1) // 2,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DilatedBlock(nn.Module):
    """
    Dilated Residual Block with optional SE attention

    Args:
        inplanes: Number of input channels
        planes: Number of output channels
        stride: Stride for the first convolution (default: 1)
        dilation: Dilation rate for convolutions (default: 1)
        downsample: Downsample layer for residual connection (default: None)
        use_se: Whether to use Squeeze-and-Excitation attention (default: False)
    """
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_se=False):
        super(DilatedBlock, self).__init__()
        self.conv1 = DilatedConv1d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DilatedConv1d(planes, planes, kernel_size=3, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SELayer(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class EnhancedMSResNetDilated(nn.Module):
    """
    Enhanced Multi-Scale ResNet with Dilated Convolutions

    A sophisticated architecture combining:
    - Multi-stage dilated residual blocks with increasing dilation rates
    - Squeeze-and-Excitation attention mechanisms
    - Bidirectional LSTM for temporal modeling
    - Attention-weighted feature aggregation

    Args:
        input_channel: Number of input channels (default: 1)
        num_classes: Number of output classes (default: 1)
        num_blocks: Number of blocks per layer (default: 3)
        use_se: Whether to use SE attention blocks (default: True)
        initial_channels: Number of channels after first conv (default: 32)
    """
    def __init__(self, input_channel=1, num_classes=1, num_blocks=3, use_se=True, initial_channels=32):
        super(EnhancedMSResNetDilated, self).__init__()
        self.use_se = use_se

        self.conv1 = nn.Conv1d(input_channel, initial_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(initial_channels, initial_channels, blocks=num_blocks, stride=1, dilation=1)
        self.layer2 = self._make_layer(initial_channels, initial_channels*2, blocks=num_blocks, stride=2, dilation=2)
        self.layer3 = self._make_layer(initial_channels*2, initial_channels*4, blocks=num_blocks, stride=2, dilation=4)
        self.layer4 = self._make_layer(initial_channels*4, initial_channels*8, blocks=num_blocks, stride=2, dilation=8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.attention = nn.Sequential(
            nn.Linear(initial_channels*8, initial_channels*4),
            nn.Tanh(),
            nn.Linear(initial_channels*4, 1)
        )

        self.lstm = nn.LSTM(initial_channels*8, initial_channels*4, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(initial_channels*8, initial_channels*4)
        self.fc2 = nn.Linear(initial_channels*4, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, inplanes, planes, blocks, stride=1, dilation=1):
        """
        Create a layer consisting of multiple dilated blocks

        Args:
            inplanes: Number of input channels
            planes: Number of output channels
            blocks: Number of blocks in the layer
            stride: Stride for the first block (default: 1)
            dilation: Dilation rate for all blocks (default: 1)

        Returns:
            Sequential container of dilated blocks
        """
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(DilatedBlock(inplanes, planes, stride, dilation, downsample, use_se=self.use_se))
        for _ in range(1, blocks):
            layers.append(DilatedBlock(planes, planes, dilation=dilation, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the enhanced network

        Args:
            x: Input tensor of shape (batch_size, input_channel, length)

        Returns:
            Output predictions of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        attention_weights = F.softmax(self.attention(x), dim=1)
        x = x * attention_weights

        x, _ = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
