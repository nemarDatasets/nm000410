"""
Optimized Multi-Scale ResNet for SEEG Signal Classification

This module implements an optimized multi-scale residual neural network architecture
designed for analyzing stereoelectroencephalography (SEEG) signals to identify the
seizure onset zone (SOZ).

Architecture Overview:
    - Uses parallel pathways with different temporal receptive fields
    - Kernel sizes: 3, 11, 21, 65, 129 samples
    - Each pathway consists of residual blocks with increasing channel dimensions
    - Features from all pathways are concatenated for final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution. Default: 1

    Returns:
        nn.Conv1d: 1D convolution layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv11x11(in_planes, out_planes, stride=1):
    """
    11x11 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution. Default: 1

    Returns:
        nn.Conv1d: 1D convolution layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=11, stride=stride,
                     padding=5, bias=False)


def conv21x21(in_planes, out_planes, stride=1):
    """
    21x21 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution. Default: 1

    Returns:
        nn.Conv1d: 1D convolution layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=21, stride=stride,
                     padding=10, bias=False)


def conv65x65(in_planes, out_planes, stride=1):
    """
    65x65 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution. Default: 1

    Returns:
        nn.Conv1d: 1D convolution layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=65, stride=stride,
                     padding=32, bias=False)


def conv129x129(in_planes, out_planes, stride=1):
    """
    129x129 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for convolution. Default: 1

    Returns:
        nn.Conv1d: 1D convolution layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=129, stride=stride,
                     padding=64, bias=False)


class BasicBlock3x3(nn.Module):
    """
    Basic residual block with 3x3 convolutions.

    This block performs two 3x3 convolutions with batch normalization and ReLU
    activation, followed by a residual connection. Suitable for capturing
    fine-grained temporal features.

    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock)
        conv1 (nn.Conv1d): First convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        relu (nn.ReLU): ReLU activation
        conv2 (nn.Conv1d): Second convolution layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Sequential): Downsample layer for residual connection
        stride (int): Stride for first convolution
    """
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock3x3.

        Args:
            inplanes3 (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution. Default: 1
            downsample (nn.Module): Downsample layer for residual. Default: None
        """
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Output tensor after residual block
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock11x11(nn.Module):
    """
    Basic residual block with 11x11 convolutions.

    This block performs two 11x11 convolutions with batch normalization and ReLU
    activation, followed by a residual connection. Suitable for capturing
    medium-range temporal features.

    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock)
        conv1 (nn.Conv1d): First convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        relu (nn.ReLU): ReLU activation
        conv2 (nn.Conv1d): Second convolution layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Sequential): Downsample layer for residual connection
        stride (int): Stride for first convolution
    """
    expansion = 1

    def __init__(self, inplanes11, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock11x11.

        Args:
            inplanes11 (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution. Default: 1
            downsample (nn.Module): Downsample layer for residual. Default: None
        """
        super(BasicBlock11x11, self).__init__()
        self.conv1 = conv11x11(inplanes11, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv11x11(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Output tensor after residual block
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock21x21(nn.Module):
    """
    Basic residual block with 21x21 convolutions.

    This block performs two 21x21 convolutions with batch normalization and ReLU
    activation, followed by a residual connection. Suitable for capturing
    longer-range temporal features.

    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock)
        conv1 (nn.Conv1d): First convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        relu (nn.ReLU): ReLU activation
        conv2 (nn.Conv1d): Second convolution layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Sequential): Downsample layer for residual connection
        stride (int): Stride for first convolution
    """
    expansion = 1

    def __init__(self, inplanes21, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock21x21.

        Args:
            inplanes21 (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution. Default: 1
            downsample (nn.Module): Downsample layer for residual. Default: None
        """
        super(BasicBlock21x21, self).__init__()
        self.conv1 = conv21x21(inplanes21, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv21x21(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Output tensor after residual block
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock65x65(nn.Module):
    """
    Basic residual block with 65x65 convolutions.

    This block performs two 65x65 convolutions with batch normalization and ReLU
    activation, followed by a residual connection. Suitable for capturing
    very long-range temporal features (approximately 130ms at 512Hz).

    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock)
        conv1 (nn.Conv1d): First convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        relu (nn.ReLU): ReLU activation
        conv2 (nn.Conv1d): Second convolution layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Sequential): Downsample layer for residual connection
        stride (int): Stride for first convolution
    """
    expansion = 1

    def __init__(self, inplanes65, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock65x65.

        Args:
            inplanes65 (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution. Default: 1
            downsample (nn.Module): Downsample layer for residual. Default: None
        """
        super(BasicBlock65x65, self).__init__()
        self.conv1 = conv65x65(inplanes65, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv65x65(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Output tensor after residual block
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock129x129(nn.Module):
    """
    Basic residual block with 129x129 convolutions.

    This block performs two 129x129 convolutions with batch normalization and ReLU
    activation, followed by a residual connection. Suitable for capturing
    extremely long-range temporal features (approximately 250ms at 512Hz).

    Attributes:
        expansion (int): Channel expansion factor (always 1 for BasicBlock)
        conv1 (nn.Conv1d): First convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        relu (nn.ReLU): ReLU activation
        conv2 (nn.Conv1d): Second convolution layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Sequential): Downsample layer for residual connection
        stride (int): Stride for first convolution
    """
    expansion = 1

    def __init__(self, inplanes129, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock129x129.

        Args:
            inplanes129 (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for first convolution. Default: 1
            downsample (nn.Module): Downsample layer for residual. Default: None
        """
        super(BasicBlock129x129, self).__init__()
        self.conv1 = conv129x129(inplanes129, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv129x129(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Output tensor after residual block
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OptimizedMSResNet(nn.Module):
    """
    Optimized Multi-Scale Residual Network for SEEG signal classification.

    This architecture processes input signals through 5 parallel pathways, each using
    different kernel sizes (3, 11, 21, 65, 129) to capture features at different
    temporal scales. Features from all pathways are combined and passed through a
    classification head.

    Architecture:
        1. Initial feature extraction: Conv(15x15) -> BN -> ReLU -> MaxPool
        2. Five parallel pathways with different kernel sizes
        3. Each pathway: 3 residual stages with increasing channels
        4. Global adaptive pooling per pathway
        5. Feature concatenation
        6. Multi-layer classification head with dropout

    Attributes:
        initial_channels (int): Number of channels after initial convolution
        conv1 (nn.Sequential): Initial feature extraction block
        path3 (nn.Sequential): Pathway with 3x3 kernels
        path11 (nn.Sequential): Pathway with 11x11 kernels
        path21 (nn.Sequential): Pathway with 21x21 kernels
        path65 (nn.Sequential): Pathway with 65x65 kernels
        path129 (nn.Sequential): Pathway with 129x129 kernels
        adaptive_pool (nn.AdaptiveMaxPool1d): Global pooling layer
        classifier (nn.Sequential): Classification head
    """

    def __init__(self, input_channel=1, num_classes=1, initial_channels=32,
                 layers=[1, 1, 1, 1], use_se=False, kernel_size=None):
        """
        Initialize OptimizedMSResNet.

        Args:
            input_channel (int): Number of input channels. Default: 1
            num_classes (int): Number of output classes. Default: 1
            initial_channels (int): Number of channels after first conv. Default: 32
            layers (list): Number of blocks per stage. Default: [1, 1, 1, 1]
            use_se (bool): Whether to use squeeze-excitation (unused). Default: False
            kernel_size: Unused parameter for compatibility
        """
        super(OptimizedMSResNet, self).__init__()

        self.initial_channels = initial_channels

        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channel, initial_channels, kernel_size=15,
                     stride=3, padding=7, bias=False),
            nn.BatchNorm1d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Channel progression across stages
        c1 = initial_channels
        c2 = initial_channels * 2
        c3 = initial_channels * 4

        # Create 5 parallel pathways with different temporal receptive fields
        self.path3 = self._create_path(BasicBlock3x3, c1, c2, c3, layers, kernel_size=3)
        self.path11 = self._create_path(BasicBlock11x11, c1, c2, c3, layers, kernel_size=11)
        self.path21 = self._create_path(BasicBlock21x21, c1, c2, c3, layers, kernel_size=21)
        self.path65 = self._create_path(BasicBlock65x65, c1, c2, c3, layers, kernel_size=65)
        self.path129 = self._create_path(BasicBlock129x129, c1, c2, c3, layers, kernel_size=129)

        # Global pooling
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head with layer normalization and GELU activation
        combined_features = c3 * 5  # 5 pathways

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, combined_features // 2),
            nn.LayerNorm(combined_features // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(combined_features // 2, combined_features // 4),
            nn.LayerNorm(combined_features // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(combined_features // 4, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _create_path(self, block, c1, c2, c3, layers, kernel_size):
        """
        Create a processing pathway with the specified block type.

        Each pathway consists of 3 stages with increasing channel dimensions.
        The stages progressively downsample the temporal dimension while
        increasing the feature depth.

        Args:
            block (nn.Module): BasicBlock class to use (e.g., BasicBlock3x3)
            c1 (int): Channels for first stage
            c2 (int): Channels for second stage
            c3 (int): Channels for third stage
            layers (list): Number of blocks per stage
            kernel_size (int): Kernel size for this pathway

        Returns:
            nn.Sequential: Complete pathway module
        """
        return nn.Sequential(
            self._make_layer(block, self.initial_channels, c1, layers[0], stride=2),
            self._make_layer(block, c1, c2, layers[1], stride=2),
            self._make_layer(block, c2, c3, layers[2], stride=2),
        )

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization.

        Special handling for different layer types:
        - Conv1d with large kernels (>256): Center-focused initialization
        - Conv1d with normal kernels: Kaiming normal initialization
        - BatchNorm1d: Ones for weight, zeros for bias
        - Linear: Small normal initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size[0] > 256:
                    # Special initialization for very long kernels
                    center = m.kernel_size[0] // 2
                    with torch.no_grad():
                        # Initialize center weights with larger variance
                        m.weight[:, :, center-5:center+6].normal_(0, 0.02)
                        # Initialize outer weights with smaller variance
                        m.weight[:, :, :center-5].normal_(0, 0.01)
                        m.weight[:, :, center+6:].normal_(0, 0.01)
                else:
                    # Regular Kaiming initialization for smaller kernels
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Class predictions of shape (batch, num_classes)
        """
        # Initial feature extraction
        x = self.conv1(x)

        # Process through parallel pathways
        x3 = self.path3(x)
        x11 = self.path11(x)
        x21 = self.path21(x)
        x65 = self.path65(x)
        x129 = self.path129(x)

        # Global pooling for each pathway
        x3 = self.adaptive_pool(x3)
        x11 = self.adaptive_pool(x11)
        x21 = self.adaptive_pool(x21)
        x65 = self.adaptive_pool(x65)
        x129 = self.adaptive_pool(x129)

        # Concatenate features from all pathways
        out = torch.cat([x3, x11, x21, x65, x129], dim=1)
        out = out.view(out.size(0), -1)

        # Classification
        return self.classifier(out)

    def _make_layer(self, block, inplanes, planes, blocks, stride=2):
        """
        Create a stage consisting of multiple residual blocks.

        Args:
            block (nn.Module): BasicBlock class to use
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            blocks (int): Number of blocks in this stage
            stride (int): Stride for the first block. Default: 2

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # Create downsampling layer for residual connection
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        # First block might have a stride and downsample
        layers.append(block(inplanes, planes, stride, downsample))

        # Remaining blocks have stride=1 and no downsample
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers)
