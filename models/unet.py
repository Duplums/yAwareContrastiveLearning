"""
The U-Net is a convolutional encoder-decoder neural network.
"""

# Imports
import torch
import torch.nn as nn

class UNet(nn.Module):
    """ UNet.

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:

    - padding is used in 3x3x3 convolutions to prevent loss
      of border pixels
    - merging outputs does not require cropping due to (1)
    - residual connections can be used by specifying
      UNet(merge_mode='add')
    """

    def __init__(self, num_classes, in_channels=1, depth=5,
                  mode="seg"):
        """ Init class.

        Parameters
        ----------
        num_classes: int
            the number of features in the output segmentation map (in 'seg' mode) or
            number of classes (in 'classif' mode)
        in_channels: int, default 1
            number of channels in the input tensor.
        depth: int, default 5
            number of layers in the U-Net.
        mode: 'str', default 'seg'
            Whether the network is turned in 'segmentation' mode ("seg"), 'classification' ("classif"),
            'encoder' (only encoder pathway) or 'simCLR' (encoder pathway and non-linear head projection, cf.
            https://github.com/google-research/simclr)
        """
        # Inheritance
        super(UNet, self).__init__()

        if mode in ("seg", "encoder", "simCLR", "classif"):
            self.mode = mode
        else:
            raise ValueError("'{}' is not a valid mode.".format(mode))

        # Declare class parameters
        self.start_filts = 16 # Initial nb of kernels
        self.down_mode = "maxpool"
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.down = []
        self.up = [] # Useful in seg mode
        self.classifier = None # Useful in classif mode
        self.name = "UNet_D%i_%s" % (self.depth, self.mode)

        # Create the encoder pathway
        out_channels = 1
        for cnt in range(depth):
            in_channels = self.in_channels if cnt == 0 else out_channels
            out_channels = self.start_filts * (2**cnt)
            down_sampling = False if cnt == 0 else True
            self.down.append(
                Down(in_channels, out_channels, down_mode=self.down_mode,
                     pooling=down_sampling, batchnorm=True))

        if self.mode == "seg":
            # Create the decoder pathway
            # - careful! decoding only requires depth-1 blocks
            for cnt in range(depth - 1):
                in_channels = out_channels
                out_channels = in_channels // 2
                self.up.append(
                    Up(in_channels, out_channels, batchnorm=True))

        if self.mode == "classif":
            self.classifier = Classifier(self.num_classes, features=self.start_filts * 2**(self.depth-1))

        elif self.mode == 'simCLR':
            self.hidden_representation = nn.Linear(self.start_filts * (2**(self.depth-1)), 512)
            self.head_projection = nn.Linear(512, 128)

        # Add the list of modules to current module
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

        # Get ouptut segmentation
        if self.mode == "seg":
            self.conv_final = Conv1x1x1(out_channels, self.num_classes)

        # Kernel initializer
        # Weight initialization
        self.weight_initializer()

    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_outs = []
        for module in self.down:
            x = module(x)
            encoder_outs.append(x)
        x_enc = x
        if self.mode == "encoder":
            # Avg over each output feature map, output size should be == nb_filters
            return nn.functional.adaptive_avg_pool3d(x_enc, 1)
        elif self.mode == 'simCLR':
            x_enc = nn.functional.relu(x_enc)
            x_enc = nn.functional.adaptive_avg_pool3d(x_enc, 1)
            x_enc = torch.flatten(x_enc, 1)
            x_enc = self.hidden_representation(x_enc)
            x_enc = nn.functional.relu(x_enc)
            x_enc = self.head_projection(x_enc)
            return x_enc
        if self.mode == "seg":
            encoder_outs = encoder_outs[:-1][::-1]
            for cnt, module in enumerate(self.up):
                x_up = encoder_outs[cnt]
                x = module(x, x_up)
            # No softmax is used. This means you need to use
            # nn.CrossEntropyLoss in your training script,
            # as this module includes a softmax already.
            x_seg = self.conv_final(x)
            return x_seg
        if self.mode == "classif":
            # No softmax used
            x_classif = self.classifier(x_enc)
            return x_classif

        raise ValueError("Unknown mode: %s"%self.mode)


class Classifier(nn.Module):
    def __init__(self, nb_classes, features):
        super(Classifier, self).__init__()
        self.num_classes = nb_classes
        self.features = features
        self.fc1 = nn.Linear(self.features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, self.num_classes, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x.squeeze(1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, batchnorm=True):
        super(DoubleConv, self).__init__()
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        if batchnorm:
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.batchnorm:
            x = self.norm2(x)
        x = self.relu(x)

        return x


def UpConv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def Conv1x1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class Down(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True,
                 down_mode='maxpool', batchnorm=True):
        super(Down, self).__init__()

        self.pooling = pooling
        self.down_mode = down_mode
        if self.down_mode == "maxpool":
            self.maxpool = nn.MaxPool3d(2)
            self.doubleconv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x):
        if self.pooling:
            x = self.maxpool(x)
        x = self.doubleconv(x)
        return x


class Up(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, up_mode="transpose",
                 batchnorm=True):
        super(Up, self).__init__()
        self.up_mode = up_mode
        self.upconv = UpConv(in_channels, out_channels)
        self.doubleconv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x_down, x_up):
        x_down = self.upconv(x_down)
        x = torch.cat((x_up, x_down), dim=1)
        x = self.doubleconv(x)
        return x
