import torch.nn as nn
from .basic_layers import ResidualBlock


class AttentionModule_pre(nn.Module):
    def __init__(
        self, in_channels, out_channels, size1, size2, size3, retrieve_mask=False
    ):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.residual4_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.residual5_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        out_residual2 = self.residual2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_residual2)
        out_mpool3 = self.mpool3(out_residual2)
        out_residual3 = self.residual3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_residual3)
        out = out_interp3 + out_skip2_connection
        out_residual4 = self.residual4_blocks(out)
        out_interp2 = self.interpolation2(out_residual4)
        out = out_interp2 + out_skip1_connection
        out_residual5 = self.residual5_blocks(out)
        out_interp1 = self.interpolation1(out_residual5)
        out_residual6 = self.residual6_blocks(out_interp1)
        out = (1 + out_residual6) * out_trunk
        out_last = self.last_blocks(out)
        if self.retrieve_mask:
            return out_last, out_residual6
        return out_last


class AttentionModule_stage0(nn.Module):
    # input size is 112x112
    def __init__(
        self,
        in_channels,
        out_channels,
        size1=(112, 112),
        size2=(56, 56),
        size3=(28, 28),
        size4=(14, 14),
        retrieve_mask=False,
    ):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # size 56x56
        self.residual1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # size 28x28
        self.residual2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # size 14x14
        self.residual3_blocks = ResidualBlock(in_channels, out_channels)
        self.skip3_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # size 7x7
        self.residual4_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )
        self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
        self.residual5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.residual6_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.residual7_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual8_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        # size 112x112
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        # size 56x56
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        # size 28x28
        out_residual2 = self.residual2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_residual2)
        out_mpool3 = self.mpool3(out_residual2)
        # size 14x14
        out_residual3 = self.residual3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_residual3)
        out_mpool4 = self.mpool4(out_residual3)
        # size 7x7
        out_residual4 = self.residual4_blocks(out_mpool4)
        out_interp4 = self.interpolation4(out_residual4) + out_residual3
        out = out_interp4 + out_skip3_connection
        out_residual5 = self.residual5_blocks(out)
        out_interp3 = self.interpolation3(out_residual5) + out_residual2
        out = out_interp3 + out_skip2_connection
        out_residual6 = self.residual6_blocks(out)
        out_interp2 = self.interpolation2(out_residual6) + out_residual1
        out = out_interp2 + out_skip1_connection
        out_residual7 = self.residual7_blocks(out)
        out_interp1 = self.interpolation1(out_residual7) + out_trunk
        out_residual8 = self.residual8_blocks(out_interp1)
        out = (1 + out_residual8) * out_trunk
        out_last = self.last_blocks(out)
        if self.retrieve_mask:
            return out_last, out_residual8
        return out_last


class AttentionModule_stage1(nn.Module):
    # input size is 56x56
    def __init__(
        self,
        in_channels,
        out_channels,
        size1=(56, 56),
        size2=(28, 28),
        size3=(14, 14),
        retrieve_mask=False,
    ):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.residual4_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.residual5_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        out_residual2 = self.residual2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_residual2)
        out_mpool3 = self.mpool3(out_residual2)
        out_residual3 = self.residual3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_residual3) + out_residual2
        out = out_interp3 + out_skip2_connection
        out_residual4 = self.residual4_blocks(out)
        out_interp2 = self.interpolation2(out_residual4) + out_residual1
        out = out_interp2 + out_skip1_connection
        out_residual5 = self.residual5_blocks(out)
        out_interp1 = self.interpolation1(out_residual5) + out_trunk
        out_residual6 = self.residual6_blocks(out_interp1)
        out = (1 + out_residual6) * out_trunk
        out_last = self.last_blocks(out)
        if self.retrieve_mask:
            return out_last, out_residual6
        return out_last


class AttentionModule_stage2(nn.Module):
    # input image size is 28x28
    def __init__(
        self,
        in_channels,
        out_channels,
        size1=(28, 28),
        size2=(14, 14),
        retrieve_mask=False,
    ):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.residual3_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual4_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        out_residual2 = self.residual2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_residual2) + out_residual1
        out = out_interp2 + out_skip1_connection
        out_residual3 = self.residual3_blocks(out)
        out_interp1 = self.interpolation1(out_residual3) + out_trunk
        out_residual4 = self.residual4_blocks(out_interp1)
        out = (1 + out_residual4) * out_trunk
        out_last = self.last_blocks(out)
        if self.retrieve_mask:
            return out_last, out_residual4
        return out_last


class AttentionModule_stage3(nn.Module):
    # input image size is 14x14
    def __init__(self, in_channels, out_channels, size1=(14, 14), retrieve_mask=False):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_residual1) + out_trunk
        out_residual2 = self.residual2_blocks(out_interp1)
        out = (1 + out_residual2) * out_trunk
        out_last = self.last_blocks(out)
        if self.retrieve_mask:
            return out_last, out_residual2
        return out_last
