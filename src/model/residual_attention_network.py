import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import (
    AttentionModule_stage1,
    AttentionModule_stage2,
    AttentionModule_stage3,
)


class ResidualAttentionModel_56(nn.Module):
    # for input size 224 x 224
    def __init__(self, retrieve_layers=False, retrieve_masks=False):
        super(ResidualAttentionModel_56, self).__init__()
        self.retrieve_layers = retrieve_layers
        self.retrieve_masks = retrieve_masks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(
            256, 256, retrieve_mask=self.retrieve_masks
        )
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(
            512, 512, retrieve_mask=self.retrieve_masks
        )
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(
            1024, 1024, retrieve_mask=self.retrieve_masks
        )
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        if self.retrieve_layers:
            conv1_out = out

        out = self.mpool1(out)
        if self.retrieve_layers:
            mpool1_out = out

        out = self.residual_block1(out)
        if self.retrieve_layers:
            res1_out = out
        if self.retrieve_masks:
            out, atten1_mask = self.attention_module1(out)
        else:
            out = self.attention_module1(out)
        if self.retrieve_layers:
            atten1_out = out

        out = self.residual_block2(out)
        if self.retrieve_layers:
            res2_out = out

        if self.retrieve_masks:
            out, atten2_mask = self.attention_module2(out)
        else:
            out = self.attention_module2(out)
        if self.retrieve_layers:
            atten2_out = out

        out = self.residual_block3(out)
        if self.retrieve_layers:
            res3_out = out

        if self.retrieve_masks:
            out, atten3_mask = self.attention_module3(out)
        else:
            out = self.attention_module3(out)
        if self.retrieve_layers:
            atten3_out = out

        out = self.residual_block4(out)
        if self.retrieve_layers:
            res4_out = out

        out = self.residual_block5(out)
        if self.retrieve_layers:
            res5_out = out

        out = self.residual_block6(out)
        if self.retrieve_layers:
            res6_out = out

        out = self.mpool2(out)
        if self.retrieve_layers:
            mpool2_out = out

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.retrieve_layers and not self.retrieve_masks:
            layers = (
                conv1_out,
                mpool1_out,
                res1_out,
                atten1_out,
                res2_out,
                atten2_out,
                res3_out,
                atten3_out,
                res4_out,
                res5_out,
                res6_out,
                mpool2_out,
            )
            return out, layers
        elif self.retrieve_layers and self.retrieve_masks:
            layers = (
                conv1_out,
                mpool1_out,
                res1_out,
                atten1_out,
                res2_out,
                atten2_out,
                res3_out,
                atten3_out,
                res4_out,
                res5_out,
                res6_out,
                mpool2_out,
            )
            masks = (atten1_mask, atten2_mask, atten3_mask)
            return out, layers, masks
        elif not self.retrieve_layers and self.retrieve_masks:
            masks = (atten1_mask, atten2_mask, atten3_mask)
            return out, masks
        else:
            return out
