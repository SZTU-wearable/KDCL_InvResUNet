import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from typing import Callable, List, Optional
from pytorch_model_summary import summary


# from torchsummary import summary


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool1d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv1d(in_channels, num_classes, kernel_size=1)
        )


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv1d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv1d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool1d(x, output_size=1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv1d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, exp_size, stride, use_se, use_hs,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (stride == 1 and in_planes == out_planes)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # expand
        if exp_size != in_planes:
            layers.append(ConvBNActivation(in_planes,
                                           exp_size,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(exp_size,
                                       exp_size,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       groups=exp_size,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if use_se:
            layers.append(SqueezeExcitation(exp_size))

        # project
        layers.append(ConvBNActivation(exp_size,
                                       out_planes,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_planes
        self.is_strided = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class invertedResidual_large_unet(nn.Module):
    def __init__(self, out_planes=1):
        super(invertedResidual_large_unet, self).__init__()
        in_planes = 4
        norm_layer = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.in_0 = InvertedResidual(in_planes=4, out_planes=16, kernel_size=3, exp_size=16, stride=2, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_1 = InvertedResidual(in_planes=16, out_planes=16, kernel_size=3, exp_size=16, stride=1, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_2 = InvertedResidual(in_planes=16, out_planes=24, kernel_size=3, exp_size=64, stride=2, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_3 = InvertedResidual(in_planes=24, out_planes=24, kernel_size=3, exp_size=72, stride=1, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_4 = InvertedResidual(in_planes=24, out_planes=40, kernel_size=5, exp_size=72, stride=2, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_5 = InvertedResidual(in_planes=40, out_planes=40, kernel_size=5, exp_size=120, stride=1, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_6 = InvertedResidual(in_planes=40, out_planes=40, kernel_size=5, exp_size=120, stride=1, use_se=1, use_hs=0, norm_layer=norm_layer)
        self.in_7 = InvertedResidual(in_planes=40, out_planes=80, kernel_size=3, exp_size=240, stride=2, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_8 = InvertedResidual(in_planes=80, out_planes=80, kernel_size=3, exp_size=200, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_9 = InvertedResidual(in_planes=80, out_planes=80, kernel_size=3, exp_size=184, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_10 = InvertedResidual(in_planes=80, out_planes=80, kernel_size=3, exp_size=184, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_11 = InvertedResidual(in_planes=80, out_planes=112, kernel_size=3, exp_size=480, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_12 = InvertedResidual(in_planes=112, out_planes=112, kernel_size=3, exp_size=672, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_13 = InvertedResidual(in_planes=112, out_planes=160, kernel_size=5, exp_size=672, stride=2, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_14 = InvertedResidual(in_planes=160, out_planes=160, kernel_size=5, exp_size=960, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)
        self.in_15 = InvertedResidual(in_planes=160, out_planes=160, kernel_size=5, exp_size=960, stride=1, use_se=1, use_hs=1, norm_layer=norm_layer)

        self.up1 = Up(self.in_15.out_channels, self.in_12.out_channels)
        self.up2 = Up(self.in_12.out_channels, self.in_6.out_channels)
        self.up3 = Up(self.in_6.out_channels, self.in_3.out_channels)
        self.up4 = Up(self.in_3.out_channels, self.in_1.out_channels)
        self.conv = OutConv(self.in_1.out_channels, num_classes=out_planes)

    def forward(self, x):
        input_shape = x.shape[-1:]
        x0 = self.in_0(x)
        x1 = self.in_1(x0)
        x2 = self.in_2(x1)
        x3 = self.in_3(x2)
        x4 = self.in_4(x3)
        x5 = self.in_5(x4)
        x6 = self.in_6(x5)
        x7 = self.in_7(x6)
        x8 = self.in_8(x7)
        x9 = self.in_9(x8)
        x10 = self.in_10(x9)
        x11 = self.in_11(x10)
        x12 = self.in_12(x11)
        x13 = self.in_13(x12)
        x14 = self.in_14(x13)
        x15 = self.in_15(x14)
        x = self.up1(x15, x12)
        x = self.up2(x, x6)
        x = self.up3(x, x3)
        x = self.up4(x, x1)
        x = self.conv(x)
        x = F.interpolate(input=x, size=input_shape)
        return x


def print_model_parameters(model):
    # 打印每一层的结构和参数量
    for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters())
        # print(f"{name}: {num_params} parameters")
        print(f"{name}: {num_params / 1e6} Million parameters")

    # 打印总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # print("total_params = ", total_params)
    print(f"模型的参数数量：{total_params / 1e6} Million")
    return total_params


if __name__ == "__main__":
    from thop import profile
    from pytorch_model_summary import summary

    a = torch.randn(1, 4, 625)
    model = invertedResidual_large_unet()
    # print(model)
    # print(model.features)
    print("input shape: ", a.shape)
    print("output shape: ", model(a).shape)
    # summary(model, (4,625), device="cpu")
    summary(model,
            a,
            show_input=False,
            show_hierarchical=False,
            print_summary=True)
    flops, params = profile(model, inputs=(a,))
    print(f"模型的计算量估计：{flops / 1e9} GFLOPs")
    print(f"模型的参数数量：{params / 1e6} Million")
    # print_model_parameters(model)
