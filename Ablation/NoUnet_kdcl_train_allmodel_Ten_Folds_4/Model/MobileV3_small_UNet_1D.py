from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from MobileNetV3 import mobilenet_v3_large
from MobileNetV3 import mobilenet_v3_small
from pytorch_model_summary import summary
from thop import profile
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


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class MobileV3Unet(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileV3Unet, self).__init__()
        backbone = mobilenet_v3_small()

        # if pretrain_backbone:
        #     # 载入mobilenetv3 large backbone预训练权重
        #     # https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

        backbone = backbone.features
        stage_indices = [1, 3, 6, 8, 11]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]

        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        c = self.stage_out_channels[4]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-1:]
        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
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
    a = torch.randn(32, 4, 625)
    model = MobileV3Unet()
    print("input shape: ", a.shape)
    print("output shape: ", model(a).shape)
    flops, params = profile(model, inputs=(a, ))

    print(f"模型的计算量估计：{flops / 1e9} GFLOPs")
    print(f"模型的参数数量：{params / 1e6} Million")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            total_params += param.numel()

    print("Total number of parameters: ", total_params)
    print_model_parameters(model)
    # summary(model, input_size=(4, 625), device="cpu")
    # summary(model,
    #         a,
    #         show_input=False,
    #         show_hierarchical=False,
    #         print_summary=True)
