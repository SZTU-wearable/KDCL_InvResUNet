from .Teacher import Teacher_Model
from .U_Net_1D import UNet
# from MobileNetV3_small import modile_net_v3_small
from .InvertedResidual_small_Unet import invertedResidual_small_unet
from .InvertedResidual_large_Unet import invertedResidual_large_unet

model_dict = {
    'UTransBPNet': Teacher_Model,
    # 'UNet': UNet, # noUNet
    'InvertedResidual_UNet_small': invertedResidual_small_unet,
    'InvertedResidual_UNet_large': invertedResidual_large_unet
}
