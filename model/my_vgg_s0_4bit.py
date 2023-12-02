import torch
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from torch.nn.utils.prune import ln_structured
from quan.quant_layers import QuanConv2d, QuanLinear
from quan.quant_sparse_layers import QuanSparseConv2dSRP_feng,QuanSparseLinearSRP_feng
from quan.SiMAN import SiMAN_BinarizeConv2d
from quan.BiNet import BiNet_BinarizeConv2d
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        vb=[4,4,8]
        va=[4,4,8]
        vs=[0.00,0.00,0.00]
        vg=[1,1,1]
        tin=[16,16,16]
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # QuanLinear(512 * 7 * 7, 4096),
            QuanSparseLinearSRP_feng(in_features=512 * 7 * 7, out_features=4096, 
                             sparsity=vs[0], 
                             TIN=tin[0], group_size=vg[0], 
                             weight_bit_width=vb[0],activation_bit_width=va[0]),
            # QuanSparseLinearSRP(512 * 7 * 7, 4096,0.5),
            nn.ReLU(True),
            nn.Dropout(),
            # QuanLinear(4096, 4096),
            # QuanSparseLinearSRP(4096, 4096,0.5),
            QuanSparseLinearSRP_feng(in_features=4096, out_features=4096, 
                             sparsity=vs[1],
                             TIN=tin[1], group_size=vg[1], 
                             weight_bit_width=vb[1],activation_bit_width=va[1]),
            # ln_structured(nn.Linear(4096, 4096), name='weight',amount=0.1,dim=0,n=float('-inf')),
            nn.ReLU(True),
            nn.Dropout(),
            # QuanLinear(4096, num_classes),
            # QuanSparseLinearSRP(4096, num_classes,0.5),
            QuanSparseLinearSRP_feng(in_features=4096, out_features=num_classes, sparsity=vs[2],
                             TIN=tin[2], group_size=vg[2], 
                             weight_bit_width=vb[2],activation_bit_width=va[2]),
            
            # ln_structured(nn.Linear(4096, num_classes), name='weight',amount=0.1,dim=0,n=float('-inf')),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, QuanConv2d) or isinstance(m,QuanLinear) or isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                m._init_weight()
    
    def rest_mask(self):
        for m in self.modules():
            if isinstance(m, QuanConv2d) or isinstance(m,QuanLinear) or isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                m._reset_mask()
    def unset_mask(self):
        for m in self.modules():
            if isinstance(m, QuanConv2d) or isinstance(m,QuanLinear) or isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                m._unset_mask()
    def set_mask(self):
        for m in self.modules():
            if isinstance(m, QuanConv2d) or isinstance(m,QuanLinear) or isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                m._set_mask()
    def calculate_complexity(self):
        total = 0
        for m in self.modules():
            if isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                # print(m._calculate_complexity())
                total+=m._calculate_complexity()
        return total
    def calculate_coded_complexity(self):
        total = 0
        for m in self.modules():
            if isinstance(m,QuanSparseLinearSRP_feng) or isinstance(m,QuanSparseConv2dSRP_feng):
                # print(m._calculate_complexity())
                total+=m._calculate_coded_complexity_MB()
        return total

def make_layers_vgg16(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    l=0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            l=l+1
        else:
            vo  = cast(int, v[0])
            vww = cast(int, v[1])
            vaw = cast(int, v[2])
            vs  = cast(float, v[3])
            vTin= cast(int, v[4])
            vg  = cast(int, v[5])
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # conv2d = QuanConv2d(in_channels, v, kernel_size=3, padding=1)
            if(vww==1 and vaw==1):
                conv2d = SiMAN_BinarizeConv2d(in_channels, vo, kernel_size=3, stride=1, padding=1, bias=False)
                #conv2d = BiNet_BinarizeConv2d(in_channels, vo, kernel_size=3, stride=1, padding=1, bias=True)
                print("weight_bit_width: ",vww,","," activation_bit_width: ",vaw,", sparsity: ",vs,", Tin: ", vTin, ", Group: ", vg, ", Using SiMAN !!!")
            else:
                conv2d = QuanSparseConv2dSRP_feng(in_channels = in_channels, out_channels=vo, 
                                      weight_bit_width =vww,
                                      activation_bit_width=vaw,
                                      sparsity = vs, TIN=vTin, group_size=vg,
                                      kernel_size=3, padding=1)
            # conv2d = ln_structured(nn.Conv2d(in_channels, v, kernel_size=3, padding=1), name='weight',amount = 0.1, dim = 0,n=float('-inf'))
            
            next_cfg=cfg[l+1]
            if(next_cfg!='M'):
                vaw_next=cast(int, next_cfg[2])
            else:
                vaw_next="M"

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(vo), nn.ReLU(inplace=True)]
            elif (vaw_next==1): #or vaw_next==2):
                #layers += [conv2d, nn.PReLU()] #Hardtanh for BiNet #PReLU for SiMAN
                layers += [conv2d, nn.BatchNorm2d(vo), nn.PReLU()]
                print("using PReLU !!! \n")
            else:
                layers += [conv2d, nn.BatchNorm2d(vo), nn.ReLU(inplace=True)]
                print("using ReLU !!! \n")
            in_channels = vo
            l=l+1
    return nn.Sequential(*layers)

kbs_cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [[ 64,8,8,0.00,16,1], [ 64,6,6,0.00,16,1], 'M', #【Cin, Weight_bit_width, Activation_bit_width, sparsity, Tin, Group�?     
          [128,4,4,0.00,16,1], [128,4,4,0.00,16,1], 'M', 
          [256,4,4,0.00,16,1], [256,4,4,0.00,16,1], [256,4,4,0.00,16,1], 'M', 
          [512,4,4,0.00,16,1], [512,4,4,0.00,16,1], [512,4,4,0.00,16,1], 'M', 
          [512,4,4,0.00,16,1], [512,4,4,0.00,16,1], [512,4,4,0.00,16,1], 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# kbs_cfgs: Dict[str, List[Union[str, int]]] = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [[ 64,8,8,0.00,16,1], [ 64,4,4,0.00,16,1], 'M', #【Cin, Weight_bit_width, Activation_bit_width, sparsity, Tin, Group�?#           [128,4,4,0.00,16,1], [128,2,2,0.00,16,1], 'M', 
#           [256,2,2,0.00,16,1], [256,1,1,0.00,16,1], [256,2,2,0.00,16,1], 'M', 
#           [512,2,2,0.00,16,1], [512,1,1,0.00,16,1], [512,2,2,0.00,16,1], 'M', 
#           [512,2,2,0.00,16,1], [512,2,2,0.00,16,1], [512,4,4,0.00,16,1], 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }



def _vgg16(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers_vgg16(kbs_cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                            progress=progress)
        print(state_dict)
        model.load_state_dict(state_dict,strict=False)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg16('vgg16_bn', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg16('vgg16_bn', 'D', True, pretrained, progress, **kwargs)
