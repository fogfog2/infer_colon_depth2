# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
from functools import partial

from packnet_sfm.networks.layers.resnet.resnet_encoder_h import ResnetEncoder
from packnet_sfm.networks.depth.SwinTransformer_h import SwinTransformer
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class fcconv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super().__init__()
        self.conv =  nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.nonlin = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(self.bn1(out))
        ##norm?
        return out

class DepthResNetSwin(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.resnet_encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)

        embed_dim=96
        depths=[ 2, 2, 2, 2  ]
        num_heads=[ 3, 6, 12, 24]
        drop_path_rate=0.2
        ape=False

        self.embed_dim= embed_dim    
        self.depths=depths
        self.num_heads =num_heads
        self.drop_path_rate = drop_path_rate
        self.ape = ape       
        #self.in_channels=[ self.embed_dim , self.embed_dim*2, self.embed_dim * 4, self.embed_dim * 8]         
        self.in_channels=[ 64 , self.embed_dim*2, self.embed_dim * 4, self.embed_dim * 8]         
        init_dim = 96
        self.de_channels=[64, 64,  self.embed_dim*2, self.embed_dim * 4, self.embed_dim * 8]      

        self.swin = SwinTransformer(embed_dim= self.embed_dim,
                                    depths=self.depths,
                                    num_heads=self.num_heads,
                                    drop_path_rate=self.drop_path_rate,
                                    ape = self.ape,
                                    init_dim = init_dim,
                                    out_indices=(0, 1, 2, 3),
                                    in_channels = self.in_channels)

        self.upconv = fcconv(64,96)
        self.decoder = DepthDecoder(num_ch_enc=self.de_channels)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, rgb):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        resnet_out = self.resnet_encoder(rgb)
        #out = self.upconv(resnet_out[-1])
        #swin_out = self.swin(out)
        swin_out = self.swin(resnet_out[-1])

        x = resnet_out + swin_out

        x = self.decoder(x) 
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return {
                'inv_depths': [self.scale_inv_depth(d)[0] for d in disps],
            }
        else:
            return {
                'inv_depths': self.scale_inv_depth(disps[0])[0],
            }

########################################################################################################################
