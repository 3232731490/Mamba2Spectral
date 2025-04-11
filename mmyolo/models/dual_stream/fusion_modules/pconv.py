
import torch 
import torch.nn as nn 

from torch import Tensor 

from mmyolo.registry import MODELS

@MODELS.register_module()
class PConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size = 3,
                 n_div: int = 4,
                 forward: str = 'split_cat'):
        super(PConv2d, self).__init__()
        assert in_channels > 4, "in_channels should > 4, but got {} instead.".format(in_channels)
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv
        self.in_channels = in_channels
        
        self.conv = nn.Conv2d(in_channels=self.dim_conv,
                              out_channels=self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        
        self.out1 = nn.Conv2d(in_channels,in_channels//2,kernel_size=1)
        self.out2 = nn.Conv2d(in_channels,in_channels//2,kernel_size=1)
        
        if forward == 'slicing':
            self.forward = self.forward_slicing
            
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat 
            
        else:
            raise NotImplementedError("forward method: {} is not implemented.".format(forward))
        
        
    def forward_slicing(self, x1: torch.Tensor,x2:torch.Tensor) -> Tensor:
        out = self.conv(torch.cat([x1[:,:self.dim_conv//2,:,:],x2[:,:self.dim_conv//2,:,:]],dim=1))
        x1[:,:self.dim_conv//2,:,:],x2[:,:self.dim_conv//2,:,:] = out[:,:self.dim_conv//2,:,:],out[:,self.dim_conv//2:,:,:]

        return torch.cat([self.out1(x1),self.out2(x2)],dim=1)
    
    def forward_split_cat(self, x1: torch.Tensor,x2:torch.Tensor) -> Tensor:
        x11, x12 = torch.split(x1, [self.dim_conv//2, self.in_channels-self.dim_conv//2], dim=1)
        x21, x22 = torch.split(x2, [self.dim_conv//2, self.in_channels-self.dim_conv//2], dim=1)
        x11,x21 = torch.split(self.conv(torch.cat([x11,x21],dim=1)),[self.dim_conv//2,self.dim_conv//2],dim=1)
        x = torch.cat([self.out1(torch.cat((x11, x12), dim=1)),self.out2(torch.cat((x21, x22), dim=1))],dim=1)
        return x