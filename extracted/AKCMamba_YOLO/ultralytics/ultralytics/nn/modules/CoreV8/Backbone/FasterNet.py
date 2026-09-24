import torch
import torch.nn as nn
import numpy as np

class PConv(nn.Module):
    # PConv Block
    def __init__(self,dim, n_div,forward= "split_cat",kernel_size=3):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv=nn.Conv2d(
            self.dim_conv, 
            self.dim_conv, 
            kernel_size, 
            stride=1,
            padding=(kernel_size -1) // 2,
            bias=False)
        
        if forward == "slicing":
            self.forward = self.forward_slicing 
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x
    
    def forward_split_cat(self, x):
        x1, x2 = torch.split(x,[self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x=torch.cat((x1,x2),1)
        return x

class FasterBlock(nn.Module):
    # FasterBlock
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = AKConv(c1, 2, "split_cat", 3)
        self.cv1 = Conv(c1, 2, "split_cat", 3)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

# from ultralytics.nn.modules.block import Conv
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
# https://github.com/iscyy/ultralyticsPro
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
################# 
class FasterNeXt(nn.Module):
    def __init__(self, c1, c2, n=1, extra=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1) 
        self.m = nn.Sequential(*(FasterNetBlock(self.c, self.c) for _ in range(n)))
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class C3_FasterNeXt(nn.Module):
    # 3CFaster with 3 convolutions
    def __init__(self, c1, c2, n=1, extra=2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(FasterBlock(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C2f_FasterNeXt(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, extra=2, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  
        self.m = nn.Sequential(*(FasterNetBlock(self.c, self.c) for _ in range(n)))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CSCFasterNeXt(nn.Module):
    def __init__(self, c1, c2, n=1, extra=2, shortcut=True, k=(1, 1), g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c1, c_, k[0], 1)
        self.cv3 = Conv(c_, c_, k[0], 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(FasterBlock(c_, c_) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

class ReNBC(nn.Module):
    def __init__(self, c1, c2, n=1, extra=2, isUse=False, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        if isUse:
            self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(FasterNetBlock(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class ReNLANFasterNeXt(nn.Module):
    # ReNLANFasterNeXt Block
    def __init__(self, c1, c2, c3, c4, c=True, n=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(ReNBC(c3//2, c4, n, isUse=False))
        self.cv3 = nn.Sequential(ReNBC(c4, c4, n, isUse=False))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
import torch
import torch.nn as nn
import math
from einops import rearrange


# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc), nn.SiLU())
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
    # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽

    # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

# Add WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension into your PYTHONPATH by the following commands:
sys.path.append(
    "WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension"
)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

use_sync_bn = False
kernel_TYPEs = ["ori", "div", "group"]
kernel_TYPE = 2


def get_conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
):
    try:
        paddings = (kernel_size[0] // 2, kernel_size[1] // 2)
    except:
        paddings = padding
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, paddings, dilation, groups, bias
    )


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn_relu_ori(
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1
):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
    )
    result.add_module("nonlinear", nn.ReLU())
    return result


def conv_bn_ori(
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True
):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
    )

    if bn:
        result.add_module("bn", get_bn(out_channels))
    return result


class conv_bn_div(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation=1,
            bn=True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.result = nn.Sequential()
        self.result.add_module(
            "conv",
            get_conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
        )

        if bn:
            self.result.add_module("bn", get_bn(out_channels))

    def forward(self, inputs):
        out = self.result(inputs)
        return out


class conv_bn_relu_div(conv_bn_div):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation=1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
        )
        self.result.add_module("nonlinear", nn.ReLU())

class ConvGroupShift(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel, stride=1, group=1,
                 bn=True, weight=None):
        super().__init__()
        assert len(set(kernel)) == 2, "must have two different kernel size"
        mink, maxk = min(kernel), max(kernel)
        self.kernels = kernel
        self.stride = stride
        if (mink, maxk) == kernel:
            self.VH = 'H'
        else:
            self.VH = 'V'
        padding, after_padding_index, index = self.shift(kernel)
        padding = (padding, mink // 2) if self.VH == 'V' else (mink // 2, padding)
        self.pad = after_padding_index, index
        print(padding, after_padding_index, index)
        self.nk = math.ceil(maxk / mink)
        self.split_convs = nn.Conv2d(in_channels, out_channels * self.nk,
                                     kernel_size=mink,  stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)
        self.reloc_weight(weight)
        self.use_bn = bn
        if bn:
            self.bn = get_bn(out_channels)


    def shift(self, kernels):
        '''
        Regardless of the step size, the convolution can slide the window to the boundary at most
        '''
        mink, maxk = min(kernels), max(kernels)
        mid_p = maxk // 2
        offset_idx_left = mid_p % mink
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink
        # 2. padding
        padding = offset_idx_left % mink
        while padding < offset_idx_right:
            padding += mink
        # 3. make sure last pixel can be scan by min window
        while padding < (mink - 1):
            padding += mink
        # 4. index of windows start point of middle point
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink
        return padding, after_padding_index, real_start_idx

    def forward(self, inputs):
        out = self.split_convs(inputs)
        b, c, h, w = out.shape
        # split output
        *_, ori_h, ori_w = inputs.shape
        # out = torch.split(out, c // self.nk, 1)
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        x = 0
        for i in range(self.nk):
            outi = self.rearrange_data(out[i], i, ori_h, ori_w)
            x = x + outi
        if self.use_bn:
            x = self.bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w):
        pad, index = self.pad
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        if (idx + 1) >= index:
            pad_l = 0
            s = (idx + 1 - index) * (k // stride)
        else:
            pad_l = (index - 1 - idx) * (k // stride)
            s = 0
        if self.VH == 'H':
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            e = w + pad_l + pad_r - s - suppose_len
        else:
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            e = h + pad_l + pad_r - s - suppose_len
        # print('new_pad', new_pad)
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
        split_list = [s, suppose_len, e]
        # print('split_list', split_list)
        xs = torch.split(x, split_list, dim=dim)
        return xs[1]

    def reloc_weight(self, w):
        if w is None: return
        c1, c2, k1, k2 = self.split_convs.weight.data.shape

        if self.VH == 'H':
            pad_r = k1 - w.shape[3] % k1
            w = F.pad(w, (0, pad_r, 0, 0))
            w = torch.split(w.unsqueeze(1), k1, dim=3 + 1)  # ※※※※※※※※
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)  # ※※※※※※※※
        else:
            pad_r = k1 - w.shape[2] % k1
            w = F.pad(w, (0, 0, 0, pad_r))
            w = torch.split(w.unsqueeze(1), k1, dim=2 + 1)
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)
        self.split_convs.weight.data = w


# 5. mini kernel as main character
class ConvGroupShiftTaichi(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel, stride=1, group=1,
                 bn=True, weight=None):
        super().__init__()
        assert len(set(kernel)) == 2, "must have two different kernel size"
        mink, maxk = min(kernel), max(kernel)
        self.kernels = kernel
        self.stride = stride
        if (mink, maxk) == kernel:
            self.VH = 'H'  # 横向
        else:
            self.VH = 'V'
        padding, after_padding_index, index = self.shift(kernel)
        # padding = (padding, mink // 2) if self.VH == 'V' else (mink // 2, padding)
        if self.VH == 'V':
            padding = (padding, mink // 2)
            self.another_pad=mink // 2-mink//2
        else:
            padding = (mink // 2, padding)
            self.another_pad=mink // 2-mink//2
        self.pad = after_padding_index, index
        print(padding, after_padding_index, index)
        self.nk = math.ceil(maxk / mink)
        self.split_convs = nn.Conv2d(in_channels, out_channels * self.nk,
                                     kernel_size=mink,  stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)
        self.reloc_weight(weight)
        self.use_bn = bn
        if bn:
            self.bn = get_bn(out_channels)
        self.pads=self.getpads()
        self.idxs=torch.arange(out_channels * self.nk).reshape(out_channels,-1)
        self.datas=None

    def shift(self, kernels):
        '''
        Regardless of the step size, the convolution can slide the window to the boundary at most
        '''
        mink, maxk = min(kernels), max(kernels)
        mid_p = maxk // 2
        # 1. new window size is mink. middle point index in the window
        offset_idx_left = mid_p % mink
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink
        # 2. padding
        padding = offset_idx_left % mink
        while padding < offset_idx_right:
            padding += mink
        # 3. make sure last pixel can be scan by min window
        while padding < (mink - 1):
            padding += mink
        # 4. index of windows start point of middle point
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink
        return padding, after_padding_index, real_start_idx

        # todo:stride 时候的extra_padding

    def forward(self, inputs):
        out = self.split_convs(inputs)
        BB=len(inputs)
        if self.datas is None:
            b, c, h, w = out.shape
            *_, ori_h, ori_w = inputs.shape
            ori_k = max(self.kernels)
            ori_p = ori_k // 2
            W=(ori_w + 2 * ori_p - ori_k) // self.stride + 1
            H=(ori_h + 2 * ori_p - ori_k) // self.stride + 1
            c1=c//self.nk
            t=self.another_pad
            isH =1 if self.VH == 'H' else 0
            self.datas=(c1, c, H, W, h, w, t,isH)
        else:
            c1, c, H, W, h, w, t, isH=self.datas

        if self.idxs.device!=out.device:
            self.idxs=self.idxs.cuda()
            self.pads=self.pads.cuda()
        x = RUN_TAICHI(out, self.idxs, self.pads, BB, c1, c, H, W, h, w, t, isH)
        if self.use_bn:
            x = self.bn(x)
        return x

    def getpads(self):
        # get index and corresponding pads for every group
        pads = []
        for idx in range(self.nk):
            pad, index = self.pad
            # x = x.squeeze(2)  # ※※※※※※※
            # *_, h, w = x.shape
            k = min(self.kernels)
            ori_k = max(self.kernels)
            ori_p = ori_k // 2
            stride = self.stride
            # need to calculate start point after conv
            # how many windows shift from real start window index
            if (idx + 1) >= index:
                pads.append(-(idx + 1 - index) * (k // stride))
            else:
                pads.append((index - 1 - idx) * (k // stride))
        pads=torch.IntTensor(pads)
        return pads


    def reloc_weight(self, w):
        if w is None: return
        c1, c2, k1, k2 = self.split_convs.weight.data.shape

        if self.VH == 'H':
            pad_r = k1 - w.shape[3] % k1
            w = F.pad(w, (0, pad_r, 0, 0))
            w = torch.split(w.unsqueeze(1), k1, dim=3 + 1)  # ※※※※※※※※
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)  # ※※※※※※※※
        else:
            pad_r = k1 - w.shape[2] % k1
            w = F.pad(w, (0, 0, 0, pad_r))
            w = torch.split(w.unsqueeze(1), k1, dim=2 + 1)
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)
        self.split_convs.weight.data = w


def conv_bn_group(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    # if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:
    if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:
        return conv_bn_ori(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bn)
    else:
        return ConvGroupShift(in_channels, out_channels, kernel_size, stride, groups, bn)
        # return ConvGroupShiftTaichi(in_channels, out_channels, kernel_size, stride, groups, bn)

if kernel_TYPE == 0:
    conv_bn_relu = conv_bn_relu_ori
    conv_bn = conv_bn_ori
elif kernel_TYPE == 1:
    conv_bn_relu = conv_bn_relu_div
    conv_bn = conv_bn_div
else:
    conv_bn = conv_bn_group


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            small_kernel,
            small_kernel_merged=False,
            Decom=False,
            bn=True,
    ):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:  # cpp版本的conv，加快速度
            self.lkb_reparam = get_conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            if self.Decom:
                self.LoRA1 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, small_kernel),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )
                self.LoRA2 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(small_kernel, kernel_size),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )
            else:
                self.lkb_origin = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )

            if (small_kernel is not None) and small_kernel < kernel_size:
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bn=bn,
                )

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            out = self.LoRA1(inputs) + self.LoRA2(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


class ReparamLargeKernelBlock(nn.Module):
    r"""SLaK_reg ReparamLargeKernelBlock. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            drop_path=0.0,
            layer_scale_init_value=1e-6,
            kernel_size=(7, 7),
            Decom=None,
            bn=True,
    ):
        super().__init__()

        self.large_kernel = ReparamLargeKernelConv(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size[0],
            stride=1,
            groups=dim,
            small_kernel=kernel_size[1],
            small_kernel_merged=False,
            Decom=Decom,
            bn=bn,
        )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.large_kernel(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

from ultralytics.nn.modules.conv import Conv
class ReparamLKBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ReparamLargeKernelBlock(self.c) for _ in range(n))
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
