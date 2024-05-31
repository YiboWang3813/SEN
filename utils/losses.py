# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch

from typing import Tuple, List, Dict 


def Warper3d(img, flow):
    B, _, D, H, W = img.shape
    # mesh grid
    xx = torch.arange(0, W).view(1,1,-1).repeat(D,H,1).view(1,D,H,W)
    yy = torch.arange(0, H).view(1,-1,1).repeat(D,1,W).view(1,D,H,W)
    zz = torch.arange(0, D).view(-1,1,1).repeat(1,H,W).view(1,D,H,W)
    grid = torch.cat((xx,yy,zz),0).repeat(B,1,1,1,1).float().to(img.device) # [bs, 3, D, H, W]
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:,0] = 2.0*vgrid[:,0]/(W-1)-1.0 # max(W-1,1)
    vgrid[:,1] = 2.0*vgrid[:,1]/(H-1)-1.0 # max(H-1,1)
    vgrid[:,2] = 2.0*vgrid[:,2]/(D-1)-1.0 # max(D-1,1)
    vgrid = vgrid.permute(0,2,3,4,1) # [bs, D, H, W, 3]        
    output = F.grid_sample(img, vgrid, padding_mode='border')

    return output
    
def Dice(segment, label, return_dice=False):
    N = segment.shape[0]
    segment_flat = segment.view(N, -1)
    label_flat = label.view(N, -1)
    intersection = segment_flat * label_flat 
    dice = (2 * intersection.sum(1) + 0.001) / (segment_flat.sum(1) + label_flat.sum(1) + 0.001)
    if return_dice:
        return dice.mean()
    loss = 1 - dice.mean()
    return loss

def Gradient(feild):
    g_H = (feild[:, :, 1:, :, :]-feild[:, :, :-1, :, :]).abs()
    g_W = (feild[:, :, :, 1:, :]-feild[:, :, :, :-1, :]).abs()
    g_D = (feild[:, :, :, :, 1:]-feild[:, :, :, :, :-1]).abs()
    loss = g_H.mean() + g_W.mean() + g_D.mean()
    return loss

def Jac(feild):
    D_x = (feild[:, :, :-1, :-1, 1:]-feild[:, :, :-1, :-1, :-1])
    D_y = (feild[:, :, :-1, 1:, :-1]-feild[:, :, :-1, :-1, :-1])
    D_z = (feild[:, :, 1:, :-1, :-1]-feild[:, :, :-1, :-1, :-1])

    D1 = (D_x[:,0,...]+1)*((D_y[:,1,...]+1)*(D_z[:,2,...]+1) - D_y[:,2,...]*D_z[:,1,...])
    D2 = (D_x[:,1,...])*(D_y[:,2,...]*D_z[:,0,...] - D_y[:,0,...]*(D_z[:,2,...]+1))
    D3 = (D_x[:,2,...])*(D_y[:,0,...]*D_z[:,1,...] - (D_y[:,1,...]+1)*D_z[:,0,...])

    D = D1 + D2 + D3

    return D

def Jac_loss(feild):
    Jac_mat = Jac(feild)
    loss = F.relu(-Jac_mat).mean()
    return loss

def Ncc(Ii, Ji):
    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)


class GradLoss(nn.Module):
    """ Compute the gradient loss of the deformation field. 
    This loss is always used to regularize generated DVF, which should be accompanied with a lambda weight. 
    
    Parameters: 
        penalty (str): choose which norm method will be used (||\cdot||_1, ||\cdot||_2) """
    def __init__(self, penalty: str = 'l2') -> None: 
        super().__init__() 

        assert penalty in ('l1', 'l2'), "Unsupported penalty method in GradientLoss"
        self.penalty = penalty

    def forward(self, x: torch.Tensor) -> float: 
        """ Forward function. 
        
        Inputs:     
            x (Tensor): 5d deformation field [B, 3, D, H, W] """
        dh = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dw = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dd = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dh = dh ** 2 
            dw = dw ** 2 
            dd = dd ** 2 
            
        loss = (torch.mean(dh) + torch.mean(dw) + torch.mean(dd)) / 3.0 
        return loss 


class SingleGradLoss(nn.Module): 
    def __init__(self) -> None:
        super().__init__() 

    def forward(self, feild): 
        g_H = (feild[:, :, 1:, :, :]-feild[:, :, :-1, :, :]).abs()
        g_W = (feild[:, :, :, 1:, :]-feild[:, :, :, :-1, :]).abs()
        g_D = (feild[:, :, :, :, 1:]-feild[:, :, :, :, :-1]).abs()
        loss = g_H.mean() + g_W.mean() + g_D.mean()
        return loss


class NCCLoss(nn.Module): 
    """ Compute the 3d local window-based normalized cross correlation loss.
    This loss is used to increase similarity between fixed image and warped image. 
    
    Parameters: 
        in_channels (int): input channels of fixed and warped images tensor 
        kernel (tuple): convolutional kernel size used in conv3d to achieve fast correlation computation """ 
    def __init__(self, in_ch=1, kernel=(9, 9, 9), device=torch.device('cuda:0')): 
        # type: (int, Tuple[int, int, int], torch.device) -> None 
        super().__init__() 

        self.filt = torch.ones((1, in_ch, kernel[0], kernel[1], kernel[2])).to(device) 
        self.padding = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2), int((kernel[2] - 1) / 2))
        self.k_sum = kernel[0] * kernel[1] * kernel[2]
 
    def forward(self, I: torch.Tensor, J: torch.Tensor) -> float: 
        """ Forward function. 
        
        Inputs: 
            I (Tensor): fixed image tensor [B, 1, D, H, W] 
            J (Tensor): warped image tensor [B, 1, D, H, W] """
        II = I * I 
        JJ = J * J 
        IJ = I * J 

        I_sum = F.conv3d(I, self.filt, stride=1, padding=self.padding)
        J_sum = F.conv3d(J, self.filt, stride=1, padding=self.padding)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=self.padding)
        JJ_sum = F.conv3d(JJ, self.filt, stride=1, padding=self.padding)
        IJ_sum = F.conv3d(IJ, self.filt, stride=1, padding=self.padding)

        I_u = I / self.k_sum
        J_u = J / self.k_sum

        cross = IJ_sum - I_sum * J_u - J_sum * I_u + I_u * J_u * self.k_sum
        I_var = II_sum - 2 * I_sum * I_u + I_u * I_u * self.k_sum
        J_var = JJ_sum - 2 * J_sum * J_u + J_u * J_u * self.k_sum

        top = cross * cross
        bottom = I_var * J_var + 1e-5 

        loss = top / bottom
        return -torch.mean(loss)


class SingleNCCLoss(nn.Module): 
    def __init__(self) -> None:
        super().__init__() 

    def forward(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    

class MultiScaleNCCLoss(nn.Module): 
    """ 对一个三维体直接做内部自生成的多尺度的NCC """
    def __init__(self, num_scales=3, in_ch=1, kernel=(9, 9, 9), device=torch.device('cuda:0')):
        super().__init__() 

        self.num_scales = num_scales 
        self.ncc_modules = [] 

        for i in range(num_scales): 
            kernel_per_scale = (kernel[0] - i * 2, kernel[1] - i * 2, kernel[2] - i * 2)  
            self.ncc_modules.append(
                NCCLoss(in_ch, kernel_per_scale, device)  
            ) 

    def forward(self, I, J): 
        total_ncc = [] 

        for i in range(self.num_scales): 
            current_ncc = self.ncc_modules[i](I, J) 
            total_ncc.append(current_ncc / 2 ** i) 

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)  
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False) 

        return sum(total_ncc) 


def conv3x3_leakyrelu(in_channels, out_channels, stride=1, groups=1, dilation=1, leaky=0.2):  
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, 
                  groups=groups, padding=dilation, dilation=dilation, bias=False), 
        nn.LeakyReLU(leaky, inplace=True)   
    ) 

def normalize_feature(feature: torch.Tensor) -> torch.Tensor: 
    B, C, D, H, W = feature.shape 
    res = [] 
    for c in range(C): 
        feature_c = feature[:, c, ...] 
        _min, _max = torch.min(feature_c), torch.max(feature_c) 
        feature_c = (feature_c - _min) / (_max - _min) 
        res.append(feature_c.reshape(B, 1, D, H, W)) 
    return torch.cat(res, dim=1) 

class PydLoss(nn.Module): 
    _levels_to_channels_mapping_dict = {
        (1, 2): (8, 16, 32), 
        (2, 3): (16, 32, 32), 
        (3, 4): (32, 32, 32), 
        (4, 5): (32, 32, 32) 
    }

    def __init__(self, pre_level: int, cur_level: int, normal: bool = True):  
        super().__init__() 

        in_channels, out_channels, inter_channels = self._levels_to_channels_mapping_dict[(pre_level, cur_level)] 
        self.enc_converter = conv3x3_leakyrelu(in_channels, out_channels, stride=2) 
        self.dec_converter = conv3x3_leakyrelu(in_channels, out_channels, stride=2) 
        self.enc_inter_converter = conv3x3_leakyrelu(out_channels, inter_channels, stride=2) 
        self.dec_inter_converter = conv3x3_leakyrelu(out_channels, inter_channels, stride=2) 
        self.normal = normal 

    def forward(self, enc_feature_pair, inter_feature, dec_feature_pair): 

        # encoder 
        pre_enc_feature, cur_enc_feature = enc_feature_pair # bigger, smaller 
        pre_enc_feature = self.enc_converter(pre_enc_feature) 
        if self.normal: 
            enc_feature_diff = normalize_feature(pre_enc_feature) - normalize_feature(cur_enc_feature) 
        else: 
            enc_feature_diff = pre_enc_feature - cur_enc_feature 
        enc_loss = torch.norm(enc_feature_diff, p=2) # L2-norm 

        # decoder 
        cur_dec_feature, pre_dec_feature = dec_feature_pair # smaller, bigger 
        pre_dec_feature = self.dec_converter(pre_dec_feature) 
        if self.normal: 
            dec_feature_diff = normalize_feature(pre_dec_feature) - normalize_feature(cur_dec_feature) 
        else: 
            dec_feature_diff = pre_dec_feature - cur_dec_feature 
        dec_loss = torch.norm(dec_feature_diff, p=2) 

        # encoder to intermediate 
        pre_enc_feature = self.enc_inter_converter(pre_enc_feature) 
        if self.normal: 
            enc_inter_feature_diff = normalize_feature(pre_enc_feature) - normalize_feature(inter_feature) 
        else: 
            enc_inter_feature_diff = pre_enc_feature - inter_feature 
        enc_inter_loss = torch.norm(enc_inter_feature_diff, p=2) 

        # decoder to intermediate 
        pre_dec_feature = self.dec_inter_converter(pre_dec_feature) 
        if self.normal: 
            dec_inter_feature_diff = normalize_feature(pre_dec_feature) - normalize_feature(inter_feature) 
        else: 
            dec_inter_feature_diff = pre_dec_feature - inter_feature 
        dec_inter_loss = torch.norm(dec_inter_feature_diff, p=2) 

        return (enc_loss + dec_loss + enc_inter_loss + dec_inter_loss) / 4   
    

class CasLoss(nn.Module): 
    def __init__(self, flag: str = "mean_three_dims_together"):  
        super().__init__() 

        self.flag = flag 
    
    def forward(self, pre_feature_list, cur_feature_list): 

        res = [] 
        for pre_feature, cur_feature in zip(pre_feature_list, cur_feature_list): 
            loss = None 
            if self.flag == "mean_three_dims_respective": 
                pre_d, pre_h, pre_w = self._mean_feature_on_three_dims_respective(pre_feature) 
                cur_d, cur_h, cur_w = self._mean_feature_on_three_dims_respective(cur_feature) 
                loss = self._cal_sym_kl_div(pre_d, cur_d, -1) + \
                       self._cal_sym_kl_div(pre_h, cur_h, -1) + \
                       self._cal_sym_kl_div(pre_w, cur_w, -1) 
                loss /= 3 
            elif self.flag == "mean_three_dims_together": 
                pre_dhw = self._mean_feature_on_three_dims_together(pre_feature) 
                cur_dhw = self._mean_feature_on_three_dims_together(cur_feature) 
                loss = self._cal_sym_kl_div(pre_dhw, cur_dhw, -1) # for channels   
            else: 
                loss = self._cal_sym_kl_div(pre_feature, cur_feature, -3) + \
                       self._cal_sym_kl_div(pre_feature, cur_feature, -2) + \
                       self._cal_sym_kl_div(pre_feature, cur_feature, -1) 
                loss /= 3 
            res.append(loss) 
        return torch.stack(res).mean()  

    def _mean_feature_on_three_dims_respective(self, x: torch.Tensor) -> List[torch.Tensor]: 
        x_d = torch.mean(x, dim=(-2, -1)) # depth 
        x_h = torch.mean(x, dim=(-3, -1)) # height 
        x_w = torch.mean(x, dim=(-3, -2)) # width 

        return x_d, x_h, x_w 
    
    def _mean_feature_on_three_dims_together(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.mean(x, dim=(-1, -2, -3)) 

    def _cal_sym_kl_div(self, x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor: 
        x = F.softmax(x, dim=dim) 
        y = F.softmax(y, dim=dim) 
        mid = ((x + y) / 2).log() 

        return (F.kl_div(mid, x) + F.kl_div(mid, y)) / 2 
    

class DiceLoss(nn.Module): 
    def __init__(self, return_dice: bool = False): 
        super().__init__()

        self.return_dice = return_dice 

    def forward(self, segment, label): 
        N = segment.shape[0] 
        segment_flat = segment.view(N, -1) 
        label_flat = label.view(N, -1) 
        intersection = segment_flat * label_flat 
        dice = (2 * intersection.sum(1) + 0.001) / (segment_flat.sum(1) + label_flat.sum(1) + 0.001)
        if self.return_dice:
            return dice.mean()
        loss = 1 - dice.mean()
        return loss
