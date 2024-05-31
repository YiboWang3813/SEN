import os 
import numpy as np 
import nibabel 
import matplotlib.pyplot as plt 
from PIL import Image 
import cv2 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import matplotlib.pyplot as plt
import matplotlib as mpl  
import matplotlib.cm as cm
import matplotlib.colors as mcolors
plt.rc('font',family='Times New Roman')

from typing import List, Tuple, Dict 


def _normalize(x: np.ndarray) -> np.ndarray: 
    _min, _max = np.min(x), np.max(x)  
    return (x - _min) / (_max - _min) 
    # _mean, _std = np.mean(x), np.std(x) 
    # return (x- _mean) / _std, _mean, _std


def draw_displacement_rgb(work_dir: str, which_ids: List) -> None: 
    """ Draw rgb displacement slice images using min-max normalization. 
    
    Parameters: 
        work_dir (str): experimental sample dir 
        which_ids (List): specify position to draw sliced image """
    
    # get raw displacement 
    disp_path = os.path.join(work_dir, 'disp.nii.gz') 
    assert os.path.exists(disp_path), "Lack displacement in {}".format(work_dir) 
    # get rgb displacement 
    disp_rgb_path = os.path.join(work_dir, 'disp_rgb.nii.gz') 

    # if not os.path.exists(disp_rgb_path): 

    disp = nibabel.load(disp_path).dataobj 
    disp = np.array(disp, dtype='float32') 
    print("disp shape: {}".format(disp.shape)) 

    buffer = [] 
    for c in range(disp.shape[-1]): 
        vol = _normalize(disp[..., c]) * 255
        buffer.append(vol) 
    buffer[0], buffer[1], buffer[2] = buffer[0], buffer[2], buffer[1] 
    disp_rgb = np.stack(buffer, axis=-1) # [D, H, W, 3] 
    print(disp_rgb.shape) 

    # disp_rgb, min_value, max_value = _normalize(disp) 
    # disp_rgb = disp_rgb * 255 
    # print("min: {}, max: {}".format(min_value, max_value)) 

    disp_rgb = np.array(disp_rgb, dtype='uint8') 
    # disp_rgb_nii = nibabel.Nifti1Image(disp_rgb, np.eye(4)) 
    # nibabel.save(disp_rgb_nii, os.path.join(work_dir, 'disp_rgb.nii.gz')) 
    # print("Generate rgb displacement in {}".format(work_dir)) 

    # load it and slice out 
    # disp_rgb = nibabel.load(disp_rgb_path).dataobj 
    # disp_rgb = np.array(disp_rgb, dtype='uint8') 
    D, H, W, _ = disp_rgb.shape 

    d_idx, h_idx, w_idx = which_ids 
    if d_idx == None: 
        d_idx = D // 2 
    if h_idx == None: 
        h_idx = H // 2 
    if w_idx == None: 
        w_idx = W // 2 

    slice_d = disp_rgb[d_idx, :, :, :] 
    # plt.figure() 
    # plt.imshow(slice_d) 
    # plt.savefig(os.path.join(work_dir, 'Disp_RGB_D_{}.png'.format(d_idx)))

    slice_d = Image.fromarray(slice_d) 
    slice_d.save(os.path.join(work_dir, 'Disp_RGB_D_{}.png'.format(d_idx))) 

    # plt.figure() 
    # plt.imshow(slice_d) 
    # plt.savefig(os.path.join(work_dir, 'Disp_RGB_D_{}.png'.format(d_idx)), bbox_inches='tight')  

    slice_h = disp_rgb[:, h_idx, :, :] 
    slice_h = Image.fromarray(slice_h) 
    slice_h.save(os.path.join(work_dir, 'Disp_RGB_H_{}.png'.format(h_idx)))  

    # plt.figure() 
    # plt.imshow(slice_h) 
    # plt.savefig(os.path.join(work_dir, 'Disp_RGB_H_{}.png'.format(h_idx)), bbox_inches='tight')  

    slice_w = disp_rgb[:, :, w_idx, :] 
    slice_w = Image.fromarray(slice_w) 
    slice_w.save(os.path.join(work_dir, 'Disp_RGB_W_{}.png'.format(w_idx)))  

    # plt.figure() 
    # plt.imshow(slice_w) 
    # plt.savefig(os.path.join(work_dir, 'Disp_RGB_W_{}.png'.format(w_idx)), bbox_inches='tight')  

    print("Generate RGB displacement slices in {}".format(work_dir))


def draw_jacobian_determinant_colormap(work_dir: str, which_ids: List) -> None: 
    """ Draw jacobian determinant rgb with colormap. 

    Parameters: 
        work_dir (str): experimental sample dir 
        which_ids (List): specify position to draw sliced image """
    # get raw displacement 
    jdet_path = os.path.join(work_dir, 'jdet.nii.gz') 
    assert os.path.exists(jdet_path), "Lack jacobian determinant in {}".format(work_dir) 
    jdet = nibabel.load(jdet_path).dataobj 
    jdet = np.array(jdet, dtype='float32') 

    # prepare to draw 
    D, H, W = jdet.shape 
    
    d_idx, h_idx, w_idx = which_ids 
    if d_idx == None: 
        d_idx = D // 2 
    if h_idx == None: 
        h_idx = H // 2 
    if w_idx == None: 
        w_idx = W // 2 

    jdet = np.clip(jdet, -1, 6)  
    
    # set colormap 
    newcmap=(plt.get_cmap('bwr')) 
    bins = np.array([-1, 0, 1, 2, 3, 4, 5, 6]) 
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=1,vmax=6) 
    # fig, ax = plt.subplots() 
    # ax.imshow(slice_d, cmap='bwr')    
    # ax=fig.add_axes([0,0,0.5,0.05])  
    # fc1=fig.colorbar(
    #             mpl.cm.ScalarMappable(norm=norm,cmap=newcmap,),   
    #             ticks = bins,                                               
    #             orientation='vertical',                          
    #              )    
    # cmap = mcolors.LinearSegmentedColormap.from_list('xmap', mpl.cm.ScalarMappable(norm=norm,cmap=newcmap,))  

    # get slice at d_idx 
    slice_d = jdet[d_idx, :, :] 
    slice_d = np.rot90(slice_d) 
    x_ticks = np.linspace(0, slice_d.shape[1], 6, dtype='int16') 
    y_ticks = np.linspace(0, slice_d.shape[0], 8, dtype='int16')  
    plt.figure(dpi=600, figsize=(2, 1.5))  
    plt.imshow(slice_d, cmap='bwr', norm=norm)  
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcmap,),   
                ticks=bins,                                               
                orientation='vertical',
                fraction=0.037)  
    colorbar.ax.tick_params(labelsize=6)
    plt.xticks(x_ticks, fontsize=6) 
    plt.yticks(y_ticks, fontsize=6) 
    plt.savefig(os.path.join(work_dir, 'Jdet_Colormap_D_{}.png'.format(d_idx)), bbox_inches='tight', pad_inches=0.01) 

    slice_h = jdet[:, h_idx, :] 
    slice_h = np.rot90(slice_h) 
    x_ticks = np.linspace(0, slice_h.shape[1], 6, dtype='int16') 
    y_ticks = np.linspace(0, slice_h.shape[0], 8, dtype='int16')  
    plt.figure(dpi=600, figsize=(2, 1.5))  
    plt.imshow(slice_h, cmap='bwr', norm=norm) 
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcmap,),   
                ticks=bins,                                               
                orientation='vertical',
                fraction=0.037)  
    colorbar.ax.tick_params(labelsize=6)
    plt.xticks(x_ticks, fontsize=6) 
    plt.yticks(y_ticks, fontsize=6) 
    plt.savefig(os.path.join(work_dir, 'Jdet_Colormap_H_{}.png'.format(h_idx)), bbox_inches='tight', pad_inches=0.01) 

    slice_w = jdet[:, :, w_idx] 
    slice_w = np.rot90(slice_w) 
    x_ticks = np.linspace(0, slice_w.shape[1], 6, dtype='int16') 
    y_ticks = np.linspace(0, slice_w.shape[0], 8, dtype='int16')  
    plt.figure(dpi=600, figsize=(2, 1.5))  
    plt.imshow(slice_w, cmap='bwr', norm=norm) 
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcmap,),   
                ticks=bins,                                               
                orientation='vertical',
                fraction=0.037)  
    colorbar.ax.tick_params(labelsize=6)
    plt.xticks(x_ticks, fontsize=6) 
    plt.yticks(y_ticks, fontsize=6) 
    plt.savefig(os.path.join(work_dir, 'Jdet_Colormap_W_{}.png'.format(w_idx)), bbox_inches='tight', pad_inches=0.01) 

    print("Generate Colormap Jacobian Determinant slices in {}".format(work_dir))


def draw_segmentation_rgb(work_dir: str, which_ids: List, mapping_dict: Dict) -> None: 
    """ Draw seperated rgb segmentation images. 
    
    Parameters: 
        work_dir (str): experimental sample dir 
        which_ids (List): specify position to draw sliced image 
        mapping_dict (Dict): RGB color mapping dict which can be loaded from DataLoader package """
    
    label_names = ['fixed_label.nii.gz', 'moving_label.nii.gz', 'warped_label.nii.gz'] 
    label_rgb_names = ['fixed_label_rgb.nii.gz', 'moving_label_rgb.nii', 'warped_label_rgb.nii'] 

    for label_idx, (label_name, label_rgb_name) in enumerate(zip(label_names, label_rgb_names)): 
        label_path = os.path.join(work_dir, label_name) 
        assert os.path.exists(label_path), "Lack label {} in work dir {}".format(label_name, work_dir) 
        label = nibabel.load(label_path).dataobj 
        label = np.array(label, dtype='int16') 

        # find corresponding rgb file 
        label_rgb_path = os.path.join(work_dir, label_rgb_name) 
        if not os.path.exists(label_rgb_path): 
            # generate it and save it 
            D, H, W = label.shape 
            
            mapping_keys = list(mapping_dict.keys()) 
            label_r = np.zeros((D, H, W), dtype='uint8') 
            label_g = np.zeros((D, H, W), dtype='uint8') 
            label_b = np.zeros((D, H, W), dtype='uint8') 

            # loop all regions and add color value 
            for i in range(len(mapping_keys)): 
                label_c = (label == i + 1) # mask 

                channel_mapping = mapping_dict[mapping_keys[i]] # list 
                rm, gm, bm = channel_mapping # int, int, int 
            
                label_r_old = label_r * ~label_c # add old color regions 
                label_r_new = rm * label_c # add new color region 
                label_r = label_r_old + label_r_new 

                label_g_old = label_g * ~label_c 
                label_g_new = gm * label_c 
                label_g = label_g_old + label_g_new 

                label_b_old = label_b * ~label_c 
                label_b_new = bm * label_c 
                label_b = label_b_old + label_b_new 
                
            label_rgb = np.stack([label_r, label_g, label_b], axis=3) # [D, H, W, 3] 
            label_rgb_affine = np.eye(4) 
            label_rgb_nii = nibabel.Nifti1Image(label_rgb, label_rgb_affine) 
            nibabel.save(label_rgb_nii, label_rgb_path) 

            print("Generate rgb label {} in {}".format(label_rgb_name, work_dir)) 

        # read rgb label and slice it 
        label_rgb = nibabel.load(label_rgb_path).dataobj 
        label_rgb = np.array(label_rgb, dtype='uint8') 

        D, H, W, _ = label_rgb.shape 

        d_idx, h_idx, w_idx = which_ids 
        if d_idx == None: 
            d_idx = D // 2 
        if h_idx == None: 
            h_idx = H // 2 
        if w_idx == None: 
            w_idx = W // 2 

        label_name = label_name.replace('.nii.gz', '') 
        # get slice d 
        slice_d = label_rgb[d_idx, :, :, :] 
        slice_d = Image.fromarray(slice_d) 
        slice_d.save(os.path.join(work_dir, '{}_RGB_D_{}.png'.format(label_name, d_idx)))  

        # get slice h 
        slice_h = label_rgb[:, h_idx, :, :] 
        slice_h = Image.fromarray(slice_h) 
        slice_h.save(os.path.join(work_dir, '{}_RGB_H_{}.png'.format(label_name, h_idx))) 

        # get slice w  
        slice_w = label_rgb[:, :, w_idx, :] 
        slice_w = Image.fromarray(slice_w) 
        slice_w.save(os.path.join(work_dir, '{}_RGB_W_{}.png'.format(label_name, w_idx))) 

        print("Generate {} slices in {}".format(label_name, work_dir))


def draw_contour_rgb(work_dir: str, which_ids: List, mapping_dict: Dict, wanted_region_name_color_dict: Dict) -> None: 
    """ Draw label contour overlapped images. 
    
    Parameters: 
        work_dir (str): experimental sample dir 
        which_ids (List): specify position to draw sliced image 
        mapping_dict (Dict): RGB color mapping dict which can be loaded from DataLoader package
        wanted_region_name_color_dict (Dict): wanted region name as key and wanted color list as their value """
    # find needed color using name  
    needed_color_list = [] 
    needed_label_idx_list = [] 
    color_region_name_list = list(mapping_dict.keys())
    for name, color in wanted_region_name_color_dict.items(): 
        color_idx = color_region_name_list.index(name) # find needed color index using name 
        needed_color_list.append(tuple(color)) # add color list 
        needed_label_idx_list.append(color_idx + 1) # add label index 
    print("Find needed color done in {}".format(work_dir)) 
    print(needed_color_list)

    image_names = ['fixed_image.nii.gz', 'moving_image.nii.gz', 'warped_image.nii.gz']
    label_names = ['fixed_label.nii.gz', 'moving_label.nii.gz', 'warped_label.nii.gz'] 

    for idx, (image_name, label_name) in enumerate(zip(image_names, label_names)): 
        image_path = os.path.join(work_dir, image_name) 
        label_path = os.path.join(work_dir, label_name) 
        assert os.path.exists(image_path) and os.path.exists(label_path), \
        "Lack fixed image or label in {}".format(work_dir) 
        image = nibabel.load(image_path).dataobj 
        image = np.array(image, dtype='uint8')  
        label = nibabel.load(label_path).dataobj 
        label = np.array(label, dtype='int16') 
        print("Load image and label in {}".format(work_dir))  

        # find needed slice index 
        D, H, W = image.shape 
        d_idx, h_idx, w_idx = which_ids 
        if d_idx == None: 
            d_idx = D // 2 
        if h_idx == None: 
            h_idx = H // 2 
        if w_idx == None: 
            w_idx = W // 2 

        # save gray level image 
        image_name = image_name.replace('.nii.gz', '') 

        slice_d = image[d_idx, :, :] 
        slice_d = Image.fromarray(slice_d) 
        slice_d.save(os.path.join(work_dir, '{}_Slice_D_{}.jpg'.format(image_name, d_idx)))  

        slice_h = image[:, h_idx, :] 
        slice_h = Image.fromarray(slice_h) 
        slice_h.save(os.path.join(work_dir, '{}_Slice_H_{}.jpg'.format(image_name, h_idx)))  

        slice_w = image[:, :, w_idx] 
        slice_w = Image.fromarray(slice_w) 
        slice_w.save(os.path.join(work_dir, '{}_Slice_W_{}.jpg'.format(image_name, w_idx)))  
        print("Generate {} gray slice in {}".format(image_name, work_dir)) 

        # extract needed label binary image and corresponding gray level image slice 
        label_name = label_name.replace('.nii.gz', '') 

        mask_d_list = [] 
        mask_h_list = [] 
        mask_w_list = [] 
        for label_idx in needed_label_idx_list: 
            mask_d_list.append(label[d_idx, :, :] == label_idx) 
            mask_h_list.append(label[:, h_idx, :] == label_idx) 
            mask_w_list.append(label[:, :, w_idx] == label_idx) 
        
        for i in range(len(mask_d_list)): 
            mask_d = mask_d_list[i] 
            # print(mask_d.dtype) 
            mask_d = np.array(mask_d, dtype='uint8') 
            mask_d = mask_d * 255 
            # mask_d = Image.fromarray(mask_d) 
            # mask_d.save(os.path.join(work_dir, '{}_D_{}_Idx_{}.jpg'.format(label_name, d_idx, i)))    
            cv2.imwrite(os.path.join(work_dir, '{}_D_{}_Idx_{}.jpg'.format(label_name, d_idx, i)), mask_d)  

            mask_h = mask_h_list[i] 
            mask_h = np.array(mask_h, dtype='uint8') 
            mask_h = mask_h * 255 
            # mask_h = Image.fromarray(mask_h) 
            # mask_h.save(os.path.join(work_dir, '{}_H_{}_Idx_{}.jpg'.format(label_name, h_idx, i)))  
            cv2.imwrite(os.path.join(work_dir, '{}_H_{}_Idx_{}.jpg'.format(label_name, h_idx, i)), mask_h)  

            mask_w = mask_w_list[i] 
            mask_w = np.array(mask_w, dtype='uint8') 
            mask_w = mask_w * 255 
            # mask_w = Image.fromarray(mask_w) 
            # mask_w.save(os.path.join(work_dir, '{}_W_{}_Idx_{}.jpg'.format(label_name, w_idx, i)))  
            cv2.imwrite(os.path.join(work_dir, '{}_W_{}_Idx_{}.jpg'.format(label_name, w_idx, i)), mask_w)  
        print("Generate {} needed label mask in {}".format(label_name, work_dir)) 

        # read gray slice and its masks together 
        slice_d = cv2.imread(os.path.join(work_dir, '{}_Slice_D_{}.jpg'.format(image_name, d_idx))) 
        mask_d_list = [] 
        for i in range(len(needed_color_list)): 
            mask_d_list.append(cv2.imread(os.path.join(work_dir, '{}_D_{}_Idx_{}.jpg'.format(label_name, d_idx, i)), cv2.IMREAD_GRAYSCALE)) 
        for mask_d, color in zip(mask_d_list, needed_color_list): 
            mask_d_copy = mask_d 
            mask_d_copy[mask_d > 225] = 255 
            mask_d_copy[mask_d < 225] = 0 
            contours, hierarchy = cv2.findContours(mask_d_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice_d = cv2.drawContours(slice_d, contours, -1, color, 1) 
        cv2.imwrite(os.path.join(work_dir, '{}_overlapped_RGB_D_{}.jpg'.format(image_name, d_idx)), slice_d) 

        slice_h = cv2.imread(os.path.join(work_dir, '{}_Slice_H_{}.jpg'.format(image_name, h_idx))) 
        mask_h_list = [] 
        for i in range(len(needed_color_list)): 
            mask_h_list.append(cv2.imread(os.path.join(work_dir, '{}_H_{}_Idx_{}.jpg'.format(label_name, h_idx, i)), cv2.IMREAD_GRAYSCALE)) 
        for mask_h, color in zip(mask_h_list, needed_color_list): 
            mask_h_copy = mask_h 
            mask_h_copy[mask_h > 225] = 255 
            mask_h_copy[mask_h < 225] = 0 
            contours, hierarchy = cv2.findContours(mask_h_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice_h = cv2.drawContours(slice_h, contours, -1, color, 1) 
        cv2.imwrite(os.path.join(work_dir, '{}_overlapped_RGB_H_{}.jpg'.format(image_name, h_idx)), slice_h) 

        slice_w = cv2.imread(os.path.join(work_dir, '{}_Slice_W_{}.jpg'.format(image_name, w_idx))) 
        mask_w_list = [] 
        for i in range(len(needed_color_list)): 
            mask_w_list.append(cv2.imread(os.path.join(work_dir, '{}_W_{}_Idx_{}.jpg'.format(label_name, w_idx, i)), cv2.IMREAD_GRAYSCALE)) 
        for mask_w, color in zip(mask_w_list, needed_color_list): 
            mask_w_copy = mask_w 
            mask_w_copy[mask_w > 225] = 255 
            mask_w_copy[mask_w < 225] = 0 
            contours, hierarchy = cv2.findContours(mask_w_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice_w = cv2.drawContours(slice_w, contours, -1, color, 1) 
        cv2.imwrite(os.path.join(work_dir, '{}_overlapped_RGB_W_{}.jpg'.format(image_name, w_idx)), slice_w) 
        print("Generate {} RGB overlapped image in {}".format(image_name, work_dir)) 

        # delete intermediate slices and masks 
        os.remove(os.path.join(work_dir, '{}_Slice_D_{}.jpg'.format(image_name, d_idx))) 
        os.remove(os.path.join(work_dir, '{}_Slice_H_{}.jpg'.format(image_name, h_idx))) 
        os.remove(os.path.join(work_dir, '{}_Slice_W_{}.jpg'.format(image_name, w_idx))) 
        for i in range(len(mask_d_list)): 
            os.remove(os.path.join(work_dir, '{}_D_{}_Idx_{}.jpg'.format(label_name, d_idx, i))) 
            os.remove(os.path.join(work_dir, '{}_H_{}_Idx_{}.jpg'.format(label_name, h_idx, i))) 
            os.remove(os.path.join(work_dir, '{}_W_{}_Idx_{}.jpg'.format(label_name, w_idx, i))) 


def warp3d(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor: 
    """ Warp image or label using deformatin field. 
    
    Parameters: 
        img (Tensor): image or label tensor [B, C, D, H, W] float32 
        flow (Tensor): deformation field [B, 3, D, H, W] float32 
        
    Returns: 
        output (Tensor): warped image or label tensor [B, C, D, H, W] float32 """
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


class SpatialTransformerBlock(nn.Module):
    """
    N-D Spatial Transformer N = 2,3
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        # F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        # torch v1.2.0 is short of attribute 'align_corners' 
        return F.grid_sample(src, new_locs, mode=self.mode)
    

def draw_displacement_grid(work_dir: str, line_width: int, vol_shape: Tuple, which_ids: List) -> None: 
    """ Draw displacement overlapped on grid images. 
    
    Parameters: 
        work_dir (str): experimental sample dir 
        line_width (int): line width 
        which_ids (List): specify position to draw sliced image """
    # get raw displacement 
    disp_path = os.path.join(work_dir, 'disp.nii.gz') 
    assert os.path.exists(disp_path), "Lack displacement in {}".format(work_dir) 
    disp = nibabel.load(disp_path).dataobj 
    disp = np.array(disp, dtype='float32') 
    print("Load displacement in {}, shape: {}".format(work_dir, disp.shape)) 

    # generate grid array 
    D, H, W, _ = disp.shape 
    grid = np.zeros((D, H, W), dtype='float32') 
    step_size = 32  
    # loop depth 
    for d_idx in range(0, D + 1, D // step_size): 
        if d_idx == D: 
            d_idx = D - 1 
        slice = grid[d_idx, :, :] # [H, W] 
        # loop height first 
        for h_idx in range(0, H + 1, H // step_size): 
            if h_idx == H: 
                slice[h_idx-line_width:h_idx, :] = 1 
            else: 
                slice[h_idx:h_idx + line_width, :] = 1 
        # loop width then 
        for w_idx in range(0, W + 1, W // step_size): 
            if w_idx == W: 
                slice[:, w_idx-line_width:w_idx] = 1 
            else: 
                slice[:, w_idx:w_idx + line_width] = 1 
        grid[d_idx, :, :] = slice 
    print("grid depth looped value: {}".format(np.unique(grid))) 

    # loop height   
    for h_idx in range(0, H + 1, H // step_size): 
        if h_idx == H: 
            h_idx = H - 1 
        slice = grid[:, h_idx, :] # [D, W]  
        # loop depth first 
        for d_idx in range(0, D + 1, D // step_size): 
            if d_idx == D: 
                slice[d_idx-line_width:d_idx, :] = 1 
            else: 
                slice[d_idx:d_idx + line_width, :] = 1 
        # loop width then 
        for w_idx in range(0, W + 1, W // step_size): 
            if w_idx == W: 
                slice[:, w_idx-line_width:w_idx] = 1 
            else: 
                slice[:, w_idx:w_idx + line_width] = 1 
        grid[:, h_idx, :] = slice 
    print("grid height looped value: {}".format(np.unique(grid))) 

    # loop width 
    for w_idx in range(0, W + 1, W // step_size): 
        if w_idx == W: 
            w_idx = W - 1 
        slice = grid[:, :, w_idx] # [D, H]  
        # loop depth first 
        for d_idx in range(0, D + 1, D // step_size): 
            if d_idx == D: 
                slice[d_idx-line_width:d_idx, :] = 1 
            else: 
                slice[d_idx:d_idx + line_width, :] = 1 
        # loop height then 
        for h_idx in range(0, H + 1, H // step_size): 
            if h_idx == H: 
                slice[:, h_idx-line_width:h_idx] = 1 
            else: 
                slice[:, h_idx:h_idx + line_width] = 1 
        grid[:, :, w_idx] = slice 
    print("grid width looped value: {}".format(np.unique(grid))) 

    # save grid 
    d_idx, h_idx, w_idx = which_ids 
    if d_idx == None: 
        d_idx = D // 2 
    if h_idx == None: 
        h_idx = H // 2 
    if w_idx == None: 
        w_idx = W // 2 

    slice_d = grid[d_idx, :, :] 
    slice_d = np.array(slice_d, dtype='uint8') 
    slice_d = slice_d * 255 
    slice_d = Image.fromarray(slice_d) 
    slice_d.save(os.path.join(work_dir, 'Disp_Grid_D_{}.png'.format(d_idx))) 

    slice_h = grid[:, h_idx, :] 
    slice_h = np.array(slice_h, dtype='uint8') 
    slice_h = slice_h * 255 
    slice_h = Image.fromarray(slice_h) 
    slice_h.save(os.path.join(work_dir, 'Disp_Grid_H_{}.png'.format(h_idx))) 

    slice_w = grid[:, :, w_idx] 
    slice_w = np.array(slice_w, dtype='uint8') 
    slice_w = slice_w * 255 
    slice_w = Image.fromarray(slice_w) 
    slice_w.save(os.path.join(work_dir, 'Disp_Grid_W_{}.png'.format(w_idx))) 
    print("Generate grid images in {}".format(work_dir))  

    # warp grid using flow 
    disp_tensor = torch.from_numpy(disp).permute(3, 0, 1, 2).unsqueeze(0) # [1, 3, D, H, W] 
    grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W] 
    # warped_grid_tensor = warp3d(grid_tensor, disp_tensor) 
    warp_func = SpatialTransformerBlock(vol_shape)  
    warped_grid_tensor = warp_func(grid_tensor, disp_tensor)  
    warped_grid = warped_grid_tensor.squeeze().numpy() # [D, H, W] 

    slice_d = warped_grid[d_idx, :, :] 
    slice_d = slice_d > 0 
    slice_d = np.array(slice_d, dtype='uint8')     
    slice_d = slice_d * 255 
    slice_d = Image.fromarray(slice_d) 
    slice_d.save(os.path.join(work_dir, 'Disp_Warped_Grid_D_{}.png'.format(d_idx))) 

    slice_h = warped_grid[:, h_idx, :] 
    slice_h = slice_h > 0 
    slice_h = np.array(slice_h, dtype='uint8')  
    slice_h = slice_h * 255 
    slice_h = Image.fromarray(slice_h) 
    slice_h.save(os.path.join(work_dir, 'Disp_Warped_Grid_H_{}.png'.format(h_idx))) 

    slice_w = warped_grid[:, :, w_idx] 
    slice_w = slice_w > 0 
    slice_w = np.array(slice_w, dtype='uint8') 
    slice_w = slice_w * 255 
    slice_w = Image.fromarray(slice_w) 
    slice_w.save(os.path.join(work_dir, 'Disp_Warped_Grid_W_{}.png'.format(w_idx))) 
    print("Generate warped grid images in {}".format(work_dir))  

    # remove intermediate grid images 
    os.remove(os.path.join(work_dir, 'Disp_Grid_D_{}.png'.format(d_idx)))
    os.remove(os.path.join(work_dir, 'Disp_Grid_H_{}.png'.format(h_idx)))
    os.remove(os.path.join(work_dir, 'Disp_Grid_W_{}.png'.format(w_idx)))


def draw_displacement_grid_contour(work_dir: str, scale_factor: float, which_ids: List) -> None: 

    # load displacement field 
    disp_path = os.path.join(work_dir, 'disp.nii.gz') 
    assert os.path.exists(disp_path), "Lack displacement in {}".format(work_dir) 
    disp = nibabel.load(disp_path).dataobj 
    disp = np.array(disp, dtype='float32') 
    print("Load displacement in {}, shape: {}".format(work_dir, disp.shape)) 

    D, H, W, _ = disp.shape 
    d_idx, h_idx, w_idx = which_ids 
    if d_idx == None: 
        d_idx = D // 2 
    if h_idx == None: 
        h_idx = H // 2 
    if w_idx == None: 
        w_idx = W // 2 
    
    step_size = 4 
    step_size_fig = 32 

    # slice d 
    x = np.arange(0, W) 
    y = np.arange(0, H) # 这要交换顺序 
    X, Y = np.meshgrid(x, y) 
    Z1 = disp[d_idx, :, :, 1] * scale_factor + X 
    Z1 = Z1[::-1] # [H, W] 
    Z2 = disp[d_idx, :, :, 2] * scale_factor + Y # [H, W] 

    plt.figure(figsize=(H // step_size_fig, W // step_size_fig)) 
    plt.contour(X, Y, Z1, H // step_size, colors='k', linewidths=0.5) 
    plt.contour(X, Y, Z2, W // step_size, colors='k', linewidths=0.5) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(os.path.join(work_dir, "Disp_Warped_Grid_D_{}.png".format(d_idx)), bbox_inches='tight', pad_inches=0.01) 

    # slice h 
    x = np.arange(0, W) 
    y = np.arange(0, D) # 这要交换顺序 
    X, Y = np.meshgrid(x, y) 
    Z1 = disp[:, h_idx, :, 0] * scale_factor + X 
    Z1 = Z1[::-1] # [D, W] 
    Z2 = disp[:, h_idx, :, 2] * scale_factor + Y # [D, W] 

    plt.figure(figsize=(D // step_size_fig, W // step_size_fig)) 
    plt.contour(X, Y, Z1, D // step_size, colors='k', linewidths=0.5) 
    plt.contour(X, Y, Z2, W // step_size, colors='k', linewidths=0.5) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(os.path.join(work_dir, "Disp_Warped_Grid_H_{}.png".format(h_idx)), bbox_inches='tight', pad_inches=0.01) 

    # slice w 
    # TODO 这个形状有点奇怪 
    x = np.arange(0, H) 
    y = np.arange(0, D) # 这要交换顺序 
    X, Y = np.meshgrid(x, y) 
    Z1 = disp[:, :, w_idx, 0] * scale_factor + X 
    Z1 = Z1[::-1] # [D, H] 
    Z2 = disp[:, :, w_idx, 1] * scale_factor + Y # [D, H] 

    plt.figure(figsize=(D // step_size_fig, H // step_size_fig)) 
    plt.contour(X, Y, Z1, D // step_size, colors='k', linewidths=0.5) 
    plt.contour(X, Y, Z2, H // step_size, colors='k', linewidths=0.5) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(os.path.join(work_dir, "Disp_Warped_Grid_W_{}.png".format(w_idx)), bbox_inches='tight', pad_inches=0.01) 


    # H, W, D, _ = disp.shape  
    # x = np.arange(0, W)
    # y = np.arange(0, H)
    # X, Y = np.meshgrid(x, y)
    # Z1 = disp[:,:,D//2,0] + X
    # print(Z1.shape) 
    # Z1 = Z1[::-1]
    # print(Z1.shape) 
    # Z2 = disp[:,:,D//2,1] + Y

    # plt.figure()
    # plt.contour(X, Y, Z1, 50, colors='k', linewidths=0.5)
    # plt.contour(X, Y, Z2, 50, colors='k', linewidths=0.5)
    # plt.xticks(()), plt.yticks(())
    # plt.show()  