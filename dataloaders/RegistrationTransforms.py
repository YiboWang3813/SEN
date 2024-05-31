import math 
import numpy as np 
import torch 
import torch.nn.functional as F 

import skimage.transform as T 

from typing import List, Tuple, Dict 


class AdjustNumpyType(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        moving_image = np.array(moving_image, dtype='float32') 
        fixed_image = np.array(fixed_image, dtype='float32') 
        if moving_label is not None and fixed_label is not None: 
            moving_label = np.array(moving_label, dtype='float32') 
            fixed_label = np.array(fixed_label, dtype='float32') 
        return moving_image, fixed_image, moving_label, fixed_label  
    
    def __str__(self) -> str:
        return "Adjust image type to float32 label type to int32"


class RandomRotation(object): 
    def __init__(self, dims: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]): 
        self.dims = dims 

    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        for dim in self.dims: 
            rand_k = np.random.randint(0, 4) 
            moving_image = np.rot90(moving_image, k=rand_k, axes=dim) 
            fixed_image = np.rot90(fixed_image, k=rand_k, axes=dim) 
            if moving_label is not None and fixed_label is not None: 
                moving_label = np.rot90(moving_label, k=rand_k, axes=dim) 
                fixed_label = np.rot90(fixed_label, k=rand_k, axes=dim) 
        return moving_image, fixed_image, moving_label, fixed_label 

    def __str__(self) -> str: 
        return "Random Rot90 (multi axis)" 
    

class RandomFlip(object): 
    def __init__(self, dims: List[int] = [0, 1, 2]): 
        self.dims = dims 

    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        for dim in self.dims: 
            choice = np.random.choice([True, False]) 
            if choice: 
                moving_image = np.flip(moving_image, axis=dim) 
                fixed_image = np.flip(fixed_image, axis=dim) 
                if moving_label is not None and fixed_label is not None: 
                    moving_label = np.flip(moving_label, axis=dim) 
                    fixed_label = np.flip(fixed_label, axis=dim) 
        return moving_image, fixed_image, moving_label, fixed_label  
    
    def __str__(self) -> str:
        return "Random Flip (multi axis)" 
    

class Normalize(object): 
    def __init__(self, mode: str = 'mm'): 
        assert mode in ('mm', 'ms') 
        self.mode = mode 

    @staticmethod 
    def _normalize(image, mode): 
        if mode == 'mm': 
            _min = np.min(image) 
            _max = np.max(image) 
            image = (image - _min) / (_max - _min) 
        else: 
            _mean = np.mean(image) 
            _std = np.std(image) 
            image = (image - _mean) / _std 
        return image 
    
    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        moving_image = self._normalize(moving_image, self.mode) 
        fixed_image = self._normalize(fixed_image, self.mode) 
        return moving_image, fixed_image, moving_label, fixed_label  
    
    def __str__(self) -> str:
        return "Normalize (max_min || mean_std)" 


class AdjustChannels(object): 
    def __init__(self, label_dict: Dict = None): 
        self.label_dict = label_dict 

    @staticmethod 
    def _adjust_label_channels(label, label_dict):  
        label_buffer = [] 
        for region_name, region_list in label_dict.items(): 
            region_buffer = np.zeros(label.shape, dtype='float32') 
            for region_index in region_list: 
                label_mask = label == region_index 
                label_mask = np.array(label_mask, dtype='float32') 
                region_buffer += label_mask 
            label_buffer.append(region_buffer) 
        label = np.stack(label_buffer, axis=0) 
        return label 
    
    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        moving_image = np.expand_dims(moving_image, axis=0) 
        fixed_image = np.expand_dims(fixed_image, axis=0) 
        
        if moving_label is not None and fixed_label is not None: 
            moving_label = self._adjust_label_channels(moving_label, self.label_dict) 
            fixed_label = self._adjust_label_channels(fixed_label, self.label_dict)  
        return moving_image, fixed_image, moving_label, fixed_label  
    
    def __str__(self) -> str:
        return "Adjust image and label's channels" 


class HalfResize(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        moving_image = moving_image[:, ::2, ::2, ::2] 
        fixed_image = fixed_image[:, ::2, ::2, ::2] 
        if moving_label is not None and fixed_label is not None: 
            moving_label = moving_label[:, ::2, ::2, ::2] 
            fixed_label = fixed_label[:, ::2, ::2, ::2] 
        return moving_image, fixed_image, moving_label, fixed_label 

    def __str__(self) -> str:
        return "Half resize image and label" 
    

class TargetResize(object): 
    def __init__(self, target_shape: Tuple[int, int, int]): 
        self.target_shape = target_shape 

    @staticmethod
    def _resize_label(label, target_shape): 
        label_buffer = [] 
        for c_idx in range(label.shape[0]): 
            c_label = label[c_idx, ...] 
            c_label = T.resize(c_label, target_shape) 
            c_label = np.array(c_label, dtype='float32') 
            c_label = np.around(c_label) 
            label_buffer.append(c_label) 
        label = np.stack(label_buffer, axis=0)  
        return label 
    
    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        # image  
        moving_image = T.resize(moving_image[0], self.target_shape) 
        moving_image = np.expand_dims(moving_image, axis=0)  
        fixed_image = T.resize(fixed_image[0], self.target_shape) 
        fixed_image = np.expand_dims(fixed_image, axis=0) 
        # label 
        if moving_label is not None and fixed_label is not None: 
            moving_label = self._resize_label(moving_label, self.target_shape) 
            fixed_label = self._resize_label(fixed_label, self.target_shape) 
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __str__(self) -> str:
        return "Target resize image and label" 
    

class CentralCrop(object): 
    def __init__(self, target_shape): 
        self.target_shape = target_shape 

    def _central_crop_one_volume(self, x: np.ndarray) -> np.ndarray: 
        src_depth, src_height, src_width = x.shape 
        dst_depth, dst_height, dst_width = self.target_shape 

        # depth 
        if dst_depth < src_depth: 
            front_depth_diff = int((src_depth - dst_depth) / 2) 
            rear_depth_diff = src_depth - dst_depth - front_depth_diff 
            x = x[front_depth_diff:-rear_depth_diff, :, :] 
        elif dst_depth > src_depth: 
            front_depth_diff = int((dst_depth - src_depth) / 2) 
            rear_depth_diff = dst_depth - src_depth - front_depth_diff 
            x = np.concatenate([np.zeros((front_depth_diff, x.shape[1], x.shape[2]), dtype='float32'), x], axis=0) 
            x = np.concatenate([x, np.zeros((rear_depth_diff, x.shape[1], x.shape[2]), dtype='float32')], axis=0) 
        else: 
            pass 

        # height 
        if dst_height < src_height: 
            top_height_diff = int((src_height - dst_height) / 2) 
            bottom_height_diff = src_height - dst_height - top_height_diff 
            x = x[:, top_height_diff:-bottom_height_diff, :] 
        elif dst_height > src_height: 
            top_height_diff = int((dst_height - src_height) / 2) 
            bottom_height_diff = dst_height - src_height - top_height_diff 
            x = np.concatenate([np.zeros((x.shape[0], top_height_diff, x.shape[2]), dtype='float32'), x], axis=1) 
            x = np.concatenate([x, np.zeros((x.shape[0], bottom_height_diff, x.shape[2]), dtype='float32')], axis=1) 
        else: 
            pass 

        # width  
        if dst_width < src_width: 
            left_width_diff = int((src_width - dst_width) / 2) 
            right_width_diff = src_width - dst_width - left_width_diff 
            x = x[:, :, left_width_diff:-right_width_diff] 
        elif dst_width > src_width: 
            left_width_diff = int((dst_width - src_width) / 2) 
            right_width_diff = dst_width - src_width - left_width_diff 
            x = np.concatenate([np.zeros((x.shape[0], x.shape[1], left_width_diff), dtype='float32'), x], axis=2) 
            x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], right_width_diff), dtype='float32')], axis=2) 
        else: 
            pass 

        return x 
    
    def _center_crop_one_label(self, label: np.ndarray) -> np.ndarray: 
        """ label shape: [C, D, H, W] """
        label_buffer = [] 
        for c_idx in range(label.shape[0]): 
            c_label = label[c_idx, ...] 
            c_label = self._central_crop_one_volume(c_label)
            c_label = np.array(c_label, dtype='float32') 
            label_buffer.append(c_label) 
        label = np.stack(label_buffer, axis=0)  
        return label 
    
    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        """ image shape: [1, D, H, W] label shape: [C, D, H, W] """
        # image  
        moving_image = self._central_crop_one_volume(moving_image[0]) 
        moving_image = np.expand_dims(moving_image, axis=0)  
        fixed_image = self._central_crop_one_volume(fixed_image[0]) 
        fixed_image = np.expand_dims(fixed_image, axis=0) 
        # label 
        if moving_label is not None and fixed_label is not None: 
            moving_label = self._center_crop_one_label(moving_label) 
            fixed_label = self._center_crop_one_label(fixed_label) 
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __str__(self) -> str:
        return "Center Crop image and label" 
    
    
class ToTensor(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, moving_image, fixed_image, moving_label=None, fixed_label=None): 
        moving_image = torch.from_numpy(moving_image) 
        fixed_image = torch.from_numpy(fixed_image) 
        if moving_label is not None and fixed_label is not None: 
            moving_label = torch.from_numpy(moving_label) 
            fixed_label = torch.from_numpy(fixed_label) 
        return moving_image, fixed_image, moving_label, fixed_label 

    def __str__(self) -> str:
        return "ToTensor" 
    

class Compose(object): 
    def __init__(self, ops): 
        self.ops = ops 

    def __call__(self, moving_image, fixed_image, moving_label, fixed_label): 
        for op in self.ops: 
            if op != None: 
                moving_image, fixed_image, moving_label, fixed_label = op(moving_image, fixed_image, moving_label, fixed_label) 
        return moving_image, fixed_image, moving_label, fixed_label
    
    def __str__(self) -> str:
        return "Compose (multi operations)" 




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


def forward_label(label: np.ndarray, mapping_dict: Dict) -> np.ndarray: 
    """ Adjust standard segmentation label to computing label. 
    Here we use around to deal with some interpolated values. 
    
    Paramters: 
        label (Array): segmentation label [D, H, W] int16 
        mapping_dict (Dict): region name as key and raw indices list as value  
        
    Returns: 
        new_label (Array): segmentation label for computation [C, D, H, W] float32 """
    
    D, H, W = label.shape 
    region_names = list(mapping_dict.keys()) 
    label_buffer = [] 
    for i in range(len(region_names)): 
        mask = label == i + 1 
        mask = np.array(mask, dtype='float32') 
        label_buffer.append(mask) 
    new_label = np.stack(label_buffer, axis=0) # [C, D, H, W] float32 

    return new_label   

 
def adjust_label(label: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's label to standard segmentation label. 
    Here we use around to deal with some interpolated values. 
    
    Paramters: 
        label (Array): segmentation label during computation [C, D, H, W] float32 
        
    Returns: 
        results (Array): segmentation label for visualizing [D, H, W] int16 """

    C, D, H, W = label.shape 
    results = np.zeros((D, H, W), dtype='int16') 
    for c in range(C): 
        label_c = label[c, ...] 
        label_c = np.around(label_c) 
        mask = np.array(label_c, dtype='bool') 

        old_label_c = (results * ~mask).astype(np.int16) 
        new_label_c = (mask * (c + 1)).astype(np.int16)
    
        results = old_label_c + new_label_c 

    return results 


def forward_image(image: np.ndarray) -> np.ndarray: 
    image = np.array(image, dtype='float32')  
    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    return image 


def adjust_image(image: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's image to standard gray-level image. 

    Parameters: 
        image (Array): normalized image from dataloader [D, H, W] float32 0~1 

    Returns: 
        image (Array): gray level image to store [D, H, W] uint8 0~255 """

    # normalize first 
    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    image = image * 255 
    image = np.array(image, dtype='uint8') 

    return image 
    
