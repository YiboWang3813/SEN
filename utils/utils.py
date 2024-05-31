import os 
import time 
import subprocess 
import datetime 
import pickle 
import copy 
import json 
import logging 
import shutil 
import nibabel 
from collections import defaultdict, deque 
import matplotlib.pyplot as plt 
import numpy as np 

import torch 
import torch.distributed as dist 
import torch.nn.init as init 
import torch.nn.functional as F 
import torch.nn as nn 
from medpy import metric 

from typing import Tuple, List, Dict 


def makedir(path):
    if not os.path.exists(path): 
        os.makedirs(path) 


def sort_sample_names(names: List[str]) -> List[str]: 
    indices = sorted(
        range(len(names)), 
        key=lambda index:int(names[index].split('_')[-1]) 
    ) 
    return [names[index] for index in indices] 


def append_metric_dict(src_dict: Dict, dst_dict: Dict): 
    for k, v in src_dict.items(): 
        if k not in dst_dict.keys(): 
            dst_dict.update({k: [v]}) 
        else: 
            dst_dict[k].append(v) 


def cal_mean_std_loop_columns(arr): 
    # type: (np.ndarray) -> Tuple[float, float] 
    mean, std = 0, 0 
    for i in range(arr.shape[1]): # shape[0] -> height shape[1] -> width 
        col = arr[:, i] 
        col_mean, col_std = np.mean(col), np.std(col) 
        mean += col_mean 
        std += col_std 
    mean /= arr.shape[1] 
    std /= arr.shape[1] 
    return mean, std   


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state.state_dict(), filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best_m.pth.tar')
        shutil.copyfile(filename, best_filename)


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


def adjust_image(image: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's image to standard gray-level image. 

    Parameters: 
        image (Array): image from dataloader [D, H, W] float32 

    Returns: 
        image (Array): gray level image to store [D, H, W] uint8 0~255 """

    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    image = image * 255 
    image = np.array(image, dtype='uint8') 

    return image 


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


def cal_dice_torch(fixed_label, warped_label, label_dict): 
    # type: (torch.Tensor, torch.Tensor, Dict) -> Dict 
    """ Get dice dict. 
    
    Paramters: 
        fixed_label (Tensor): ground-truth segmentation label [B, C, D, H, W] (int16) 
        warped_label (Tensor): predicted segmentation label [B, C, D, H, W] (int16) 
        label_dict (Dict): each label corresponds to one anatomical area  
    B should be 1 and C should be the number of anatomical areas 
        
    Returns: 
        dices (Dict): the dice results """
    
    warped_label = warped_label.squeeze(dim=0) # remove the batch size 
    fixed_label = fixed_label.squeeze(dim=0) 
    assert warped_label.dim() == 4 and fixed_label.dim() == 4, \
        "The shape is not suitable to calculate dice, got {}".format(warped_label.shape) 
    dices = {}
    for idx, k in enumerate(label_dict.keys()):
        top = (warped_label[idx] * fixed_label[idx]).sum()
        bottom = warped_label[idx].sum() + fixed_label[idx].sum()
        dice = ((2 * top) + 0.001) / (bottom + 0.001)
        dices.update({k: dice.item()})
    return dices 


def cal_jacobian_determinant_torch(deformation_field): 
    # type: (torch.Tensor) -> Tuple[torch.Tensor, int, float]  
    """ calculate jacobian determinant using pytorch. 
    
    Parameters: 
        deformation_field (Tensor): [B, 3, D, H, W]
    
    Returns: 
        D (Tensor): jacobian determinent [B, D-1, H-1, W-1] 
        num_folds (int): the number of folds 
        folds_ratio (float): the ratio of folds """
    
    deformation_field = deformation_field.permute(0,2,3,4,1)

    D_y = (deformation_field[:,1:,:-1,:-1,:] - deformation_field[:,:-1,:-1,:-1,:])
    D_x = (deformation_field[:,:-1,1:,:-1,:] - deformation_field[:,:-1,:-1,:-1,:])
    D_z = (deformation_field[:,:-1,:-1,1:,:] - deformation_field[:,:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = D1 - D2 + D3 
    batch_size, depth, height, width = D.shape 

    num_folds = torch.sum(D < 0) 
    folds_ratio = num_folds / (batch_size * depth * height * width)   
    
    return D, num_folds, folds_ratio 


def cal_dice_numpy(y_pred, y_true, label_dict): 
    # type: (np.ndarray, np.ndarray, Dict) -> Dict 
    """ Compute dice dict. 
    
    Parameters: 
        y_pred (Array): predicted segmentation label [D, H, W] int16 
        y_true (Array): ground truth segmentation label [D, H, W] int16 
        label_dict (Dict): label dict in data loader 
    
    Returns: 
        results (Dict): region name - dice value dict """

    results = {} 
    for idx, (k, v) in enumerate(label_dict.items()): 
        top = np.sum(np.logical_and(y_pred == idx, y_true == idx)) * 2 
        bottom = np.sum(y_pred == idx) + np.sum(y_true == idx) 
        bottom = np.maximum(bottom, 1e-8) 
        dice_value = top / bottom 
        results.update({k: dice_value}) 
    
    return results 


def cal_jacobian_determinant_numpy(displacement): 
    # type: (np.ndarray) -> Tuple[np.ndarray, int, float]
    """ Compute jacobian determinant. 
    
    Parameters: 
        displacement (Array): [D, H, W, 3] 
    
    Returns: 
        D (Array): Jacobian determinant [D-1, H-1, W-1] 
        num_folds (int): the number of folds 
        folds_ratio (float): the ratio of folds """ 

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])
 
    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = D1 - D2 + D3 # shape: [D-1, H-1, W-1] 
    dd, hh, ww= D.shape 

    num_folds = np.sum(D < 0) 
    folds_ratio = num_folds / (dd * hh * ww)   

    return D, num_folds, folds_ratio 


def cal_hausdorff95_numpy(fixed_label, warped_label, label_dict): 
    # type: (np.ndarray, np.ndarray, Dict) -> Dict 
    """ Calculate haudorsff95 using medpy and numpy. 

    Parameters: 
        fixed_label (Array): ground-truth segmentation label [D, H, W] (int16) 
        warped_label (Array): predicted segmentation label [D, H, W] (int16) 
        label_dict (Dict): each label corresponds to one anatomical area  
    The value range of fixed_label (warped_label) = (0, the number of labels). 

    Returns: 
        results (Dict): a dict storing things like {region_name: hd95_value} """

    results = {} 
    for idx, (label_name, label_indices) in enumerate(label_dict.items()): 
        fixed_one_mask = np.array(fixed_label == idx, dtype='bool')  
        warped_one_mask = np.array(warped_label == idx, dtype='bool') 
        value = metric.binary.hd95(warped_one_mask, fixed_one_mask) 
        results.update({label_name: value}) 
    return results  


def generate_sample_grid(x: torch.Tensor) -> torch.Tensor: 
    """ Generate the sample grid. """
    B, _, D, H, W = x.shape 
    # generate the sample grid 
    xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1).view(1, D, H, W) 
    yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W).view(1, D, H, W) 
    zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W).view(1, D, H, W) 
    sample_grid = torch.cat([xx, yy, zz], dim=0).repeat(B, 1, 1, 1, 1).float().to(x.device) # [B, 3, D, H, W] 
    return sample_grid 


def convert_diff(velocity: torch.Tensor, time_steps: int = 7) -> torch.Tensor: 
    """ Convert the velocity field to the displacement field in a diffeomorphic manner. 
    
    Parameters: 
        velocity (Tensor): the velocity filed predicted by the model [B, 3, D, H, W] 
        time_steps (int): the number of iterations to update the velocity field 
        
    Returns: 
        flow (Tensor): the diffeomorphic displacement field [B, 3, D, H, W] """
    # generate the sample grid 
    sample_grid = generate_sample_grid(velocity) # [B, 3, D, H, W] 
    # scale the velocity field 
    flow = velocity / (2.0 ** time_steps) 
    _, _, D, H, W = flow.shape
    # loop the number of time steps to update the velocity field 
    for _ in range(time_steps): 
        vgrid = sample_grid + flow # add velocity field to the sample grid 
        # scale the value in velocity grid to [-1, 1] 
        vgrid[:, 0] = (vgrid[:, 0] - (W - 1) / 2) / (W - 1) * 2   
        vgrid[:, 1] = (vgrid[:, 1] - (H - 1) / 2) / (H - 1) * 2 
        vgrid[:, 2] = (vgrid[:, 2] - (D - 1) / 2) / (D - 1) * 2 
        # update the velocity field this iteration 
        vgrid = vgrid.permute(0, 2, 3, 4, 1) 
        flow = flow + F.grid_sample(flow, vgrid, mode='bilinear', align_corners=True)   
    return flow 


def pad_nii(image: nibabel.Nifti1Image, dst_shape: Tuple[int, int, int]) -> nibabel.Nifti1Image: 
    """ Pad the source .nii image to wanted shape. 
    
    Parameters: 
        image (nibabel.Nifti1Image): source image 
        dst_shape (tuple): target shape 
    
    Returns: 
        image (nibabel.Nifti1Image): target image """ 
    data = image.dataobj 
    affine = image.affine 
    header = image.header 

    old_width, old_height, old_depth = data.shape 
    new_width, new_height, new_depth = dst_shape 

    # adjust width 
    if new_width > old_width: 
        left_half_width = (new_width - old_width) // 2 
        right_half_width = new_width - old_width - left_half_width 
        data = np.concatenate([np.zeros((left_half_width, old_height, old_depth)), data], axis=0) 
        data = np.concatenate([data, np.zeros((right_half_width, old_height, old_depth))], axis=0) 
    elif new_width == old_width: 
        pass 
    else:
        left_half_width = (old_width - new_width) // 2 
        right_half_width = old_width - new_width - left_half_width 
        data = data[left_half_width:-right_half_width, :, :] 
    old_width = new_width 

    # adjust height 
    if new_height > old_height: 
        top_half_height = (new_height - old_height) // 2 
        bottom_half_height = new_height - old_height - top_half_height 
        data = np.concatenate([np.zeros((old_width, top_half_height, old_depth)), data], axis=1) 
        data = np.concatenate([data, np.zeros((old_width, bottom_half_height, old_depth))], axis=1) 
    elif new_height == old_height: 
        pass 
    else: 
        top_half_height = (old_height - new_height) // 2 
        bottom_half_height = old_height - new_height - top_half_height 
        data = data[:, top_half_height:-bottom_half_height, :] 
    old_height = new_height 

    # adjust depth 
    if new_depth > old_depth: 
        front_half_depth = (new_depth - old_depth) // 2 
        back_half_depth = new_depth - old_depth - front_half_depth 
        data = np.concatenate([np.zeros((old_width, old_height, front_half_depth)), data], axis=2) 
        data = np.concatenate([data, np.zeros((old_width, old_height, back_half_depth))], axis=2) 
    elif new_depth == old_depth: 
        pass 
    else: 
        front_half_depth = (old_depth - new_depth) // 2 
        back_half_depth = old_depth - new_depth - front_half_depth 
        data = data[:, :, front_half_depth:-back_half_depth]  
    old_depth = new_depth 

    # rewrite the related information in header 
    header['dim'][1] = new_width 
    header['dim'][2] = new_height 
    header['dim'][3] = new_depth 

    return nibabel.Nifti1Image(data, affine, header)  


def normalize_nii(image: nibabel.Nifti1Image, scale=None) -> nibabel.Nifti1Image: 
    data = image.dataobj 
    affine = image.affine 
    header = image.header 

    data = np.array(data, dtype='float32') 
    _min, _max = np.min(data), np.max(data) 
    data = (data - _min) / (_max - _min) 
    if scale is not None: 
        data *= scale 

    return nibabel.Nifti1Image(data, affine, header)  


def set_nii_type(image: nibabel.Nifti1Image, dtype) -> nibabel.Nifti1Image: 
    data = image.dataobj 
    affine = image.affine 
    header = image.header 

    data = np.array(data, dtype=dtype) 

    return nibabel.Nifti1Image(data, affine, header) 


class SmoothedValue(object): 
    """ Track a series of values and provide access to smoothed values over a window orthe global series average. 
    This class contains a double-ended-queue (deque) to save a series of values. 

    Parameters: 
        whindow_size (int): window size to smoothe a series of values 
        fmt (str): printing format """
    def __init__(self, window_size=20, fmt=None): 
        # fmt is used for printing formatly 
        if fmt is None: 
            fmt = "{median:.4f} ({global_avg:.4f})" 
        # maxlen限制最大长度,超出部分将覆盖原有数据 
        self.deque = deque(maxlen=window_size)
        self.total = 0.0 
        self.count = 0 
        self.fmt = fmt 

    def update(self, value, n=1): 
        # update n value into deque 
        self.deque.append(value) 
        self.count += n 
        self.total += value * n 

    def synchronize_between_processes(self): 
        # synchronize processes used in multi-GPU training 
        if not is_dist_avail_and_initialized(): 
            return 
        t = torch.tensor([self.count, self.total], dtype=torch.float32, device='cuda') 
        dist.barrier() 
        dist.all_reduce(t) 
        t = t.tolist() 
        self.count, self.total = int(t[0]), t[1] 

    @property 
    def median(self): 
        # return the median value of deque 
        d = torch.tensor(list(self.deque)) 
        return d.median().item() 
    
    @property
    def avg(self): 
        # return the average value of deque 
        d = torch.tensor(list(self.deque), dtype=torch.float32) 
        return d.mean().item() 
    
    @property 
    def global_avg(self): 
        # return the global average value of deque 
        return self.total / self.count 
    
    @property 
    def max(self): 
        # return the max value of deque 
        return max(self.deque) 
    
    @property 
    def value(self): 
        # return the last value of deque 
        return self.deque[-1] 
    
    def __str__(self):
        return self.fmt.format(
            median=self.median, 
            avg=self.avg, 
            global_avg=self.global_avg, 
            max=self.max, 
            value=self.value) 


class MetricLogger(object): 
    """ Define a logger to save metrices. """
    def __init__(self, delimeter='\t'): 
        # defaultdict return a dict, the values in this dict is SmoothedValue
        self.meters = defaultdict(SmoothedValue) 
        self.delimeter = delimeter 

    def update(self, **kwargs): 
        for k, v in kwargs.items(): 
            if isinstance(v, torch.Tensor): 
                v = v.item() 
            assert isinstance(v, (float, int)) 
            self.meters[k].update(v) 

    def synchronize_between_processes(self): 
        # synchronize several processes used in distributed training 
        for meter in self.meters.values(): 
            meter.synchronize_between_processes() 

    def add_meter(self, name, meter): 
        # add a new meter to meters 
        self.meters[name] = meter 

    def log_every(self, iterable, print_freq, header=None): 
        i = 0 
        if not header: 
            header = '' 
        start_time = time.time() 
        end = time.time() 
        iter_time = SmoothedValue(fmt='{avg:.4f}') 
        data_time = SmoothedValue(fmt='{avg:.4f}') 
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd' 
        if torch.cuda.is_available(): 
            log_msg = self.delimeter.join([
                header, 
                '[{0' + space_fmt + '}/{1}]', 
                'eta: {eta}', 
                '{meters}', 
                'time: {time}', 
                'data: {data}', 
                'max mem: {memory:.0f}' 
            ])
        else: 
            log_msg = self.delimeter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0 
        for obj in iterable: 
            data_time.update(time.time() - end) 
            yield obj 
            iter_time.update(time.time() - end) 
            if i % print_freq == 0 or i == len(iterable) - 1: 
                # datetime.timedelta returns time interval between two datetime objects 
                eta_seconds = iter_time.global_avg * (len(iterable) - 1) 
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds))) 
                if torch.cuda.is_available(): 
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, 
                        meters=str(self), 
                        time=str(iter_time), data=str(data_time), 
                        memory=torch.cuda.max_memory_allocated() / MB)) 
                else: 
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, 
                        meters=str(self), 
                        time=str(iter_time), data=str(data_time))) 
                i += 1 
                end = time.time() 
        total_time = time.time() - start_time 
        total_time_str = str(datetime.timedelta(seconds=int(total_time))) 
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def __str__(self):
        # return logged losses 
        loss_str = [] 
        for name, meter in self.meters.items(): 
            loss_str.append(
                '{}: {}'.format(name, str(meter))) 
        # .join will extract elements from loss_str and insert delimeter between every two elements 
        return self.delimeter.join(loss_str) 
    
    def __getattr__(self, attr): 
        # return specific attribute 
        if attr in self.meters: 
            return self.meters[attr] 
        if attr in self.__dict__: 
            return self.__dict__[attr] 
        raise AttributeError(f"{type(self).__name__} object has no attribute {attr}") 
    

""" The functions for distributed models and processes start here. """ 

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def init_distributed_mode(args): 
    """ Initialize the distributed training mode. """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) # RANK 代表某一块GPU 
        args.world_size = int(os.environ['WORLD_SIZE']) # WORLD_SIZE 代表有几块GPU 
        args.gpu = int(os.environ['LOCAL_RANK']) # LOCAL_RANK 代表某一块GPU的编号 
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0) 


def setup_for_distributed(is_master):
    """ This function disables printing when not in master process """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def save_on_master(*args, **kwargs): 
    # save the resultes only on the main process, which means save only occur in the GPU0 
    if is_main_process(): 
        torch.save(*args, **kwargs) 


def is_dist_avail_and_initialized(): 
    # check the distributed training is available and initialized 
    if not dist.is_available(): 
        return False 
    if not dist.is_initialized(): 
        return False 
    return True 


def get_world_size(): 
    # get the world size which is equal to the number of GPUs 
    if not is_dist_avail_and_initialized(): 
        return 1  
    return dist.get_world_size() 


def get_rank(): 
    # get the running process's rank  
    if not is_dist_avail_and_initialized(): 
        return 0 
    return dist.get_rank() 


def is_main_process(): 
    # check the running process is the main process or not 
    return get_rank() == 0 


def reduce_dict(input_dict, average=True): 
    # type: (Dict, bool) -> Dict 
    """ Reduce the values in the dictionary from all processes. 
    So that all processes have the averaged results. 
    
    Parameters: 
        input_dict (Dict): input dict needed to be reduced 
        average (bool): whether to average results using world size 
    
    Returns: 
        reduced_dict (Dict): reduced dict """
    world_size = get_world_size() 
    if world_size < 2: 
        return input_dict 
    with torch.no_grad(): 
        names = [] 
        values = [] 
        # sort the keys so that they're consistent across processes 
        for k in sorted(input_dict.keys()): 
            names.append(k) 
            values.append(input_dict[k]) 
        values = torch.stack(values, dim=0) 
        dist.all_reduce(values) 
        if average: 
            values /= world_size 
        reduced_dict = {k:v for k, v in zip(names, values)} 
    return reduced_dict 
