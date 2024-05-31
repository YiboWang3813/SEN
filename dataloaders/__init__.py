import os 
import argparse

from dataloaders.LiverDataLoader import LiverDataset, LIVER_LABEL_DICT
from dataloaders.LPBA40DataLoader import LPBA40DataSet, LPBA40_LABEL_DICT, LPBA40_RGB_MAPPING_DICT 
from dataloaders.OASISDataLoader import OASISDataset, OASIS_LABEL_DICT, OASIS_RGB_MAPPING_DICT 
from dataloaders.IXIDataLoader import IXIDataset, IXI_LABEL_DICT, IXI_RGB_MAPPING_DICT 
from dataloaders.MindBoggle101DataLoader import MindBoggle101Dataset, MINDBOGGLE101_LABEL_DICT 
from dataloaders.MMWHS2017DataLoader import MMWHS2017Dataset, MMWHS2017_LABEL_DICT, MMWHS2017_RGB_MAPPING_DICT 
from dataloaders.SLIEVERDataLoader import SLIVERDataset, SLIVER_LABEL_DICT, SLIVER_RGB_MAPPING_DICT
from dataloaders.LSPIGDataLoader import LSPIGDataset, LSPIG_LABEL_DICT, LSPIG_RGB_MAPPING_DICT 
from dataloaders.OASISFullDataLoader import OASISNewDataset, OASIS_FULL_LABEL_DICT, OASIS_FULL_RGB_MAPPING_DICT 
from dataloaders.BraTS2020DataLoader import BraTS2020Dataset, BRATS2020_LABEL_DICT, BRATS2020_RGB_MAPPING_DICT

import dataloaders.RegistrationTransforms as RT 

from torch.utils.data import DataLoader 


# volume's shape in different dataset 
DATA_SHAPE_MAPPING = {
    'lpba40': (160, 192, 160), 
    'oasis':  (160, 192, 160), 
    'oasis_full': (160, 192, 224), 
    'ixi': (160, 192, 160), 
    'mindboggle101': (192, 192, 192), 
    'mmwhs2017': (96, 96, 96), 
    'liver': (128, 128, 128), 
    'sliver': (128, 128, 128), 
    'lspig': (128, 128, 128), 
    'brats2020': (240, 240, 155)
}

# map different label dict 
LABEL_DICT_MAPPING = {
    'lpba40': LPBA40_LABEL_DICT, 
    'oasis': OASIS_LABEL_DICT, 
    'oasis_full': OASIS_FULL_LABEL_DICT, 
    'ixi': IXI_LABEL_DICT, 
    'mindboggle101': MINDBOGGLE101_LABEL_DICT, 
    'mmwhs2017': MMWHS2017_LABEL_DICT, 
    'lspig': LSPIG_LABEL_DICT, 
    'sliver': SLIVER_LABEL_DICT, 
    'liver': LIVER_LABEL_DICT,
    'brats2020': BRATS2020_LABEL_DICT 
} 

# dataset sub dir name to find data 
DIR_NAME_MAPPING = {
    'lpba40': 'LPBA40_Brain_MRI_T1', 
    'oasis': 'OASIS1_Brain_MRT1', 
    'oasis_full': 'OASIS1_Brain_MRT1_Full',
    'ixi': 'IXI_Brain_MR1',
    'mindboggle101': 'MindBoggle101_Brain_MR1',
    'mmwhs2017': 'MMWHS2017_Heart_CT_MR1', 
    'sliver': 'SLIVER_Liver_CT',
    'lspig': 'LSPIG_Liver_CT',
    'brats2020': 'BraTS2020_Brain_MRI_MM'
}

# instantize dataset 
DATASET_MAPPING = {
    'lpba40': LPBA40DataSet, 
    'oasis': OASISDataset, 
    'oasis_full': OASISNewDataset, 
    'ixi': IXIDataset, 
    'mindboggle101': MindBoggle101Dataset, 
    'mmwhs2017': MMWHS2017Dataset, 
    'liver': LiverDataset,
    'sliver': SLIVERDataset, 
    'lspig': LSPIGDataset,
    'brats2020': BraTS2020Dataset 
} 

# rgb mapping dict 
RGB_MAPPING_DICT = {
    'lpba40': LPBA40_RGB_MAPPING_DICT, 
    'oasis': OASIS_RGB_MAPPING_DICT, 
    'oasis_full': OASIS_FULL_RGB_MAPPING_DICT, 
    'ixi': IXI_RGB_MAPPING_DICT, 
    'mmwhs2017': MMWHS2017_RGB_MAPPING_DICT, 
    'sliver': SLIVER_RGB_MAPPING_DICT, 
    'lspig': LSPIG_RGB_MAPPING_DICT,
    'brats2020': BRATS2020_RGB_MAPPING_DICT
} 


def add_dataloader_related_args(parser: argparse.ArgumentParser): 
    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples in a batch') 
    parser.add_argument('--dataroot', type=str, default='../', help='root dir where stores data') 
    parser.add_argument('--root_dir', type=str, default="", help='specified dir to store data')
    parser.add_argument('--which_set', type=str, default='lpba40', 
                        help='used dataset: see __init__.py in dataloaders package for more details')
    # these two arguments are used to control if you need to resize data to avoid memory overflow 
    parser.add_argument('--is_resize', action='store_true') 
    parser.add_argument('--is_crop', action='store_true')
    parser.add_argument('--target_shape', type=str, default='128,128,128', 
                        help='target shape used in RT TargetResize') 
    # these two arguments are called in LPBA40DataLoader and OASISDataLoader 
    parser.add_argument('--fixed_idx', type=int, default=0, help='which sample will be choosed to be fixed')
    parser.add_argument('--is_atlas', action='store_true')  
    parser.add_argument('--split_factor', type=float, default=0.2, 
                        help='how many samples will be splited to be validation or test') 
    # these three arguments are called in LiverDataLoader 
    parser.add_argument('--which_train', type=str, default='MSD', 
                        help='which liver dataset will be used to train MSD || BFH') 
    parser.add_argument('--which_valid_test', type=str, default='SLIVER', 
                        help='which liver dataset will be used to valid and test SLIVER || LSPIG') 
    # this argument will be called in LSPIGDataLoader 
    parser.add_argument('--pre_gas', action='store_true')  
    # these two arguments are called in MMWHS2017DataLoader and ProstateMultiModalityDataLoader 
    parser.add_argument('--fixed_mode', type=str, default='ct', help='which mode as fixed') 
    parser.add_argument('--moving_mode', type=str, default='mr', help='which mode as moving use') 
    return parser 


def get_transforms(args): 
    """ Return an unified data transforms for all datasets. """ 
    channels_trans = None 
    if args.which_set in ('lpba40', 'oasis', 'oasis_full', 
                          'ixi', 'mindboggle101', 'mmwhs2017',
                          'lspig', 'sliver', 'brats2020'): 
        channels_trans = RT.AdjustChannels(LABEL_DICT_MAPPING[args.which_set]) 
    else: # liver, here we use args.which_valid_test to set a sub dir (LSPIG or SLIVER) 
        channels_trans = RT.AdjustChannels(LABEL_DICT_MAPPING[args.which_valid_test.lower()]) 

    resize_trans = None 
    if args.is_resize: 
        target_shape = tuple([int(x) for x in args.target_shape.split(',')]) 
        resize_trans = RT.TargetResize(target_shape) 
        
    center_crop_trans = None 
    if args.is_crop: 
        target_shape = tuple([int(x) for x in args.target_shape.split(',')]) 
        center_crop_trans = RT.CentralCrop(target_shape)  

    train_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        # RT.RandomRotation(), 
        RT.RandomFlip(), 
        RT.Normalize(), 
        channels_trans, 
        resize_trans, 
        center_crop_trans,
        RT.ToTensor() 
    ]) 

    valid_test_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.Normalize(), 
        channels_trans, 
        resize_trans, 
        center_crop_trans,
        RT.ToTensor() 
    ]) 

    return train_transforms, valid_test_transforms 

   
# directly use this function to get your dataloader 
def get_dataloader(args): 
    train_trans, valid_test_trans = get_transforms(args) 

    if args.which_set in ('lpba40', 'oasis', 'oasis_full', 
                          'ixi', 'mindboggle101', 'mmwhs2017', 
                          'lspig', 'sliver', 'brats2020'):  
        args.root_dir = os.path.join(args.dataroot, DIR_NAME_MAPPING[args.which_set]) 

    train_dataset = DATASET_MAPPING[args.which_set](args, True, train_trans) 
    valid_test_dataset = DATASET_MAPPING[args.which_set](args, False, valid_test_trans) 

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2) 
    # valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False, num_workers=2) 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False) 

    return train_dataloader, valid_test_dataloader


# here we list some demo commands to rightly use use these datasets 
""" 
# for lpba40 (atlas or non-atlas) 
python main.py --which_set 'lpba40' --is_atlas 
python main.py --which_set 'lpba40' 

# for oasis (atlas or non-atlas)
python main.py --which_set 'oasis' --is_atlas 
python main.py --which_set 'oasis' 

# for oasis full (atlas or non-atlas)
python main.py --which_set 'oasis_full' --is_atlas 
python main.py --which_set 'oasis_full' 

# for mindboggle101 (non-atlas only) 
python main.py --which_set 'mindboggle101'  

# for ixi (atlas or non-atlas) 
python main.py --which_set 'ixi' --is_atlas 
python main.py --which_set 'ixi' 

# for sliver (non-atlas only) 
python main.py --which_set 'sliver' 

# for lspig (non-atlas only) 
python main.py --which_set 'lspig' 

# for mmwhs2017 (multi-modality)
python main.py --which_set 'mmwhs2017' --fixed_mode 'ct' --moving_mode 'mr' 
python main.py --which_set 'mmwhs2017' --fixed_mode 'mr' --moving_mode 'ct' 

# for brats2020 (multi-modality) 
python main.py --which_set 'brats2020' --is_resize --target_shape '224,224,128' --fixed_mode 't1' --moving_mode 't2'
python main.py --which_set 'brats2020' --is_resize --target_shape '224,224,128' --fixed_mode 't2' --moving_mode 't1'
"""
