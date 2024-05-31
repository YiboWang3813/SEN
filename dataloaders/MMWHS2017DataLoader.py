import os 
import torch 
import numpy as np 
import nibabel  

from torch.utils.data import Dataset, DataLoader 
import dataloaders.RegistrationTransforms as RT 

from typing import List, Tuple, Dict 


MMWHS2017_LABEL_DICT = {
    "Myocardium": [205], 
    "Left Atrium": [420], 
    "Left Ventricle": [500], 
    "Right Atrium": [550], 
    "Right Ventricle": [600], 
    "Pulmonary Artery": [820], 
    "Ascending Aorta": [850]
} 

MMWHS2017_RGB_MAPPING_DICT = {
    "Myocardium": [252, 110, 180], 
    "Left Atrium": [51, 82, 171], 
    "Left Ventricle": [109, 95, 177], 
    "Right Atrium": [110, 190, 231], 
    "Right Ventricle": [66, 186, 115], 
    "Pulmonary Artery": [255, 46, 23], 
    "Ascending Aorta": [255, 161, 43]
} 

def sort_mmwhs2017_dirs(names: List[str]) -> List[str]: 
    indices = sorted(
        range(len(names)), 
        key=lambda index: int(names[index].split('_')[1]) 
    ) 
    sorted_names = [names[index] for index in indices] 
    return sorted_names


def remove_intro_files(sample_names: List[str]): 
    name_buffer = [] 
    for sample_name in sample_names: 
        if not sample_name.find('.') != -1: 
            name_buffer.append(sample_name) 
    return name_buffer 


def parse_indices(txt_path: str) -> List[int]: 
    res = [] 

    with open(txt_path, 'r') as f: 
        for line in f.readlines(): 
            res.append(int(line)) 

    return res  


class MMWHS2017Dataset(Dataset): 
    def __init__(self, args, is_train: bool, transforms=None) -> None:
        super().__init__() 

        _supported_domains = ['ct', 'mr']
        assert args.fixed_mode in _supported_domains and args.moving_mode in _supported_domains, \
        "fixed and moving mode should be in {}, but got {}, {}".format(
            _supported_domains, args.fixed_mode, args.moving_mode)
        assert args.fixed_mode in ('ct', 'mr') and args.moving_mode in ('ct', 'mr'), \
        "Deformed mode has not implemented {} and {}".format(args.fixed_mode, args.moving_mode)
        
        self.fixed_mode = args.fixed_mode 
        self.moving_mode = args.moving_mode 

        root_dir = args.root_dir  
        sample_names = os.listdir(root_dir) 
        sample_names = remove_intro_files(sample_names) 
        sample_names = sort_mmwhs2017_dirs(sample_names) 

        self.is_train = is_train 

        train_indices_path = os.path.join(root_dir, 'IndicesTrain.txt') 
        self.train_indices = parse_indices(train_indices_path) 
        valid_test_indices_path = os.path.join(root_dir, 'IndicesTest.txt') 
        self.valid_test_indices = parse_indices(valid_test_indices_path) 

        self.train_sample_dirs = [os.path.join(root_dir, sample_names[i]) for i in self.train_indices] 
        self.valid_test_sample_dirs = [os.path.join(root_dir, sample_names[i]) for i in self.valid_test_indices] 

        print("MMWHS2017 data split: train: {}, valid and test: {}".format(
            len(self.train_indices), len(self.valid_test_indices)
        )) 

        self.transforms = transforms 

    @staticmethod 
    def _load_image_and_label(sample_dir: str, fixed_mode: str, moving_mode: str): 
        # load fixed image and label 
        fixed_image_name = "{}_image_padded.nii.gz".format(fixed_mode) 
        fixed_image_path = os.path.join(sample_dir, fixed_image_name) 
        fixed_image = nibabel.load(fixed_image_path).dataobj 

        fixed_label_name = "{}_label_padded.nii.gz".format(fixed_mode) 
        fixed_label_path = os.path.join(sample_dir, fixed_label_name)   
        fixed_label = nibabel.load(fixed_label_path).dataobj 

        # load moving image and label  
        moving_image_name = "{}_image_deformed.nii.gz".format(moving_mode) 
        moving_image_path = os.path.join(sample_dir, moving_image_name) 
        moving_image = nibabel.load(moving_image_path).dataobj  

        moving_label_name = "{}_label_deformed.nii.gz".format(moving_mode) 
        moving_label_path = os.path.join(sample_dir, moving_label_name)   
        moving_label = nibabel.load(moving_label_path).dataobj  

        moving_image = np.array(moving_image, dtype='float32') 
        fixed_image = np.array(fixed_image, dtype='float32') 
        moving_label_bool = np.array(moving_label, dtype='bool') 
        fixed_label_bool = np.array(fixed_label, dtype='bool') 
        moving_image = moving_image * moving_label_bool
        fixed_image = fixed_image * fixed_label_bool
         
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __getitem__(self, index): 
        if self.is_train: 
            moving_image, fixed_image, moving_label, fixed_label = self._load_image_and_label(self.train_sample_dirs[index], self.fixed_mode, self.moving_mode)
        else: 
            moving_image, fixed_image, moving_label, fixed_label = self._load_image_and_label(self.valid_test_sample_dirs[index], self.fixed_mode, self.moving_mode)
        
        # transform here 
        if self.transforms != None: 
            moving_image, fixed_image, moving_label, fixed_label = self.transforms(moving_image, fixed_image, moving_label, fixed_label)
        
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __len__(self): 
        if self.is_train: 
            return len(self.train_sample_dirs) 
        else: 
            return len(self.valid_test_sample_dirs) 
        
    def __str__(self) -> str:
        return "MMWHS2017Dataset" 
