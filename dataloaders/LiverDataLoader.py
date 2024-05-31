import os 
import torch 
import numpy as np 
import nibabel  

from torch.utils.data import Dataset, DataLoader 
import dataloaders.RegistrationTransforms as RT 

from typing import List, Tuple, Dict 


SLIVER_LABEL_DICT = {
    'Liver': [127]
} 

LSPIG_LABEL_DICT = {
    'Liver': [127] 
} 

LIVER_LABEL_DICT = {
    'Liver': [127] 
}

def sort_liver_dirs(names: List[str]) -> List[str]: 
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


class LiverDataset(Dataset): 
    """ In this dataset, we use unlabeled data for training and use labeled data for validating and testing. """
    def __init__(self, args, is_train: bool, transforms=None) -> None:
        super().__init__() 

        root_dir = args.dataroot 
        msd_dir = os.path.join(root_dir, 'MSD_Liver_CT') 
        bfh_dir = os.path.join(root_dir, 'BFH_Liver_CT') 
        assert os.path.exists(msd_dir) and os.path.exists(bfh_dir), "Lack MSD or BFH dir under {}".format(root_dir) 
        sliver_dir = os.path.join(root_dir, 'SLIVER_Liver_CT') 
        lspig_dir = os.path.join(root_dir, 'LSPIG_Liver_CT')   
        assert os.path.exists(sliver_dir) and os.path.exists(lspig_dir), "Lack SLIVER or LSPIG dir under {}".format(root_dir) 

        # set train, (valid and test) dir 
        if args.which_train == 'MSD': 
            train_dir = msd_dir 
        elif args.which_train == 'BFH': 
            train_dir = bfh_dir 
        else: 
            raise NotImplementedError("You must use MSD or BFH to train within liver dataset" ) 
        
        if args.which_valid_test == 'SLIVER': 
            valid_test_dir = sliver_dir 
        elif args.which_valid_test == 'LSPIG': 
            valid_test_dir = lspig_dir 
        else: 
            raise NotImplementedError("You must use SLIVER or LSPIG to valid and test within liver dataset" )

        # NOTE here we train and (valid, test) in the same subject-to-subject (non-atlas based) mode 
        # NOTE here we set i-th sample as fixed and j-th sample as moving 

        # set train pair paths list 
        sample_pairs_train = [] 
        sample_names_train = os.listdir(train_dir) 
        sample_names_train = remove_intro_files(sample_names_train) 
        sample_names_train = sort_liver_dirs(sample_names_train) 
        num_samples_train = len(sample_names_train) 
        for i in range(num_samples_train): 
            for j in range(num_samples_train): 
                if i != j: 
                    sample_dir_i = os.path.join(train_dir, sample_names_train[i]) 
                    sample_dir_j = os.path.join(train_dir, sample_names_train[j]) 
                    pairs = (sample_dir_i, sample_dir_j) 
                    sample_pairs_train.append(pairs) 
        self.sample_pairs_train = sample_pairs_train 

        # set valid, test pair paths list 
        sample_pairs_valid_test = [] 
        sample_names_valid_test = os.listdir(valid_test_dir) 
        sample_names_valid_test = remove_intro_files(sample_names_valid_test) 
        if args.which_valid_test == 'LSPIG': 
            # add an interval to deal with lspig 
            sample_names_pregas = [] 
            sample_names_aftergas = [] 
            for name in sample_names_valid_test: 
                if name.find('AfterGas') != -1: 
                    sample_names_aftergas.append(name) 
                elif name.find('PreGas') != -1: 
                    sample_names_pregas.append(name) 
                else: 
                    raise ValueError("Unexpected dir name in LSPIG dataset") 
            sample_names_pregas = sort_liver_dirs(sample_names_pregas) 
            sample_names_aftergas = sort_liver_dirs(sample_names_aftergas) 
            if args.pre_gas: 
                sample_names_valid_test = sample_names_pregas 
            else: 
                sample_names_valid_test = sample_names_aftergas 
        sample_names_valid_test = sort_liver_dirs(sample_names_valid_test) 
        num_samples_valid_test = len(sample_names_valid_test) 
        for i in range(num_samples_valid_test): 
            for j in range(num_samples_valid_test): 
                if i != j: 
                    sample_dir_i = os.path.join(valid_test_dir, sample_names_valid_test[i]) 
                    sample_dir_j = os.path.join(valid_test_dir, sample_names_valid_test[j]) 
                    pairs = (sample_dir_i, sample_dir_j) 
                    sample_pairs_valid_test.append(pairs) 
        self.sample_pairs_valid_test = sample_pairs_valid_test 
        print("Liver data split: train: {}, valid and test: {}".format(len(self.sample_pairs_train), len(self.sample_pairs_valid_test)))

        self.is_train = is_train 
        self.transforms = transforms 

    @staticmethod 
    def _load_image_and_label(sample_pair: Tuple[str, str]): 
        fixed_dir, moving_dir = sample_pair[0], sample_pair[1] 
        if not os.path.exists(os.path.join(fixed_dir, 'segmentation.nii')) \
        and not os.path.exists(os.path.join(moving_dir, 'segmentation.nii')): 
            # w/o label only return image back 
            fixed_image_path = os.path.join(fixed_dir, 'volume.nii') 
            fixed_image = nibabel.load(fixed_image_path).dataobj 
            moving_image_path = os.path.join(moving_dir, 'volume.nii') 
            moving_image = nibabel.load(moving_image_path).dataobj 
            return moving_image, fixed_image 
        else: 
            # w/ label also return label back 
            fixed_image_path = os.path.join(fixed_dir, 'volume.nii') 
            fixed_image = nibabel.load(fixed_image_path).dataobj 
            moving_image_path = os.path.join(moving_dir, 'volume.nii') 
            moving_image = nibabel.load(moving_image_path).dataobj 

            fixed_label_path = os.path.join(fixed_dir, 'segmentation.nii') 
            fixed_label= nibabel.load(fixed_label_path).dataobj 
            moving_label_path = os.path.join(moving_dir, 'segmentation.nii') 
            moving_label = nibabel.load(moving_label_path).dataobj 
            return moving_image, fixed_image, moving_label, fixed_label

    def __getitem__(self, index): 
        moving_label, fixed_label = None, None 

        if self.is_train: 
            train_sample_pair = self.sample_pairs_train[index] 
            moving_image, fixed_image = self._load_image_and_label(train_sample_pair) 
        else: 
            valid_test_sample_dir = self.sample_pairs_valid_test[index] 
            moving_image, fixed_image, moving_label, fixed_label = self._load_image_and_label(valid_test_sample_dir) 
        
        if self.transforms != None: 
            moving_image, fixed_image, moving_label, fixed_label = self.transforms(moving_image, fixed_image, moving_label, fixed_label)

        if self.is_train: 
            return moving_image, fixed_image 
        else: 
            return moving_image, fixed_image, moving_label, fixed_label 

    def __len__(self): 
        if self.is_train: 
            return len(self.sample_pairs_train) 
        else: 
            return len(self.sample_pairs_valid_test) 


# if __name__ == "__main__": 
#     import argparse 
    
#     parser = argparse.ArgumentParser() 
#     parser.add_argument('--dataroot', type=str, default='./Dataset', help='root dir where stores data') 
#     parser.add_argument('--which_train', type=str, default='MSD', help='which liver dataset will be used to train MSD || BFH') 
#     parser.add_argument('--which_valid_test', type=str, default='SLIVER', help='which liver dataset will be used to valid and test SLIVER || LSPIG') 
#     parser.add_argument('--batch_size', type=int, default=1, help='batch size') 
#     parser.add_argument('--pre_gas', action='store_true')  

#     args = parser.parse_args() 

#     # set your transforms here 
#     train_transforms = RT.Compose([
#         RT.AdjustNumpyType(),
#         RT.RandomRotation(), 
#         RT.RandomFlip(), 
#         RT.Normalize(), 
#         RT.AdjustChannels(None), 
#         RT.ToTensor() 
#     ]) 
#     valid_test_transforms = RT.Compose([
#         RT.AdjustNumpyType(),
#         RT.Normalize(), 
#         RT.AdjustChannels(SLIVER_LABEL_DICT), # or RT.AdjustChannels(SLIVER_LABEL_DICT) 
#         RT.ToTensor() 
#     ]) 

#     train_dataset = LiverDataset(args, True, train_transforms) 
#     valid_test_dataset = LiverDataset(args, False, valid_test_transforms) 

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) 
#     valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4) 
