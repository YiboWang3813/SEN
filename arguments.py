import os 
import argparse 


def add_dataloader_related_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
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


def add_helper_related_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    # add train related arguments 
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for weights')
    parser.add_argument('--gpus', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') 
    # add save arguments 
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--exp_name', type=str, default="", help="experiment dir's name")
    parser.add_argument('--output_dir', type=str, default='') 
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints') 
    parser.add_argument('--resume', type=str, default='') 
    parser.add_argument('--eval', action='store_true', default=False)  
    # add the number of training iterations 
    parser.add_argument('--iter_cnt', type=int, default=0) 
    parser.add_argument('--max_iters', type=int, default=800, help='maximum number of iterations to train') 
    parser.add_argument('--max_epoch', type=int, default=1, help='maximum number of epochs to train') 
    parser.add_argument('--save_freq', type=int, default=200, help='validate the model per save_freq iterations') 
    parser.add_argument('--max_samples', type=int, default=50, help='maximum number of samples to be validated and evaluated')
    return parser 


def add_running_model_related_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='regulization weight 0.05 | 0.1 | 0.2')  
    parser.add_argument('--lambda_pyd', type=float, default=0.001, help='pyramid weight') 
    parser.add_argument('--lambda_cas', type=float, default=0.001, help='cascade weight') 
    parser.add_argument('--init_levels', type=str, default="3", help="init levels to build running model") 
    parser.add_argument('--is_diff', action='store_true') 
    parser.add_argument('--is_semi_sup', action='store_true') 
    parser.add_argument('--is_distill_offline', action='store_true')   
    parser.add_argument('--is_pyd_distill', action='store_true') 
    parser.add_argument('--is_cas_distill', action='store_true')    
    return parser  


def get_default_args_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    parser = add_dataloader_related_arguments(parser) 
    parser = add_helper_related_arguments(parser) 
    parser = add_running_model_related_arguments(parser) 
    return parser 


_which_set_exp_name_mapping_dict = {
    "lpba40": "LPBA40", 
    "oasis": "OASIS", 
    "oasis_full": "OASIS", 
    "mindboggle101": "MindBoggle101", 
    "ixi": "IXI", 
    "sliver": "SLIVER", 
    "lspig": "LSPIG", 
    "mmwhs2017": "MMWHS2017" 
} 


def set_dataloader_related_exp_name(args, exp_name: str) -> str:  
    # dataset name 
    global _which_set_exp_name_mapping_dict 
    exp_name += _which_set_exp_name_mapping_dict[args.which_set] 

    # atlas 
    exp_name = "{}_A".format(exp_name) if args.is_atlas else "{}_NA".format(exp_name) 
    
    # multi modality 
    if args.which_set in ["mmwhs2017"]: 
        # add fixed image's modality 
        exp_name = "{}_{}".format(exp_name, args.fixed_mode.upper()) 
        # add moving image's modality 
        exp_name = "{}_{}".format(exp_name, args.moving_mode.upper()) 

    return exp_name 
 
