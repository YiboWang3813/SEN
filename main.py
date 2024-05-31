import os 
import torch
import numpy as np
import argparse

from dataloaders import get_dataloader, LABEL_DICT_MAPPING 
from model import PydCasUnetRunningModel 
from utils import utils 
from engine import train_one_model, evaluate  
from arguments import get_default_args_parser, set_dataloader_related_exp_name

from setproctitle import setproctitle 


def set_auxiliary_attributes(args): 
    device = torch.device(args.device) 
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    torch.backends.cudnn.benchmark = True 
    # set dataset and dataloader  
    train_loader, test_loader = get_dataloader(args) 
    return device, train_loader, test_loader 


def main(args): 
    device, train_loader, test_loader = set_auxiliary_attributes(args) 
    exp_name_data_preffix = args.exp_name  

    # set output dir and info logger 
    exp_name = "{}_L5_L5_L5_Cas".format(exp_name_data_preffix) 
    exp_name = "{}_Diff".format(exp_name) if args.is_diff else exp_name 
    exp_name = "{}_SemiSup".format(exp_name) if args.is_semi_sup else exp_name 
    exp_name = "{}_Reg_{}".format(exp_name, args.lambda_reg)  
    exp_name = "{}_PydDistill_{}".format(exp_name, args.lambda_pyd) if args.is_pyd_distill else exp_name 
    exp_name = "{}_CasDistill_{}".format(exp_name, args.lambda_cas) if args.is_cas_distill else exp_name  
    setproctitle(exp_name) 
    print(exp_name) 
    args.output_dir = os.path.join(args.checkpoints_dir, exp_name) 
    utils.makedir(args.output_dir) 
    if os.path.exists(os.path.join(args.output_dir, "train.log")): 
        os.remove(os.path.join(args.output_dir, "train.log"))  
    logger = utils.get_logger(os.path.join(args.output_dir, "train.log")) 
    logger.info("Logger is set - training start") 

    """ initialize and train {Pyd0_Cas0_L3} """
    args.init_levels = "3" 
    running_model = PydCasUnetRunningModel(args)  
    running_model.to_device() 
    running_model.update_optimizer() 
    # train {Pyd0_Cas0_L3} 800 
    args.iter_cnt, args.max_iters = 0, 800  
    args.max_epoch = int(args.max_iters / len(train_loader)) + 1  
    train_one_model(args, train_loader, test_loader, running_model, device, logger) 

    """ pyd evolve -> {Pyd1_Cas0_L4} """ 
    if args.is_pyd_distill: 
        running_model.set_is_pyd_distill(True) 
    running_model.pyd_evolve() 
    # trian {Pyd1_Cas0_L4} 1600 
    args.iter_cnt, args.max_iters = 0, 1600 
    args.max_epoch = int(args.max_iters / len(train_loader)) + 1  
    train_one_model(args, train_loader, test_loader, running_model, device, logger) 
    if args.is_pyd_distill: 
        running_model.reset() 

    """ cas evolve -> {Pyd1_Cas0_L4 Pyd1_Cas1_L4} """ 
    if args.is_cas_distill: 
        running_model.set_is_cas_distill(True) 
    running_model.cas_evolve() 
    # train {Pyd1_Cas0_L4 Pyd1_Cas1_L4} 1600 
    args.iter_cnt, args.max_iters = 0, 1600 
    args.max_epoch = int(args.max_iters / len(train_loader)) + 1  
    train_one_model(args, train_loader, test_loader, running_model, device, logger) 
    if args.is_cas_distill: 
        running_model.reset() 

    """ pyd evolve -> {Pyd2_Cas0_L5 Pyd2_Cas1_L5} """ 
    if args.is_pyd_distill: 
        running_model.set_is_pyd_distill(True) 
    running_model.pyd_evolve() 
    # trian {Pyd2_Cas0_L5 Pyd2_Cas1_L5} 1600 
    args.iter_cnt, args.max_iters = 0, 1600 
    args.max_epoch = int(args.max_iters / len(train_loader)) + 1  
    train_one_model(args, train_loader, test_loader, running_model, device, logger) 
    if args.is_pyd_distill: 
        running_model.reset() 

    """ cas evolve -> {Pyd2_Cas0_L5 Pyd2_Cas1_L5, Pyd2_Cas2_L5} """ 
    if args.is_cas_distill: 
        running_model.set_is_cas_distill(True) 
    running_model.cas_evolve() 
    # train {Pyd2_Cas0_L5 Pyd2_Cas1_L5, Pyd2_Cas2_L5} 1600 
    args.iter_cnt, args.max_iters = 0, 1600 
    args.max_epoch = int(args.max_iters / len(train_loader)) + 1  
    train_one_model(args, train_loader, test_loader, running_model, device, logger) 
    if args.is_cas_distill: 
        running_model.reset() 

    """ evaluate {Pyd2_Cas0_L5 Pyd2_Cas1_L5, Pyd2_Cas2_L5} """ 
    evaluate(args, test_loader, running_model, LABEL_DICT_MAPPING[args.which_set]) 
    del running_model, logger 


if __name__ == "__main__": 
    # get argument parser 
    parser = argparse.ArgumentParser() 
    parser = get_default_args_parser(parser) 
    args = parser.parse_args() 
    print(args) 

    # set dataset-related experiment name 
    exp_name = ""
    exp_name = set_dataloader_related_exp_name(args, exp_name) 
    print(exp_name)  

    args.exp_name = exp_name 

    main(args) 