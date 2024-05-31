import os 
import copy 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from utils import losses 
from utils import utils 

from typing import List, Tuple, Dict 


def conv3x3_leakyrelu(in_channels, out_channels, stride=1, groups=1, dilation=1, leaky=0.2):  
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, 
                  groups=groups, padding=dilation, dilation=dilation, bias=False), 
        nn.LeakyReLU(leaky, inplace=True)   
    ) 

class Unet(nn.Module): 
    __annotations__ = {
        "level_name_to_resolution_list": {
            '1': [1], 
            '2': [1, 1/2], 
            '3': [1, 1/2, 1/4], 
            '4': [1, 1/2, 1/4, 1/8], 
            '5': [1, 1/2, 1/4, 1/8, 1/16]
        }
    }

    def __init__(self, in_channels=1, out_channels=3, 
                 enc_channels=[8, 16, 32, 32, 32], dec_channels=[32, 32, 32, 16, 8], 
                 level=2, freezed_unet=None): 
        """ Initialize the symmetric unet. 
        
        Parameters: 
            in_channels (int): the input channels always equal to 1 for the gray level image 
            out_channels (int): the output channels, and 3 represents the 3-dimensional deformation
            enc_channels (List[int]): the channels list of encoders 
            dec_channels (List[int]): the channels list of decoders 
            level (int): the level of current unet
            freezed_unet (Unet): the trained unet note that its weights are not always freezed """
        super(Unet, self).__init__() 

        self.input_layer = nn.Conv3d(in_channels * 2, enc_channels[0], 
                                     kernel_size=3, stride=1, padding=1, bias=False) 
        self.output_layer = nn.Conv3d(dec_channels[-1], out_channels, 
                                      kernel_size=3, stride=1, padding=1, bias=False) 
        
        self.enc_layers = [] 
        self.dec_layers = [] 

        if freezed_unet is not None: 
            # copy the input layer, output layer, and all encoder and decoder layers from the trained unet 
            # NOTE use deepcopy to reallocate a memory 
            deep_copied_input_layer = copy.deepcopy(freezed_unet.input_layer) 
            deep_copied_output_layer = copy.deepcopy(freezed_unet.output_layer) 
            self.input_layer = deep_copied_input_layer 
            self.output_layer = deep_copied_output_layer 
            for freezed_enc_layer, freezed_dec_layer in zip(freezed_unet.enc_layers, freezed_unet.dec_layers): 
                deep_copied_freezed_enc_layer = copy.deepcopy(freezed_enc_layer) 
                deep_copied_freezed_dec_layer = copy.deepcopy(freezed_dec_layer) 
                self.enc_layers.append(deep_copied_freezed_enc_layer) 
                self.dec_layers.append(deep_copied_freezed_dec_layer) 

        assert len(self.enc_layers) == len(self.dec_layers), "The number of enc_layers must equal to the number of dec_layers" 

        assert level >= len(self.enc_layers), "The number of layers in new unet must be equal to or larger than that of the old one" 

        self.intermediate_layer = None 
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) 

        self.enc_layers = nn.ModuleList(self.enc_layers) 
        self.dec_layers = nn.ModuleList(self.dec_layers) 

        self.level = level 

    def unfreeze_layers(self): 

        def _unfreeze_layer(layer: nn.Module): 
            for param in layer.parameters(): 
                param.requires_grad = True             

        # release the weights of all layers (input, output, encoders, intermediate, decoders) 
        for enc_layer, dec_layer in zip(self.enc_layers, self.dec_layers): 
            _unfreeze_layer(enc_layer) 
            _unfreeze_layer(dec_layer) 
        _unfreeze_layer(self.input_layer) 
        _unfreeze_layer(self.output_layer)
        _unfreeze_layer(self.intermediate_layer) 

    def forward(self, moving, fixed): 
        """ Forward the symmetric unet. 
        
        Parameters: 
            moving (Tensor): the moving image [B, 1, D, H, W] 
            fixed (Tensor): the fixed image [B, 1, D, H, W] 
            
        Returns: 
            flow (Tensor): the deformation field or
                           the stationary velocity field (in diffeomorphic mode) [B, 3, D, H, W]  
            enc_results (List[Tensor]): the feature maps of all encoder layers
            inter_feature (Tensor): the feature map of the intermediate layer 
            dec_results (List[Tensor]): the feature maps of all decoder layers """
        
        # input 
        x = torch.cat([moving, fixed], dim=1) 
        x = self.input_layer(x) 

        # encode 
        enc_results = [] 
        for idx, enc_layer in enumerate(self.enc_layers): 
            x = enc_layer(x) 
            enc_results.append(x) 

        # intermediate 
        x = self.intermediate_layer(x) 
        inter_feature = x 

        # decode 
        dec_results = [] 
        for idx, dec_layer in enumerate(self.dec_layers): 
            x = self.upsample_layer(x) 
            x = torch.cat([x, enc_results[-(idx+1)]], dim=1) 
            x = dec_layer(x) 
            dec_results.append(x) 

        # output 
        flow = self.output_layer(x) 

        return flow, enc_results, inter_feature, dec_results


class PydCasUnetRunningModel(object): 
    def __init__(self, args): 
        super(PydCasUnetRunningModel, self).__init__() 

        self.args = args 
        self.output_dir = args.output_dir 
        self.device = torch.device(args.device) 
        
        self.label_dict = None 

        # set loss weight 
        self.sim_criterion = losses.SingleNCCLoss() 
        self.reg_criterion = losses.SingleGradLoss() 

        self.seg_criterion = losses.DiceLoss()  

        self.weight_dict = {
            'sim': 1, 
            'seg': 1, 
            'reg': args.lambda_reg, 
            'pyd': args.lambda_pyd, 
            'cas': args.lambda_cas
        } 
        
        # initialize models 
        self.models = nn.ModuleDict() 
        for idx, level in enumerate(args.init_levels.split(",")): 
            level = int(level) 
            self.models.update({
                "Pyd0_Cas{}_L{}".format(idx, level): Unet(level=level) 
            })

        self.is_diff = False # is to use diffeomorphic flows 
        self.is_semi_sup = False # is to use semi-supervised learning 

        """ These attributes will be changed frequently """
        self.optimizer = None 
        self.lr_scheduler = None 

        self.warped_images = {} 
        self.warped_labels = {} 
        self.flows = {} 
        self.features = {} 

        self.is_distill_online = True  
        self.is_pyd_distill = False 
        self.is_cas_distill = False 

    def pyd_evolve(self): 
        """ apply once the pyramid evolution, and the distillation can be initialized here too """
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        
        old_pyd_idx = int(model_names[0][model_names[0].index('Pyd')+3]) 
        new_pyd_idx = old_pyd_idx + 1 
        old_level = int(model_names[0][model_names[0].index('L')+1]) 
        new_level = old_level + 1  

        for model_name in model_names: 
            # create the new model 
            new_model_name = model_name 
            new_model_name = new_model_name.replace("Pyd{}".format(old_pyd_idx), "Pyd{}".format(new_pyd_idx)) 
            new_model_name = new_model_name.replace("L{}".format(old_level), "L{}".format(new_level)) 

            old_model = self.models[model_name] 
            self._freeze_weight(old_model) # freeze the weights of old model 
            new_model = Unet(level=new_level, freezed_unet=old_model.cpu()) # create the new model using the old one 
            if self.is_distill_online: # if the online distillation is choosed, then unfreeze the new model's weights 
                self._unfreeze_weight(new_model) 
            else: # else the offline distillation is used, which means most of layers in the new model will be freezed 
                # specifically, the freezed layers include the input, output, all old encoder and decoder ones 
                pass 
            self.models[new_model_name] = new_model.to(self.device) 

            # delete the old model 
            self.models.pop(model_name) 

            # add the auxiliary pyramid distillation loss (optional) 
            if self.is_pyd_distill: 
                self.models.update({ 
                    "{}_Pyd_Criterion".format(new_model_name): losses.PydLoss(old_level, new_level) 
                }) 

        self.to_device() 
        self.update_optimizer() 

    def cas_evolve(self): 
        """ apply once the cascaded evolution, and the distillation can be initialized here too """
        model_names = list(self.models.keys())  
        model_names = self.filter_distill_model_names(model_names) 

        cur_cas_idx = int(model_names[-1][model_names[-1].index("Cas")+3]) 
        cur_level = int(model_names[-1][model_names[-1].index("L")+1]) 

        # freeze the weights of all current existing networks 
        for i in range(0, cur_cas_idx + 1): 
            template_model_name = model_names[-1] 
            template_model_name.replace("Cas{}".format(cur_cas_idx), "Cas{}".format(i))
            this_model_name = template_model_name 
            self._freeze_weight(self.models[this_model_name]) 

        # append a new network behind the current last one  
        new_model_name = model_names[-1].replace("Cas{}".format(cur_cas_idx), "Cas{}".format(cur_cas_idx+1)) 
        new_model = Unet(level=cur_level, freezed_unet=self.models[model_names[-1]].cpu()) 
        new_model.unfreeze_layers() # open the copied weights of the new network 
        self.models[new_model_name] = new_model.to(self.device) 

        if self.is_distill_online: # if the online disitllation is choosed, then the freezed pre-staged networks will be released 
            # in this way, the weights of all networks are opened 
            self.unfreeze_weights() 
        else: # else the offline distillation is choosed, then the pre-staged networks are still freezed 
            # in this way, the weights of the newest network are opened, the weights of all previous existing networks are freezed 
            pass 

        # add the auxiliary cascaded distillation loss (optional) 
        if self.is_cas_distill: 
            self.models.update({
                "{}_Cas_Criterion".format(new_model_name): losses.CasLoss()  
            })

        self.to_device() 
        self.update_optimizer() 

    def set_input_data(self, inputs: Dict): 
        self.moving_image = inputs['moving_image'].to(self.device) 
        self.fixed_image = inputs['fixed_image'].to(self.device) 
        self.moving_label = inputs['moving_label'].to(self.device) 
        self.fixed_label = inputs['fixed_label'].to(self.device)

    def forward(self): 
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        
        if len(model_names) > 1: # multi cascaded models 
            for model_name in model_names: 
                agg_flow = None 

                cur_cas_idx = int(model_name[model_name.find('Cas') + 3]) 
                if cur_cas_idx == 0: # the first one 
                    flow, enc_features, inter_feature, dec_features = self.models[model_name](self.moving_image, self.fixed_image) 
                    if self.is_diff: 
                        flow = utils.convert_diff(flow) 
                    self.flows.update({model_name: flow}) 
                    warped_image = utils.warp3d(self.moving_image, flow) 
                    self.warped_images.update({model_name: warped_image}) 
                else: # the left models 
                    pre_model_name = model_names[cur_cas_idx-1] 
                    pre_warped_image = self.warped_images[pre_model_name] 
                    flow, enc_features, inter_feature, dec_features = self.models[model_name](pre_warped_image, self.fixed_image) 
                    if self.is_diff: 
                        flow = utils.convert_diff(flow) 
                    pre_flow = self.flows[pre_model_name] 
                    agg_flow = flow + utils.warp3d(pre_flow, flow) 
                    self.flows.update({model_name: agg_flow}) 
                    warped_image = utils.warp3d(self.moving_image, agg_flow) 
                    self.warped_images.update({model_name: warped_image})  

                if self.is_cas_distill or self.is_pyd_distill:  
                    self.features.update({
                        model_name: {
                            "enc_features": enc_features, 
                            "inter_feature": inter_feature, 
                            "dec_features": dec_features 
                        }
                    }) 
                
                if self.is_semi_sup: 
                    if agg_flow is not None: 
                        warped_label = utils.warp3d(self.moving_label, agg_flow) 
                    else: 
                        warped_label = utils.warp3d(self.moving_label, flow) 
                    self.warped_labels.update({model_name: warped_label}) 
                    
        else: # only one model 
            flow, enc_features, inter_feature, dec_features = self.models[model_names[0]](self.moving_image, self.fixed_image) 
            if self.is_diff: 
                flow = utils.convert_diff(flow) 
            self.flows.update({model_names[0]: flow}) 
            warped_image = utils.warp3d(self.moving_image, flow) 
            self.warped_images.update({model_names[0]: warped_image}) 

            if self.is_pyd_distill: 
                self.features.update({
                    model_names[0]: {
                        "enc_features": enc_features, 
                        "inter_feature": inter_feature, 
                        "dec_features": dec_features 
                    }
                })
            
            if self.is_semi_sup: 
                warped_label = utils.warp3d(self.moving_label, flow) 
                self.warped_labels.update({model_names[0]: warped_label})

    def backward(self): 
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        loss_dict = {} 

        if len(model_names) > 1: # cascade multi models 
            """ Calculate the similarity and regulization loss first """
            losses_sim = 0 
            losses_reg = 0 
            for model_name in model_names: 
                warped_image = self.warped_images[model_name] 
                loss_sim = self.sim_criterion(warped_image, self.fixed_image) 
                flow = self.flows[model_name] 
                loss_reg = self.reg_criterion(flow) 
                losses_sim += loss_sim 
                losses_reg += loss_reg 
            loss_sim = losses_sim / len(model_names) 
            loss_reg = losses_reg / len(model_names) 
            loss_dict.update({'sim': loss_sim}) 
            loss_dict.update({'reg': loss_reg}) 

            """ Calculate the pyd distillation loss if needed, note that all cascaded models need to be considered """ 
            if self.is_pyd_distill: 
                losses_pyd = 0 
                for model_name in model_names: 
                    # prepare model-specified criterion first 
                    pyd_criterion_per_model = self.models["{}_Pyd_Criterion".format(model_name)] 
                    # prepare corresonding feature maps then 
                    features_dict_per_model = self.features[model_name] 
                    enc_features = features_dict_per_model["enc_features"] 
                    enc_feature_pair = [enc_features[-2], enc_features[-1]] 
                    dec_features = features_dict_per_model["dec_features"] 
                    dec_feature_pair = [dec_features[0], dec_features[1]] 
                    inter_feature = features_dict_per_model["inter_feature"] 
                    # accumulate the pyd distillation loss 
                    losses_pyd += pyd_criterion_per_model(enc_feature_pair, inter_feature, dec_feature_pair)  
                loss_pyd = losses_pyd / len(model_name) 
                loss_dict.update({'pyd': loss_pyd}) 

            """ Calculate the cas distillation loss if needed, note that only the last two models need to be considered """ 
            if self.is_cas_distill: 
                loss_cas = 0 
                # prepare the cas criterion for the latest model 
                cur_model_name = model_names[-1] 
                cas_criterion_for_cur_model = self.models["{}_Cas_Criterion".format(cur_model_name)] 
                # prepare the previous and current models' features 
                pre_model_name = model_names[-2] 
                features_dict_of_pre_model = self.features[pre_model_name] 
                features_list_of_pre_model = [] 
                for k, v in features_dict_of_pre_model.items(): 
                    if k == "enc_features" or k == "dec_features": 
                        features_list_of_pre_model += v 
                    else: # "inter_feature" 
                        features_list_of_pre_model += [v] 
                features_dict_of_cur_model = self.features[cur_model_name] 
                features_list_of_cur_model = [] 
                for k, v in features_dict_of_cur_model.items(): 
                    if k == "enc_features" or k == "dec_features": 
                        features_list_of_cur_model += v 
                    else: # "inter_feature" 
                        features_list_of_cur_model += [v] 
                # calculate the cas distillation loss 
                loss_cas = cas_criterion_for_cur_model(features_list_of_pre_model, features_list_of_cur_model) 
                loss_dict.update({'cas': loss_cas}) 

            if self.is_semi_sup: 
                losses_seg = 0 
                for model_name in model_names: 
                    warped_label = self.warped_labels[model_name] 
                    loss_seg = self.seg_criterion(warped_label, self.fixed_label) 
                    losses_seg += loss_seg 
                loss_seg = losses_seg / len(model_names) 
                loss_dict.update({'seg': loss_seg}) 
                 
        else: # only one model 
            warped_image = self.warped_images[model_names[0]] 
            loss_sim = self.sim_criterion(warped_image, self.fixed_image) 
            flow = self.flows[model_names[0]] 
            loss_reg = self.reg_criterion(flow) 
            loss_dict.update({'sim': loss_sim}) 
            loss_dict.update({'reg': loss_reg}) 

            if self.is_pyd_distill: 
                enc_features = self.features[model_names[0]]["enc_features"] 
                enc_feature_pair = [enc_features[-2], enc_features[-1]] 
                dec_features = self.features[model_names[0]]["dec_features"] 
                dec_feature_pair = [dec_features[0], dec_features[1]] 
                inter_feature = self.features[model_names[0]]["inter_feature"] 
                loss_pyd = self.models["{}_Pyd_Criterion".format(model_names[0])](enc_feature_pair, inter_feature, dec_feature_pair) 
                loss_dict.update({'pyd': loss_pyd}) 

            if self.is_semi_sup: 
                warped_label = self.warped_labels[model_names[0]] 
                loss_seg = self.seg_criterion(warped_label, self.fixed_label) 
                loss_dict.update({'seg': loss_seg})  
            
        # calculate the total loss and backward 
        loss = 0 
        for k, v in loss_dict.items(): 
            loss += self.weight_dict[k] * v 
        loss.backward() 
        
        # save losses 
        _loss_dict = {} 
        _loss_dict.update({"loss": loss.item()}) 
        for k, v in loss_dict.items(): 
            _loss_dict.update({"loss_{}_unscaled".format(k): v.item()}) 
        for k, v in loss_dict.items(): 
            _loss_dict.update({"loss_{}_scaled".format(k): (self.weight_dict[k] * v).item()}) 
        self.loss_dict = _loss_dict 

    def optim_params(self): 
        self.optimizer.zero_grad() 
        self.forward() 
        self.backward() 
        self.optimizer.step() 

    def cal_metrics_validate(self): 
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        dice_dict, jacz_dict = {}, {} 
        for model_name in model_names: 
            warped_label = utils.warp3d(self.moving_label, self.flows[model_name])  
            dice = losses.Dice(warped_label, self.fixed_label, True) 
            jacz = (losses.Jac(self.flows[model_name]) < 0).float().mean() 
            dice_dict.update({model_name: dice.item()}) 
            jacz_dict.update({model_name: jacz.item()}) 
        self.dice, self.jacz = dice_dict, jacz_dict

    def cal_metrics_evaluate(self): 
        assert self.label_dict is not None, "Label dict must be set before calculating metrics during evaluation"
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        dice_dict_for_table = {} 
        dice_dict_for_boxplot = {} 
        num_folds_dict = {} 
        folds_ratio_dict = {} 
        for model_name in model_names: 
            warped_label = utils.warp3d(self.moving_label, self.flows[model_name]) 
            self.warped_labels.update({model_name: warped_label})  
            # add dice result for table 
            dice_for_table = losses.Dice(warped_label, self.fixed_label, True) 
            # add dice result for boxplot 
            dice_for_boxplot = utils.cal_dice_torch(self.fixed_label, warped_label, self.label_dict) 
            # add num_folds and folds ratio for table 
            _, num_folds, folds_ratio = utils.cal_jacobian_determinant_torch(self.flows[model_name])  
            # save 
            dice_dict_for_table.update({model_name: dice_for_table.item()}) 
            dice_dict_for_boxplot.update({model_name: dice_for_boxplot}) 
            num_folds_dict.update({model_name: num_folds.item()}) 
            folds_ratio_dict.update({model_name: folds_ratio.item()}) 
        self.dice_dict_for_table = dice_dict_for_table 
        self.dice_dict_for_boxplot = dice_dict_for_boxplot 
        self.num_folds_dict = num_folds_dict 
        self.folds_ratio_dict = folds_ratio_dict 

    def evaluate(self): 
        with torch.no_grad(): 
            self.forward() 
            self.cal_metrics_evaluate() 

    def set_is_distill_online(self, option: bool): 
        self.is_distill_online = option   

    def set_is_pyd_distill(self, option: bool): 
        self.is_pyd_distill = option  

    def set_is_cas_distill(self, option: bool): 
        self.is_cas_distill = option 

    def set_is_diff(self, option: bool): 
        self.is_diff = option 

    def set_is_semi_sup(self, option: bool): 
        self.is_semi_sup = option 

    def set_label_dict(self, label_dict: Dict): 
        self.label_dict = label_dict 

    def set_train(self): 
        for model_name, model in self.models.items(): 
            model.train() 

    def set_eval(self): 
        for model_name, model in self.models.items(): 
            model.eval() 

    def get_loss(self): 
        return self.loss_dict 

    def get_metrics_validate(self): 
        return self.dice, self.jacz 

    def get_metrics_evaluate(self): 
        return {
            "dice_dict_for_table": self.dice_dict_for_table, 
            "dice_dict_for_boxplot": self.dice_dict_for_boxplot, 
            "num_folds_dict": self.num_folds_dict, 
            "folds_ratio_dict": self.folds_ratio_dict 
        }
    
    def get_forward_results_evaluate(self): 
        return {
            "warped_images_dict": self.warped_images, 
            "warped_labels_dict": self.warped_labels, 
            "flows_dict": self.flows 
        }
    
    def get_model_names(self, is_str=True): 
        if is_str: 
            message = "" 
            for model_name in list(self.models.keys()): 
                message += "{} ".format(model_name) 
            return message 
        else: 
            return list(self.models.keys()) 

    def _freeze_weight(self, model: nn.Module): 
        for param in model.parameters(): 
            param.requires_grad = False 

    def _unfreeze_weight(self, model: nn.Module): 
        for param in model.parameters(): 
            param.requires_grad = True 

    def freeze_weights(self): 
        for model_name, model in self.models.items(): 
            self._freeze_weight(model)  
        
        self.update_optimizer() 

    def unfreeze_weights(self): 
        for model_name, model in self.models.items(): 
            for param in model.parameters(): 
                param.requires_grad = True 

        self.update_optimizer() 

    def to_device(self): 
        model_names = list(self.models.keys()) 
        for name in model_names: 
            self.models[name].to(self.device) 

    def update_optimizer(self): 
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.models.parameters()), self.args.lr, 
                                          betas=(0.9, 0.999), weight_decay=self.args.weight_decay) 
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.max_epoch)  

    def filter_distill_model_names(self, model_names): 
        _model_names = [] 
        for name in model_names: 
            if name.find("Pyd_Criterion") != -1 or name.find("Cas_Criterion") != -1:  
                continue 
            _model_names.append(name) 
        return _model_names 
    
    def reset(self): 
        # release all distillation related criterions 
        model_names = list(self.models.keys())
        for model_name in model_names: 
            if model_name.find("Pyd_Criterion") != -1 or model_name.find("Cas_Criterion") != -1: 
                self.models.pop(model_name) 

        # reset all buffered images, labels, flows, and features 
        self.warped_images.clear() 
        self.warped_labels.clear() 
        self.flows.clear() 
        self.features.clear() 

        # reset the control signals 
        self.is_distill_online = True 
        self.is_pyd_distill = False 
        self.is_cas_distill = False 

        self.unfreeze_weights() 
        self.to_device() 
        self.update_optimizer() 
    
    def load(self, wanted_model_names: List[str], suffixes: List[str] = None):   
        self.models.clear() 
        if suffixes is not None: 
            assert len(suffixes) == len(wanted_model_names), \
                "The number of suffixes must equal to the number of wanted models" 
            # NOTE a demo: model_name = "Pyd2_Cas2_L5" suffix = "iter_1600_dice_0.840" 
            for model_name, suffix in zip(wanted_model_names, suffixes): 
                trained_model_name = "{}_{}.pth".format(model_name, suffix) 
                trained_model_path = os.path.join(self.output_dir, trained_model_name) 
                trained_model_weight = torch.load(trained_model_path) 
                which_level = int(model_name[model_name.find('L') + 1]) 
                model = Unet(level=which_level) 
                model.load_state_dict(trained_model_weight) 
                model.to(self.device) 
                self.models.update({model_name: model}) 
        else: # suffixes are not given, use the best weight defaultly 
            for model_name in wanted_model_names: 
                trained_model_name = "{}_best.pth".format(model_name) 
                trained_model_path = os.path.join(self.output_dir, trained_model_name) 
                trained_model_weight = torch.load(trained_model_path) 
                which_level = int(model_name[model_name.find('L') + 1]) 
                model = Unet(level=which_level) 
                model.load_state_dict(trained_model_weight) 
                model.to(self.device) 
                self.models.update({model_name: model}) 

    def load_external(self, wanted_model_names: List[str], corresponding_weight_paths: List[str]): 
        self.models.clear() 
        assert len(wanted_model_names) == len(corresponding_weight_paths), \
        "The number of wanted models must equal to the number of weight paths" 
        for model_name, weight_path in zip(wanted_model_names, corresponding_weight_paths): 
            trained_model_weight = torch.load(weight_path) 
            which_level = int(model_name[model_name.find('L') + 1]) 
            model = Unet(level=which_level) 
            model.load_state_dict(trained_model_weight) 
            model.to(self.device) 
            self.models.update({model_name: model})

    def save(self, suffix: Dict): 
        model_names = list(self.models.keys()) 
        model_names = self.filter_distill_model_names(model_names) 
        for model_name in model_names: 
            save_name = "{}_{}.pth".format(model_name, suffix[model_name]) 
            save_path = os.path.join(self.output_dir, save_name) 
            torch.save(self.models[model_name].state_dict(), save_path) 

    def print_info(self): 
        message = "" 
        message += "Model info: \n"
        for model_name, model in self.models.items(): 
            message += "Name: {}, model: {} \n".format(model_name, model) 
        for model_name, warped_image in self.warped_images.items(): 
            message += "Name: {}, warped image shape: {} \n".format(model_name, warped_image.shape) 
        for model_name, flow in self.flows.items(): 
            message += "Name: {}, flow shape: {} \n".format(model_name, flow.shape)  
        print(message) 

    def adjust_lr(self): 
        self.lr_scheduler.step() 
