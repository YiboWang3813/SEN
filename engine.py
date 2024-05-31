import os 
import json 
import torch
import nibabel 
import numpy as np

from utils import utils 


def train_one_model(args, train_loader, test_loader, running_model, device, logger): 

    best_dice_per_model_dict = {} 
    for model_name in running_model.get_model_names(False): 
        best_dice_per_model_dict.update({model_name: 0.0})  

    for epoch in range(args.max_epoch): 
        if args.iter_cnt == args.max_iters: 
            logger.info("stop to train {}".format(running_model.get_model_names())) 
            break 

        for step, data in enumerate(train_loader): 
            # load data to cpu, running model will move them to gpu automatically 
            moving_image, fixed_image, moving_label, fixed_label = data 
            running_model.set_input_data({
                'moving_image': moving_image, 'fixed_image': fixed_image, 
                'moving_label': moving_label, 'fixed_label': fixed_label
            })

            # running model optim params 
            running_model.optim_params() 

            # running model get loss 
            loss = running_model.get_loss() 

            # running model calculate metrics 
            running_model.cal_metrics_validate() 
            dice_dict, jacz_dict = running_model.get_metrics_validate() 

            args.iter_cnt += 1 
            # log the total loss 
            message = "Train: " 
            message += "model: {}, ".format(running_model.get_model_names())
            message += "iteration: {}, ".format(args.iter_cnt) 
            # message += "loss: {:.3f}".format(loss) 
            for loss_name, loss_value in loss.items(): 
                message += "{}: {:.3f} ".format(loss_name, loss_value) 
            logger.info(message)  
            # log the dice and jacz of each sub models 
            for model_name in list(dice_dict.keys()): 
                message = "Train: " 
                message += "model: {}, ".format(model_name) 
                message += "iteration: {}, ".format(args.iter_cnt) 
                message += "dice: {:.3f}, ".format(dice_dict[model_name]) 
                message += "jacz: {:.3f}".format(jacz_dict[model_name]) 
                logger.info(message)  

            # validate 
            if args.iter_cnt >= args.save_freq and args.iter_cnt % args.save_freq == 0: 
                all_steps_dice_dict = validate(args, test_loader, running_model, device, logger, args.iter_cnt) 

                suffix_dict = {} 

                for model_name, all_steps_dice_list in all_steps_dice_dict.items(): 
                    avg_dice = np.mean(all_steps_dice_list) 

                    # save each best model 
                    if avg_dice > best_dice_per_model_dict[model_name]: 
                        best_dice_per_model_dict[model_name] = avg_dice 
                        save_name = "{}_best.pth".format(model_name) 
                        save_path = os.path.join(args.output_dir, save_name)  
                        torch.save(running_model.models[model_name].state_dict(), save_path) 
                
                    suffix_dict.update({
                        model_name: "iter_{}_dice_{:.3f}".format(args.iter_cnt, avg_dice)
                    })

                # save latest model 
                running_model.save(suffix_dict) 
                running_model.set_train()               

            if args.iter_cnt == args.max_iters: 
                logger.info("stop to train {}".format(running_model.get_model_names())) 
                break 
                
        running_model.adjust_lr()   


def validate(args, test_loader, running_model, device, logger, iter_num): 

    all_steps_dice_dict = {} 
    all_steps_jacz_dict = {} 
    
    running_model.set_eval() 
    with torch.no_grad():
        for step, data in enumerate(test_loader): 
            moving_image, fixed_image, moving_label, fixed_label = data 
            running_model.set_input_data({
                'moving_image': moving_image, 'fixed_image': fixed_image, 
                'moving_label': moving_label, 'fixed_label': fixed_label
            })

            N = moving_image.size(0) # batch size 

            # running model forward 
            running_model.forward() 

            # running model cal metrics 
            running_model.cal_metrics_validate() 
            dice_dict, jacz_dict = running_model.get_metrics_validate() 

            # appeng this step's metrics 
            utils.append_metric_dict(dice_dict, all_steps_dice_dict) 
            utils.append_metric_dict(jacz_dict, all_steps_jacz_dict)  

            if step >= args.max_samples - 1:
                break 

    # log the dice and jacz of each sub models 
    for model_name in list(all_steps_dice_dict.keys()): 
        message = "Valid: " 
        message += "model: {}, ".format(model_name) 
        message += "iteration: {}, ".format(iter_num) 
        message += "dice: {:.3f}, ".format(np.mean(all_steps_dice_dict[model_name])) 
        message += "jacz: {:.3f}".format(np.mean(all_steps_jacz_dict[model_name]))  
        logger.info(message)  
    
    return all_steps_dice_dict 


@torch.no_grad() 
def evaluate(args, test_loader, running_model, label_dict):  
    for step, data in enumerate(test_loader): 
        moving_image, fixed_image, moving_label, fixed_label = data 
        running_model.set_input_data({
            'moving_image': moving_image, 'fixed_image': fixed_image, 
            'moving_label': moving_label, 'fixed_label': fixed_label
        })

        N = moving_image.size(0) # batch size 
        assert N == 1, "Batch size must be 1 during evaluation" 

        running_model.set_label_dict(label_dict) 
        running_model.forward() 
        running_model.cal_metrics_evaluate()  
        metrics_evaluate = running_model.get_metrics_evaluate() 
        dice_dict_for_table = metrics_evaluate["dice_dict_for_table"] 
        dice_dict_for_boxplot = metrics_evaluate["dice_dict_for_boxplot"] 
        num_folds_dict = metrics_evaluate["num_folds_dict"] 
        folds_ratio_dict = metrics_evaluate["folds_ratio_dict"] 
        forward_results_evaluate = running_model.get_forward_results_evaluate() 
        warped_images_dict = forward_results_evaluate["warped_images_dict"] 
        warped_labels_dict = forward_results_evaluate["warped_labels_dict"] 
        flows_dict = forward_results_evaluate["flows_dict"] 

        save_dir = os.path.join(args.output_dir, "samples", "sample_{}".format(step)) 
        utils.makedir(save_dir) 
        
        affine_arr = np.eye(4) 
        model_names = running_model.get_model_names(False) 

        """ 
        # save results of all models 
        for model_name in model_names: 
            # save model-related metrics 
            # dice for table 
            dice_for_table_save_path = os.path.join(save_dir, "dice_for_table_{}.json".format(model_name)) 
            with open(dice_for_table_save_path, "w") as f:
                json.dump({"dice": dice_dict_for_table[model_name]}, f) 
            f.close() 
            # dice for boxplot 
            dice_for_boxplot_save_path = os.path.join(save_dir, "dice_for_boxplot_{}.json".format(model_name)) 
            with open(dice_for_boxplot_save_path, "w") as f:
                json.dump(dice_dict_for_boxplot[model_name], f)  
            f.close() 
            # num_folds 
            num_folds_save_path = os.path.join(save_dir, "num_folds_{}.json".format(model_name)) 
            with open(num_folds_save_path, "w") as f: 
                json.dump({"num_folds": num_folds_dict[model_name]}, f) 
            f.close() 
            # folds ratio 
            folds_ratio_save_path = os.path.join(save_dir, "folds_ratio_{}.json".format(model_name)) 
            with open(folds_ratio_save_path, "w") as f: 
                json.dump({"folds_ratio": folds_ratio_dict[model_name]}, f) 
            f.close() 

            # save model-related images 
            # warped image 
            warped_image = utils.adjust_image(warped_images_dict[model_name].squeeze(0).cpu().numpy()) 
            warped_image_save_path = os.path.join(save_dir, "warped_image_{}.nii.gz".format(model_name)) 
            nibabel.Nifti1Image(warped_image, affine_arr).to_filename(warped_image_save_path) 
            # warped label 
            warped_label = utils.adjust_label(warped_labels_dict[model_name].squeeze(0).cpu().numpy()) 
            warped_label_save_path = os.path.join(save_dir, "warped_label_{}.nii.gz".format(model_name)) 
            nibabel.Nifti1Image(warped_label, affine_arr).to_filename(warped_label_save_path) 

            # save model-related flows in top-5 steps 
            if step < 5: 
                # flow 
                flow = flows_dict[model_name].squeeze(0).permute(1, 2, 3, 0).cpu().numpy() 
                flow_save_path = os.path.join(save_dir, "disp_{}.nii.gz".format(model_name)) 
                nibabel.Nifti1Image(flow, affine_arr).to_filename(flow_save_path) 
        """
        
        """ save results of the latest model """ 
        last_model_name = model_names[-1] 
        # dice for table 
        dice_for_table_save_path = os.path.join(save_dir, "dice_for_table.json") 
        with open(dice_for_table_save_path, "w") as f:
            json.dump({"dice": dice_dict_for_table[last_model_name]}, f) 
        f.close() 
        # dice for boxplot 
        dice_for_boxplot_save_path = os.path.join(save_dir, "dice_for_boxplot.json") 
        with open(dice_for_boxplot_save_path, "w") as f:
            json.dump(dice_dict_for_boxplot[last_model_name], f)  
        f.close() 
        # num_folds 
        num_folds_save_path = os.path.join(save_dir, "num_folds.json") 
        with open(num_folds_save_path, "w") as f: 
            json.dump({"num_folds": num_folds_dict[last_model_name]}, f) 
        f.close() 
        # folds ratio 
        folds_ratio_save_path = os.path.join(save_dir, "folds_ratio.json") 
        with open(folds_ratio_save_path, "w") as f: 
            json.dump({"folds_ratio": folds_ratio_dict[last_model_name]}, f) 
        f.close() 
        # warped image 
        warped_image = utils.adjust_image(warped_images_dict[last_model_name].squeeze(0).cpu().numpy()) 
        warped_image_save_path = os.path.join(save_dir, "warped_image.nii.gz") 
        nibabel.Nifti1Image(warped_image, affine_arr).to_filename(warped_image_save_path) 
        # warped label 
        warped_label = utils.adjust_label(warped_labels_dict[last_model_name].squeeze(0).cpu().numpy()) 
        warped_label_save_path = os.path.join(save_dir, "warped_label.nii.gz") 
        nibabel.Nifti1Image(warped_label, affine_arr).to_filename(warped_label_save_path) 
        # save model-related flows in top-5 steps 
        if step < 5: 
            # flow 
            flow = flows_dict[last_model_name].squeeze(0).permute(1, 2, 3, 0).cpu().numpy() 
            flow_save_path = os.path.join(save_dir, "disp.nii.gz") 
            nibabel.Nifti1Image(flow, affine_arr).to_filename(flow_save_path) 

        """ save model-unrelated images and labels """ 
        # moving image 
        moving_image = utils.adjust_image(moving_image.squeeze(0).cpu().numpy()) 
        moving_image_save_path = os.path.join(save_dir, "moving_image.nii.gz") 
        nibabel.Nifti1Image(moving_image, affine_arr).to_filename(moving_image_save_path) 
        # fixed image 
        fixed_image = utils.adjust_image(fixed_image.squeeze(0).cpu().numpy()) 
        fixed_image_save_path = os.path.join(save_dir, "fixed_image.nii.gz") 
        nibabel.Nifti1Image(fixed_image, affine_arr).to_filename(fixed_image_save_path) 
        # moving label  
        moving_label = utils.adjust_label(moving_label.squeeze(0).cpu().numpy()) 
        moving_label_save_path = os.path.join(save_dir, "moving_label.nii.gz") 
        nibabel.Nifti1Image(moving_label, affine_arr).to_filename(moving_label_save_path) 
        # fixed label  
        fixed_label = utils.adjust_label(fixed_label.squeeze(0).cpu().numpy()) 
        fixed_label_save_path = os.path.join(save_dir, "fixed_label.nii.gz") 
        nibabel.Nifti1Image(fixed_label, affine_arr).to_filename(fixed_label_save_path) 

        if step >= 50:
            break
