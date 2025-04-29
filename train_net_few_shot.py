#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import os
import oss2 as oss
import torch.nn as nn
from copy import deepcopy
import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
from utils.meters import TrainMeter, ValMeter
from ipdb import set_trace
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset
from torch.cuda.amp import GradScaler, autocast
from datasets.utils.mixup import Mixup
import gc
logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer=None, val_meter=None, val_loader=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    scaler = GradScaler()
    if cfg.TRAIN.BACKBONE_FROZEN:
        for name, p in model.named_parameters():
            p.requires_grad_(False)
            if "head.backbone" in name:
                p.requires_grad_(False)
                # if 'adapter' in name:
                #     p.requires_grad_(True)
                #     print(name)
                if 'backbone.proj' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                if 'backbone.attnpool' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                # if 'backbone.ml_ln_post' in name:
                #     p.requires_grad_(True)
                #     print(name)
                # if 'backbone.ml_visu_proj' in name:
                #     p.requires_grad_(True)
                #     print(name)
                if 'backbone.layer4.2' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                if 'backbone.transformer.resblocks.11' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                # if 'backbone.transformer.resblocks.9' in name: #DO
                #     p.requires_grad_(True)
                #     print(name)
                if 'backbone.transformer.resblocks.7' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                if 'backbone.transformer.resblocks.3' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                if 'backbone.layer4.1' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                if 'backbone.ln_post' in name: #DO
                    p.requires_grad_(True)
                    print(name)
                # if 'lateral_conv' in name:
                #     p.requires_grad_(True)
                #     print(name)
                # if 'ml_attnpool' in name:
                #     p.requires_grad_(True)
                #     print(name)
            elif 'head.scale' in name:
                p.requires_grad_(False)
            # elif 'head.initial_backbone' in name:
            #     p.requires_grad_(False)
            elif 'head.dec' in name:
                p.requires_grad_(True)
                print(name)
            elif 'ctx' in name:
                p.requires_grad_(False)
                # print(name)
            elif 'prompt' in name:
                p.requires_grad_(False)
            elif 'head.transformer' in name:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
                print(name)
    # torch.autograd.set_detect_anomaly(True)
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")

    # for name, module in model.named_modules():
    #     print(f"Name: {name}")
    #
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     base_model = model.module
    # else:
    #     base_model = model
    #
    # # 然后尝试获取子模块 DiT
    # DiT_module = getattr(base_model, 'DiT', None)
    # scale = getattr(base_model, 'scale', None)
    #
    # print("\nModel Modules:")
    # for name, module in base_model.named_modules():
    #     print(f"{name}: {module}")
    #
    # # 检查 DiT_module 是否存在
    # if DiT_module is None:
    #     raise AttributeError(f"'{type(base_model).__name__}' object has no attribute 'DiT'")
    #
    #


    train_meter.iter_tic()
    data_size = len(train_loader)

    data_size = cfg.SOLVER.STEPS_ITER
    Et_Dev_lastepo = None
    target_features = []
    cofficients = []
    for cur_iter, task_dict in enumerate(train_loader):
        
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', "real_support_labels"]'''
        if cur_iter >= cfg.TRAIN.NUM_TRAIN_TASKS:
                break
        # Save a checkpoint.
        cur_epoch = cur_iter//cfg.SOLVER.STEPS_ITER
        # if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0 and cur_iter>=200:   # 
        if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0: #(cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0:   #
            
            if cfg.OSS.ENABLE:
                model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            else:
                model_bucket = None
            cur_epoch_save = cur_iter//cfg.TRAIN.VAL_FRE_ITER
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, model_ema, optimizer, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, model_bucket)

            # model.eval()
            train_DiT_epoch(target_features, cofficients, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, cur_epoch_save, cur_iter, writer=None, val_meter=None, val_loader=None)
            target_features = []
            cofficients = []
            #先model.eval, eval一下，获得和target_features，把数据对保存下来,concat[target_features.detach().clone(),support_features.detach().clone()]
            ##model.train(),把grad_require重置一下
            ####然后Dev_target=ddim_sample(target_features.shape)
            ##计算loss
            #保存整个model

            val_meter.set_model_ema_enabled(False)
            eval_epoch(val_loader, model, val_meter, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, writer)
            if model_ema is not None:
                val_meter.set_model_ema_enabled(True)
                eval_epoch(val_loader, model_ema.module, val_meter, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, writer)

            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
            state_dict = torch.load(last_checkpoint, map_location="cpu")
            ms = model.module
            ms.head.Unet1D.load_state_dict(state_dict, strict=False)
            model.train()

            if cfg.TRAIN.BACKBONE_FROZEN:
                for name, p in model.named_parameters():
                    p.requires_grad_(False)
                    if "head.backbone" in name:
                        p.requires_grad_(False)
                        # if 'adapter' in name:
                        #     p.requires_grad_(True)
                        #     print(name)
                        if 'backbone.proj' in name:  # DO
                            p.requires_grad_(True)
                            print(name)
                        if 'backbone.attnpool' in name:  # DO
                            p.requires_grad_(True)
                        # if 'backbone.ml_ln_post' in name:
                        #     p.requires_grad_(True)
                        #     print(name)
                        # if 'backbone.ml_visu_proj' in name:
                        #     p.requires_grad_(True)
                        #     print(name)
                        if 'backbone.layer4.2' in name:  # DO
                            p.requires_grad_(True)
                        if 'backbone.transformer.resblocks.11' in name:  # DO
                            p.requires_grad_(True)
                        # if 'backbone.transformer.resblocks.9' in name: #DO
                        #     p.requires_grad_(True)
                        #     print(name)
                        if 'backbone.transformer.resblocks.7' in name:  # DO
                            p.requires_grad_(True)
                        if 'backbone.transformer.resblocks.3' in name:  # DO
                            p.requires_grad_(True)
                        if 'backbone.layer4.1' in name:  # DO
                            p.requires_grad_(True)
                        if 'backbone.ln_post' in name:  # DO
                            p.requires_grad_(True)
                    elif 'head.scale' in name:
                        p.requires_grad_(False)
                    # elif 'head.initial_backbone' in name:
                    #     p.requires_grad_(False)
                    elif 'head.dec' in name:
                        p.requires_grad_(True)
                    elif 'ctx' in name:
                        p.requires_grad_(False)
                        # print(name)
                    elif 'prompt' in name:
                        p.requires_grad_(False)
                    elif 'head.transformer' in name:
                        p.requires_grad_(False)
                    else:
                        p.requires_grad_(True)


        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)
            

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])


        # Update the learning rate.
        optimizer_rl = deepcopy(optimizer)
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        optim.set_lr(optimizer_rl, lr)


        with autocast():
            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])

            else:
                task_dict['Et_Dev_lastepo'] = Et_Dev_lastepo
                model_dict = model(task_dict)
                Et_Dev_lastepo = model_dict['Et_Dev_mean']

            target_logits = model_dict['logits']
            target_features_ori_frozen = model_dict['target_features_ori_frozen']
            target_features_ori = model_dict['target_features_ori']
            context_target = model_dict['context_target']
            cofficients = model_dict['cofficients']
            # alphas_prod= model_dict['cofficients'][0]
            # sqrt_recip_alphas_cumprod= model_dict['cofficients'][1]
            # sqrt_recipm1_alphas_cumprod= model_dict['cofficients'][2]
            # betas= model_dict['cofficients'][3]
            # one_minus_alphas_bar_sqrt = model_dict['cofficients'][4]
            target_features.append([target_features_ori_frozen, target_features_ori, context_target])

            if hasattr(cfg.TRAIN,"USE_CLASSIFICATION") and cfg.TRAIN.USE_CLASSIFICATION:
                if hasattr(cfg.TRAIN,"USE_CLASSIFICATION_ONLY") and cfg.TRAIN.USE_CLASSIFICATION_ONLY:
                    loss = cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()) /cfg.TRAIN.BATCH_SIZE
                elif hasattr(cfg.TRAIN,"USE_LOCAL") and cfg.TRAIN.USE_LOCAL:
                    if hasattr(cfg.TRAIN,"TEMPORAL_LOSS_WEIGHT") and cfg.TRAIN.TEMPORAL_LOSS_WEIGHT:
                        loss =  (cfg.TRAIN.TEMPORAL_LOSS_WEIGHT*model_dict["loss_temporal_regular"] + F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE
                    else:
                        loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE

                else:
                    # set_trace()
                    if hasattr(cfg.TRAIN,"USE_CONTRASTIVE") and cfg.TRAIN.USE_CONTRASTIVE:
                        if hasattr(cfg.TRAIN,"USE_MOTION") and cfg.TRAIN.USE_MOTION:
                            if hasattr(cfg.TRAIN,"MOTION_COFF") and cfg.TRAIN.MOTION_COFF:
                                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.MOTION_COFF*(F.cross_entropy(model_dict["logits_motion"], task_dict["target_labels"].long()))
                            else:
                                if hasattr(cfg.TRAIN,"USE_RECONS") and cfg.TRAIN.USE_RECONS:
                                    loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.RECONS_COFF*model_dict["loss_recons"]
                                else:
                                    loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
                        else:
                            if hasattr(cfg.TRAIN,"USE_RECONS") and cfg.TRAIN.USE_RECONS:
                                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.RECONS_COFF*model_dict["loss_recons"]
                            else:
                                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
                    else:
                        if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
                            # print('model_dict["class_logits"]', model_dict["class_logits"].shape) #[30, 64]
                            # print('model_dict["class_logits_ori"]', model_dict["class_logits_ori"].shape) #[240, 240]
                            # print('task_dict["real_support_labels"]', task_dict["real_support_labels"].shape) #[5]
                            # print('task_dict["real_target_labels"]', task_dict["real_target_labels"].shape) #[25]
                            # print('task_dict["target_labels"]', task_dict["target_labels"].shape)
                            # print(task_dict["target_labels"])  #[25]
                            # print('task_dict["support_labels"]', task_dict["support_labels"].shape)
                            # print(task_dict["support_labels"]) # [5]
                            hid_dim = model_dict["vis_feat"].shape[-1]
                            # loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + \
                            #          cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()) + \
                            #          # F.kl_div(model_dict["vis_feat"].reshape(-1,hid_dim).log_softmax(dim=-1), model_dict["vis_feat_ori"].reshape(-1,hid_dim).softmax(dim=-1), reduction='sum') +\
                            #          F.cross_entropy(model_dict["class_logits_ori"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1))) /cfg.TRAIN.BATCH_SIZE + (model_dict['noise'] - model_dict['pred_noise']).square().mean() + model_dict['gen_cosim_loss']  #5*(model_dict['noise'] - model_dict['pred_noise']).square().mean()
                            #          # 0.5*cfg.TRAIN.USE_CLASSIFICATION_VALUE  * F.cross_entropy(model_dict["class_logits_ori"]* (1-model_dict["class_logits_ori"]), torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1))) /cfg.TRAIN.BATCH_SIZE
                            #     #TODO
                            if cfg.TRAIN.Frozen_Cliploss:
                                loss1 = F.cross_entropy(model_dict["class_logits_ori"], torch.cat(
                                    [task_dict["real_support_labels"], task_dict["real_target_labels"]],
                                    0).long().unsqueeze(1).repeat(1, cfg.DATA.NUM_INPUT_FRAMES).reshape(
                                    -1)) / cfg.TRAIN.BATCH_SIZE

                            else:
                                loss1 = F.cross_entropy(model_dict["class_logits_ori"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1)) /cfg.TRAIN.BATCH_SIZE
                            # loss1 =   0.5*cfg.TRAIN.USE_CLASSIFICATION_VALUE  * F.cross_entropy(model_dict["class_logits_ori"]* (1-model_dict["class_logits_ori"]), torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1)) /cfg.TRAIN.BATCH_SIZE
                            loss2 = (#F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + \
                            cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()))/cfg.TRAIN.BATCH_SIZE
                            loss3 = model_dict['l1_loss'] + model_dict['direction_lossv1'] #F.cross_entropy(model_dict ['cos_mseloss'],  task_dict["target_labels"].long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1))/cfg.TRAIN.BATCH_SIZE
                            #model_dict['gen_cosim_loss'] #F.kl_div(model_dict["vis_feat"].reshape(-1,hid_dim).log_softmax(dim=-1), model_dict["vis_feat_ori"].reshape(-1,hid_dim).softmax(dim=-1), reduction='sum')/cfg.TRAIN.BATCH_SIZE
                            if hasattr(cfg.TRAIN, "Diffusion") and cfg.TRAIN.Diffusion:
                                loss4 = (model_dict['noise'] - model_dict['pred_noise']).square().mean()
                            else:
                                loss4 = 0

                            loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + \
                                     cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()) + \
                                     F.cross_entropy(model_dict["class_logits_ori"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long().unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1))) /cfg.TRAIN.BATCH_SIZE #+ 0.1*(model_dict['noise'] - model_dict['pred_noise']).square().mean()
                            if (cur_iter + 1) // cfg.TRAIN.VAL_FRE_ITER < 5:
                                loss = loss + (1-((cur_iter + 1) / cfg.TRAIN.VAL_FRE_ITER)*0.20)*(model_dict['noise'] - model_dict['pred_noise']).square().mean()


                        else:
                            loss = (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(
                                model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]],
                                          0).long())) / cfg.TRAIN.BATCH_SIZE
            else:
                loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE

        # print('loss', loss)
        ###check Nan Loss.
        if torch.isnan(loss).any():
            # # logger.info(f"logits: {model_dict}")
            # loss.backward(retain_graph=False)
            # optimizer.zero_grad()
            # continue
            ##scaler_loss_back
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            continue
        # if math.isnan(loss2):
        #     # # logger.info(f"logits: {model_dict}")
        #     # loss.backward(retain_graph=False)
        #     # optimizer.zero_grad()
        #     # continue
        #     ##scaler_loss_back
        #     optimizer.zero_grad()
        #     scaler.scale(loss2).backward()  # scaler.scale(loss).backward(retain_graph=False)
        #     continue

        #loss.backward(retain_graph=False)
        # optim.set_lr(optimizer, 100*lr)

        if cur_iter%100 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        optimizer.zero_grad()
        scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN gradient in {name}")
        # optimizer.zero_grad()


        # scaler.step(optimizer_rl)

        if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)

        # optimize
        # if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
            # optimizer.step()
            # optimizer.zero_grad()

        ######Loss2
        # optim.set_lr(optimizer, lr)
        # optimizer.zero_grad()
        # scaler.scale(loss2).backward(retain_graph=True)  # scaler.scale(loss).backward(retain_graph=False)

        scaler.step(optimizer)
        # self.scheduler.step()


        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        scaler.update()

        if cfg.DETECTION.ENABLE or cfg.PRETRAIN.ENABLE:
            if misc.get_num_gpus(cfg) > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )
            if cfg.PRETRAIN.ENABLE:
                train_meter.update_custom_stats(loss_in_parts)

        else:
            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                if misc.get_num_gpus(cfg) > 1:
                    loss = du.all_reduce([loss])[0].item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = du.all_reduce([v])[0].item()
                else:
                    loss = loss.item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = v.item()
                train_meter.update_custom_stats(loss_in_parts)
                train_meter.update_custom_stats(top1_err_all)
                train_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
                        loss, loss1, loss2, loss3, loss4, top1_err, top5_err = du.all_reduce(
                            [loss, loss1, loss2, loss3, loss4, top1_err, top5_err]
                        )
                    else:
                        loss, top1_err, top5_err = du.all_reduce(
                            [loss, top1_err, top5_err]
                        )

                # Copy the stats from GPU to CPU (sync point).
                if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
                    loss, loss1, loss2, loss3, loss4, top1_err, top5_err = (
                        loss.item(),
                        loss1.item(),
                        loss2.item(),
                        loss3.item(),
                        loss4.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )
                else:
                    loss, top1_err, top5_err = (
                        loss.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )

            train_meter.iter_toc()
            # Update and log stats.

            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                train_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )

            # write to tensorboard format if available.
            if writer is not None:
                if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/decoder_loss": loss1,
                            "Train/fsar_loss": loss2,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
                else:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    # nomeaning_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer=None, model_bucket=None)
    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    state_dict = torch.load(last_checkpoint, map_location="cpu")
    ms = model.module
    ms.head.Unet1D.load_state_dict(state_dict, strict=False)


    model.eval()
    val_meter.iter_tic()

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        if misc.get_num_gpus(cfg):

            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if misc.get_num_gpus(cfg) > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'PCMGenerator'):
            preds, logits = model(inputs)
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        else:
            # preds, logits = model(inputs)
            task_dict['Et_Dev_lastepo'] = None
            model_dict = model(task_dict)

            # loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)
            target_logits = model_dict['logits']
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE


            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                labels = task_dict['target_labels']
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                val_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if misc.get_num_gpus(cfg):
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()

def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer, model_bucket)


    # #####load Unet1D
    if os.path.exists('/mnt/hdd/fpeng/CLIP-FSAR/Unet1D_gpu4.pth') and not cu.has_checkpoint(cfg.OUTPUT_DIR):
        model_path = '/mnt/hdd/fpeng/CLIP-FSAR/Unet1D_gpu4.pth'
        state_dict = torch.load(model_path, map_location="cpu")
        ms = model.module
        ms.head.Unet1D.load_state_dict(state_dict, strict=False)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None  # val

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        pass
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, cur_epoch)
    
    train_epoch(
        train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer, val_meter, val_loader
    )
    # train_DiT_epoch_merely()
    # torch.cuda.empty_cache()
    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )


def train_DiT_epoch(
        target_features, cofficients, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, cur_epoch_save, cur_iter, writer=None, val_meter=None,
        val_loader=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # 先model.eval, eval一下，获得和target_features，把数据对保存下来,concat[target_features.detach().clone(),support_features.detach().clone()]
    # model.eval()
    # target_features = []
    # cofficients = []
    # for cur_i, task_dict in enumerate(train_loader):
    #     if cur_i >= cur_iter:
    #             break
    #     print('cur_iter=', cur_iter)
    #     print('cur_i=', cur_i)
    #     # if cur_i <= cur_iter and cur_i >=((cur_iter+1)-50):#(cur_iter+1)-cfg.TRAIN.VAL_FRE_ITER <= cur_i <= cur_iter:
    #     if misc.get_num_gpus(cfg):
    #         for k in task_dict.keys():
    #             task_dict[k] = task_dict[k][0].cuda(non_blocking=True)
    #     with torch.no_grad():
    #         task_dict['Et_Dev_lastepo'] = None
    #         model_dict = model(task_dict)
    #         target_features_ori_frozen = model_dict['target_features_ori_frozen']
    #         target_features_ori = model_dict['target_features_ori']
    #         context_target = model_dict['context_target']
    #         cofficients = model_dict['cofficients']
    #         # alphas_prod= model_dict['cofficients'][0]
    #         # sqrt_recip_alphas_cumprod= model_dict['cofficients'][1]
    #         # sqrt_recipm1_alphas_cumprod= model_dict['cofficients'][2]
    #         # betas= model_dict['cofficients'][3]
    #         # one_minus_alphas_bar_sqrt = model_dict['cofficients'][4]
    #         target_features.append([target_features_ori_frozen, target_features_ori, context_target])
    #
    # # target_features=torch.stack(target_features)
    # torch.save(target_features, './DiT_training.pth')

    ###创建新的DiTtrain_loader
    print("Start DiT finetuning")

    ####trainDiT
    scaler = GradScaler()
    if cfg.TRAIN.BACKBONE_FROZEN:
        for name, p in model.named_parameters():
            p.requires_grad_(False)
            if 'DiT' in name or 'Unet1D' in name:
                p.requires_grad_(True)

    # torch.autograd.set_detect_anomaly(True)
    # Enable train mode.
    model.train()
    norm_train = False

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    # train_meter.iter_tic()

    data_size = cfg.SOLVER.STEPS_ITER
    Et_Dev_lastepo = None
    # for cur_iter, task_dict in enumerate(train_loader):
    for cur_i, (target_features_ori_frozen, target_features_ori, context_target) in enumerate(target_features):
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', "real_support_labels"]'''
        # Save a checkpoint.
        cur_epoch = cur_iter // cfg.SOLVER.STEPS_ITER
        # if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0 and cur_iter>=200:   #
        if cur_i == len(target_features)-1:  #
            if cfg.OSS.ENABLE:
                model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            else:
                model_bucket = None
            cur_epoch_save = cur_iter // cfg.TRAIN.VAL_FRE_ITER

            ##model.train(),把grad_require重置一下
            ####然后Dev_target=ddim_sample(target_features.shape)
            ##计算loss
            # 保存整个model
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, model_ema, optimizer, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                               cfg, model_bucket)

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])

        # Update the learning rate.
        optimizer_rl = deepcopy(optimizer)
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        optim.set_lr(optimizer_rl, lr)

        with autocast():
            # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #     base_model = model.module
            # else:
            #     base_model = model
            #
            # # 然后尝试获取子模块 DiT
            # DiT_module = getattr(base_model, 'DiT', None)
            # scale = getattr(base_model, 'scale', None)
            #
            # print("\nModel Modules:")
            # for name, module in base_model.named_modules():
            #     print(f"{name}: {module}")
            #
            # # 检查 DiT_module 是否存在
            # if DiT_module is None:
            #     raise AttributeError(f"'{type(base_model).__name__}' object has no attribute 'DiT'")

            # 使用 DiT_module 进行操作
            if hasattr(cfg.TRAIN, "DiffusionTransformer") and cfg.TRAIN.DiffusionTransformer:
                De_v_target = model.module.head.DiT.ddim_sample(target_features_ori.shape, y_cond=target_features_ori, alphas_prod=cofficients[0],
                                                                sqrt_recip_alphas_cumprod=cofficients[1], sqrt_recipm1_alphas_cumprod=cofficients[2],
                                                                betas=cofficients[3], one_minus_alphas_bar_sqrt=cofficients[4])

            else:
                target_bs = target_features_ori.shape[0]
                De_v_target = model.module.head.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)

                # print("model_output De_v_target NaN:", torch.isnan(De_v_target).any())
            dim = target_features_ori.shape[-1]

            # *(B*T,25)
            ####
            target_features_ori_frozen_mean = target_features_ori_frozen / (target_features_ori_frozen.norm(dim=-1,
                                                                                                           keepdim=True)+ 1e-7)
            Et_Dev_target_mean = De_v_target / (De_v_target.norm(dim=-1, keepdim=True)+ 1e-7)
            context_target_mean = context_target /(context_target.norm(dim=-1, keepdim=True)+ 1e-7)
            logits_per_image = model.module.head.scale * target_features_ori_frozen_mean.reshape(-1,
                                                                                    dim) @ Et_Dev_target_mean.reshape(
                -1, dim).t()
            global_cliploss = -torch.log((2 - (1 - logits_per_image / 100).mean()) / 2)
            ###
            edit_direction = (Et_Dev_target_mean - target_features_ori_frozen_mean)
            edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
            target_direction = (context_target_mean - target_features_ori_frozen_mean)
            target_direction /= (target_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
            cos_func = torch.nn.CosineSimilarity()
            direction_lossv1 = (-torch.log((2. - (1. - cos_func(edit_direction, target_direction))) / 2)).mean()

            ###
            logits_per_image_src = model.module.head.scale * target_features_ori_frozen_mean.reshape(-1,
                                                                                        dim) @ context_target_mean.reshape(
                -1, dim).t()
            l1_loss_fn = nn.L1Loss()
            direction_lossv2 = l1_loss_fn(logits_per_image_src, logits_per_image).mean()
            print('direction_lossv1', direction_lossv1)
            ###
            l1_loss = l1_loss_fn(context_target_mean, Et_Dev_target_mean).mean()

            loss = direction_lossv1 #+ l1_loss
            # loss = global_cliploss# + l1_loss
            # loss = direction_lossv2 + l1_loss

        # print('loss', loss)
        ###check Nan Loss.
        if torch.isnan(loss).any():
            # # logger.info(f"logits: {model_dict}")
            # loss.backward(retain_graph=False)
            # optimizer.zero_grad()
            # continue
            ##scaler_loss_back
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)  # 将所有无穷大的梯度设为1000
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            continue



        # loss.backward(retain_graph=False)
        # optim.set_lr(optimizer, 100*lr)

        if cur_i % 100 == 0:
            print('loss', loss)
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        optimizer.zero_grad()
        scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN gradient in {name}")

        # optimizer.zero_grad()

        # scaler.step(optimizer_rl)

        if hasattr(cfg.TRAIN, "CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)

        # optimize
        # if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
        # optimizer.step()
        # optimizer.zero_grad()

        ######Loss2
        # optim.set_lr(optimizer, lr)
        # optimizer.zero_grad()
        # scaler.scale(loss2).backward(retain_graph=True)  # scaler.scale(loss).backward(retain_graph=False)

        scaler.step(optimizer)
        # self.scheduler.step()

        if hasattr(cfg, "MULTI_MODAL") and \
                cfg.PRETRAIN.PROTOTYPE.ENABLE and \
                cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        scaler.update()

        # # Gather all the predictions across all the devices.
        # if misc.get_num_gpus(cfg) > 1:
        #     if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #         loss = du.all_reduce(
        #             [loss]
        #         )
        #     else:
        #         loss = du.all_reduce(
        #             [loss]
        #         )
        #
        # # Copy the stats from GPU to CPU (sync point).
        # if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #     loss = (
        #         loss.item(),
        #     )
        # else:
        #     loss = (
        #         loss.item(),
        #     )
        #
        # train_meter.iter_toc()
        # # Update and log stats.
        #
        # train_meter.update_stats(
        #     loss,
        #     lr,
        #     train_loader.batch_size
        #     * max(
        #         misc.get_num_gpus(cfg), 1
        #     ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        # )
        #
        # # write to tensorboard format if available.
        # if writer is not None:
        #     if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #         writer.add_scalars(
        #             {
        #                 "Train/loss": loss,
        #                 "Train/lr": lr,
        #             },
        #             global_step=data_size * cur_epoch + cur_iter,
        #         )
        #     else:
        #         writer.add_scalars(
        #             {
        #                 "Train/loss": loss,
        #                 "Train/lr": lr,
        #             },
        #             global_step=data_size * cur_epoch + cur_iter,
        #         )
        #
        # train_meter.log_iter_stats(cur_epoch, cur_iter)
        # train_meter.iter_tic()

    # Log epoch stats.
    # train_meter.log_epoch_stats(cur_epoch + cfg.TRAIN.NUM_FOLDS - 1)


def train_DiT_epoch_merely(
        target_features, cofficients, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, cur_epoch_save, cur_iter, writer=None, val_meter=None,
        val_loader=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # 先model.eval, eval一下，获得和target_features，把数据对保存下来,concat[target_features.detach().clone(),support_features.detach().clone()]
    # model.eval()
    # target_features = []
    # cofficients = []
    # for cur_i, task_dict in enumerate(train_loader):
    #     if cur_i >= cur_iter:
    #             break
    #     print('cur_iter=', cur_iter)
    #     print('cur_i=', cur_i)
    #     # if cur_i <= cur_iter and cur_i >=((cur_iter+1)-50):#(cur_iter+1)-cfg.TRAIN.VAL_FRE_ITER <= cur_i <= cur_iter:
    #     if misc.get_num_gpus(cfg):
    #         for k in task_dict.keys():
    #             task_dict[k] = task_dict[k][0].cuda(non_blocking=True)
    #     with torch.no_grad():
    #         task_dict['Et_Dev_lastepo'] = None
    #         model_dict = model(task_dict)
    #         target_features_ori_frozen = model_dict['target_features_ori_frozen']
    #         target_features_ori = model_dict['target_features_ori']
    #         context_target = model_dict['context_target']
    #         cofficients = model_dict['cofficients']
    #         # alphas_prod= model_dict['cofficients'][0]
    #         # sqrt_recip_alphas_cumprod= model_dict['cofficients'][1]
    #         # sqrt_recipm1_alphas_cumprod= model_dict['cofficients'][2]
    #         # betas= model_dict['cofficients'][3]
    #         # one_minus_alphas_bar_sqrt = model_dict['cofficients'][4]
    #         target_features.append([target_features_ori_frozen, target_features_ori, context_target])
    #
    # # target_features=torch.stack(target_features)
    # torch.save(target_features, './DiT_training.pth')

    ###创建新的DiTtrain_loader
    print("Start DiT finetuning")

    ####trainDiT
    scaler = GradScaler()
    if cfg.TRAIN.BACKBONE_FROZEN:
        for name, p in model.named_parameters():
            p.requires_grad_(False)
            if 'DiT' in name or 'Unet1D' in name:
                p.requires_grad_(True)

    # torch.autograd.set_detect_anomaly(True)
    # Enable train mode.
    model.train()
    norm_train = False

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    # train_meter.iter_tic()

    data_size = cfg.SOLVER.STEPS_ITER
    Et_Dev_lastepo = None
    # for cur_iter, task_dict in enumerate(train_loader):
    for cur_i, (target_features_ori_frozen, target_features_ori, context_target) in enumerate(target_features):
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', "real_support_labels"]'''
        # Save a checkpoint.
        cur_epoch = cur_iter // cfg.SOLVER.STEPS_ITER
        # if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0 and cur_iter>=200:   #
        if cur_i == len(target_features)-1:  #
            if cfg.OSS.ENABLE:
                model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            else:
                model_bucket = None
            cur_epoch_save = cur_iter // cfg.TRAIN.VAL_FRE_ITER

            ##model.train(),把grad_require重置一下
            ####然后Dev_target=ddim_sample(target_features.shape)
            ##计算loss
            # 保存整个model
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, model_ema, optimizer, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                               cfg, model_bucket)

            ####val
            val_meter.set_model_ema_enabled(False)
            eval_epoch(val_loader, model, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, writer)
            if model_ema is not None:
                val_meter.set_model_ema_enabled(True)
                eval_epoch(val_loader, model_ema.module, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg,
                           writer)

            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
            state_dict = torch.load(last_checkpoint, map_location="cpu")
            ms = model.module
            ms.head.Unet1D.load_state_dict(state_dict, strict=False)
            model.train()

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])

        # Update the learning rate.
        optimizer_rl = deepcopy(optimizer)
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        optim.set_lr(optimizer_rl, lr)

        with autocast():
            # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #     base_model = model.module
            # else:
            #     base_model = model
            #
            # # 然后尝试获取子模块 DiT
            # DiT_module = getattr(base_model, 'DiT', None)
            # scale = getattr(base_model, 'scale', None)
            #
            # print("\nModel Modules:")
            # for name, module in base_model.named_modules():
            #     print(f"{name}: {module}")
            #
            # # 检查 DiT_module 是否存在
            # if DiT_module is None:
            #     raise AttributeError(f"'{type(base_model).__name__}' object has no attribute 'DiT'")

            # 使用 DiT_module 进行操作
            if hasattr(cfg.TRAIN, "DiffusionTransformer") and cfg.TRAIN.DiffusionTransformer:
                De_v_target = model.module.head.DiT.ddim_sample(target_features_ori.shape, y_cond=target_features_ori, alphas_prod=cofficients[0],
                                                                sqrt_recip_alphas_cumprod=cofficients[1], sqrt_recipm1_alphas_cumprod=cofficients[2],
                                                                betas=cofficients[3], one_minus_alphas_bar_sqrt=cofficients[4])

            else:
                target_bs = target_features_ori.shape[0]
                De_v_target = model.module.head.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)

                # print("model_output De_v_target NaN:", torch.isnan(De_v_target).any())
            dim = target_features_ori.shape[-1]

            # *(B*T,25)
            ####
            target_features_ori_frozen_mean = target_features_ori_frozen / (target_features_ori_frozen.norm(dim=-1,
                                                                                                           keepdim=True)+ 1e-7)
            Et_Dev_target_mean = De_v_target / (De_v_target.norm(dim=-1, keepdim=True)+ 1e-7)
            context_target_mean = context_target /(context_target.norm(dim=-1, keepdim=True)+ 1e-7)
            logits_per_image = model.module.head.scale * target_features_ori_frozen_mean.reshape(-1,
                                                                                    dim) @ Et_Dev_target_mean.reshape(
                -1, dim).t()
            global_cliploss = -torch.log((2 - (1 - logits_per_image / 100).mean()) / 2)
            ###
            edit_direction = (Et_Dev_target_mean - target_features_ori_frozen_mean)
            edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
            target_direction = (context_target_mean - target_features_ori_frozen_mean)
            target_direction /= (target_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
            cos_func = torch.nn.CosineSimilarity()
            direction_lossv1 = (-torch.log((2. - (1. - cos_func(edit_direction, target_direction))) / 2)).mean()

            ###
            logits_per_image_src = model.module.head.scale * target_features_ori_frozen_mean.reshape(-1,
                                                                                        dim) @ context_target_mean.reshape(
                -1, dim).t()
            l1_loss_fn = nn.L1Loss()
            direction_lossv2 = l1_loss_fn(logits_per_image_src, logits_per_image).mean()
            print('direction_lossv2', direction_lossv2)
            ###
            l1_loss = l1_loss_fn(context_target_mean, Et_Dev_target_mean).mean()

            # loss = direction_lossv1 + l1_loss
            # loss = global_cliploss# + l1_loss
            loss = direction_lossv2 + l1_loss

        # print('loss', loss)
        ###check Nan Loss.
        if torch.isnan(loss).any():
            # # logger.info(f"logits: {model_dict}")
            # loss.backward(retain_graph=False)
            # optimizer.zero_grad()
            # continue
            ##scaler_loss_back
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)  # 将所有无穷大的梯度设为1000
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            continue



        # loss.backward(retain_graph=False)
        # optim.set_lr(optimizer, 100*lr)

        if cur_i % 100 == 0:
            print('loss', loss)
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        optimizer.zero_grad()
        scaler.scale(loss).backward(retain_graph=False)  # scaler.scale(loss).backward(retain_graph=False)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1000.0)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN gradient in {name}")

        # optimizer.zero_grad()

        # scaler.step(optimizer_rl)

        if hasattr(cfg.TRAIN, "CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)

        # optimize
        # if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
        # optimizer.step()
        # optimizer.zero_grad()

        ######Loss2
        # optim.set_lr(optimizer, lr)
        # optimizer.zero_grad()
        # scaler.scale(loss2).backward(retain_graph=True)  # scaler.scale(loss).backward(retain_graph=False)

        scaler.step(optimizer)
        # self.scheduler.step()

        if hasattr(cfg, "MULTI_MODAL") and \
                cfg.PRETRAIN.PROTOTYPE.ENABLE and \
                cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        scaler.update()

        # # Gather all the predictions across all the devices.
        # if misc.get_num_gpus(cfg) > 1:
        #     if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #         loss = du.all_reduce(
        #             [loss]
        #         )
        #     else:
        #         loss = du.all_reduce(
        #             [loss]
        #         )
        #
        # # Copy the stats from GPU to CPU (sync point).
        # if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #     loss = (
        #         loss.item(),
        #     )
        # else:
        #     loss = (
        #         loss.item(),
        #     )
        #
        # train_meter.iter_toc()
        # # Update and log stats.
        #
        # train_meter.update_stats(
        #     loss,
        #     lr,
        #     train_loader.batch_size
        #     * max(
        #         misc.get_num_gpus(cfg), 1
        #     ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        # )
        #
        # # write to tensorboard format if available.
        # if writer is not None:
        #     if hasattr(cfg.TRAIN, "Decoder") and cfg.TRAIN.Decoder:
        #         writer.add_scalars(
        #             {
        #                 "Train/loss": loss,
        #                 "Train/lr": lr,
        #             },
        #             global_step=data_size * cur_epoch + cur_iter,
        #         )
        #     else:
        #         writer.add_scalars(
        #             {
        #                 "Train/loss": loss,
        #                 "Train/lr": lr,
        #             },
        #             global_step=data_size * cur_epoch + cur_iter,
        #         )
        #
        # train_meter.log_iter_stats(cur_epoch, cur_iter)
        # train_meter.iter_tic()
