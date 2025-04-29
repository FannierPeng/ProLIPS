#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Entry file for training, evaluating and testing a video model."""

import os
import sys
import time
sys.path.append(os.path.abspath(os.curdir))

from utils.launcher import launch_task

from test import test
from train import train
from train_net_few_shot import train_few_shot
from test_net_few_shot import test_few_shot
from test_epic_localization import test_epic_localization
from submission_test import submission_test
from utils.config import Config


def _prepare_data(cfg):
    if cfg.TASK_TYPE in ['classification']:
        train_func = train
        test_func = test
    elif cfg.TASK_TYPE in ['localization']:
        train_func = train
        test_func = test_epic_localization
    elif cfg.TASK_TYPE in ['few_shot_action']:
        train_func = train_few_shot
        test_func = test_few_shot
    elif cfg.TASK_TYPE in ["submission"]:
        cfg.TRAIN.ENABLE = False
        cfg.TEST.ENABLE = False
        train_func = None
        test_func = None
        submission_func = submission_test
    else:
        raise ValueError("unknown TASK_TYPE {}".format(cfg.TASK_TYPE))
    
    run_list = []
    if cfg.TRAIN.ENABLE:
        # Training process is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), train_func])
    
    if cfg.TEST.ENABLE:
        # Test is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), test_func])
        if cfg.TEST.AUTOMATIC_MULTI_SCALE_TEST:
            """
                By default, test_func performs single view test. 
                AUTOMATIC_MULTI_SCALE_TEST automatically performs multi-view test after the single view test.
            """
            cfg.LOG_MODEL_INFO = False
            cfg.LOG_CONFIG_INFO = False

            cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
            cfg.TEST.NUM_SPATIAL_CROPS = 1

            if "kinetics" in cfg.TEST.DATASET or "epickitchen" in cfg.TEST.DATASET:
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            if "imagenet" in cfg.TEST.DATASET and not cfg.PRETRAIN.ENABLE:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            if "ssv2" in cfg.TEST.DATASET:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            cfg.TEST.LOG_FILE = "val_{}clipsx{}crops.log".format(
                cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
            )
            run_list.append([cfg.deep_copy(), test_func])

    if cfg.SUBMISSION.ENABLE:
        # currently only supports epic kitchen submission
        cfg.LOG_MODEL_INFO = False
        cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
        cfg.TEST.NUM_SPATIAL_CROPS = 3

        cfg.TEST.LOG_FILE = "test_{}clipsx{}crops.log".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        )
        run_list.append([cfg.deep_copy(), submission_func])
  
    return run_list

def main():
    """
    Entry function for spawning all the function processes. 
    """
    cfg = Config(load=True)
    # print(cfg.DATA)
    # get the list of configs and functions for running

    if not cfg.TEST.ENABLE and cfg.TRAIN.ENABLE:
        run_list = _prepare_data(cfg)
        for run in run_list:
            launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])

    if cfg.TEST.CHECKPOINT_FILE_PATH == "":
        # plot loss fig
        PATH_TO_LOG = '/mnt/hdd/fpeng/CLIP-FSAR/' + cfg.OUTPUT_DIR + '/training_log.log'
        import logging
        import numpy as np
        import matplotlib.pyplot as plt
        logger = logging.getLogger()
        handler = logging.FileHandler(PATH_TO_LOG)
        logger.addHandler(handler)
        with open(PATH_TO_LOG, 'r') as file:
            lines = file.readlines()
        min_top1_errs = []
        loss = []
        for line in lines:
            if 'min_top1_err' in line:
                min_top1_errs.append(float(line.split('"top1_err": ')[1][0:8]))
            if '"loss"' in line:
                loss.append(float(line.split('"loss": ')[1][0:8]))

        fig = plt.figure(figsize=(7, 8))
        fig.add_subplot(2, 1, 1)
        plt.plot(list(range(1, len(min_top1_errs) + 1)), min_top1_errs)
        plt.ylabel('min_top1_err')
        plt.title(PATH_TO_LOG.split('/')[6])

        fig.add_subplot(2, 1, 2)
        plt.plot(list(range(1, len(loss) + 1)), loss)
        plt.ylabel('loss')

        if not os.path.isfile('/mnt/hdd/fpeng/CLIP-FSAR/output/' + PATH_TO_LOG.split('/')[6] + '/' + PATH_TO_LOG.split('/')[6] + '.jpg'):
            plt.savefig(
                '/mnt/hdd/fpeng/CLIP-FSAR/output/' + PATH_TO_LOG.split('/')[6] + '/' + PATH_TO_LOG.split('/')[6] + '.jpg')

        print("Finish running with config: {}".format(cfg.args.cfg_file))
        index = min_top1_errs.index(min(min_top1_errs)) + 1
        for file in os.listdir('/mnt/hdd/fpeng/CLIP-FSAR/output/'+PATH_TO_LOG.split('/')[6]+'/checkpoints'):
            if file != "checkpoint_epoch_{:05d}.pyth".format(index) and os.path.isfile('/mnt/hdd/fpeng/CLIP-FSAR/output/'+PATH_TO_LOG.split('/')[6]+'/checkpoints/'+file):
                os.remove('/mnt/hdd/fpeng/CLIP-FSAR/output/'+PATH_TO_LOG.split('/')[6]+'/checkpoints/' + file)

    if cfg.TEST.ENABLE and not cfg.TRAIN.ENABLE:
        cfg.TRAIN.NUM_TEST_TASKS = 10000
        run_list = _prepare_data(cfg)
        for run in run_list:
            launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])

    if cfg.TEST.ENABLE_WITH:
        ###TEST
        cfg.TRAIN.ENABLE = False
        cfg.TRAIN.NUM_TEST_TASKS = 10000
        cfg.TEST.ENABLE = True
        run_list = _prepare_data(cfg)

        for run in run_list:
            launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])


if __name__ == "__main__":
    main()

