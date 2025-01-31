# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# pretrain weights path
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_result'

NET_NAME = 'resnet_v1_101'
DATASET_NAME = 'user' # 'layer' #
VERSION = 'v1_{}'.format(DATASET_NAME)
CLASS_NUM = 2  # exclude background
BASE_ANCHOR_SIZE_LIST = [15, 25, 40, 60, 80]
LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]
STRIDE = [4, 8, 16, 32, 64]
# ANCHOR_SCALES = [1., 5., 10.]
ANCHOR_SCALES = [2., 3., 4.]
# ANCHOR_RATIOS = [0.05, 0.1, 0.2]
ANCHOR_RATIOS = [2., 3., 4., 5.]
SCALE_FACTORS = [10., 10., 5., 5.]
OUTPUT_STRIDE = 16
SHORT_SIDE_LEN = 640

BATCH_SIZE = 1
WEIGHT_DECAY = {'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 2 # 50000
LR = 0.0001

# rpn
SHARE_HEAD = True
RPN_NMS_IOU_THRESHOLD = 0.6
MAX_PROPOSAL_NUM = 2000
RPN_IOU_POSITIVE_THRESHOLD = 0.5
RPN_IOU_NEGATIVE_THRESHOLD = 0.2
RPN_MINIBATCH_SIZE = 512
RPN_POSITIVE_RATE = 0.5
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 12000

# fast rcnn
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.2
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FINAL_SCORE_THRESHOLD = 0.5
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.45
FAST_RCNN_MINIBATCH_SIZE = 256
FAST_RCNN_POSITIVE_RATE = 0.5
