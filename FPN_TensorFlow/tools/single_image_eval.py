#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from FPN_TensorFlow.data.io.read_tfrecord import next_batch, next_batch_for_tasks
from FPN_TensorFlow.data.io.convert_to_tfrecord import convert_pascal_to_tfrecord_from_list
from FPN_TensorFlow.help_utils.tools import *
from FPN_TensorFlow.libs.fast_rcnn import build_fast_rcnn
from FPN_TensorFlow.libs.label_name_dict.label_dict import *
from FPN_TensorFlow.libs.networks.network_factory import get_flags_byname, get_network_byname
from FPN_TensorFlow.libs.rpn import build_rpn
from FPN_TensorFlow.tools import restore_model

FLAGS = get_flags_byname(cfgs.NET_NAME)

def eval_dict_convert(samples, args):
  with tf.Graph().as_default():

    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, data_num = \
        next_batch_for_tasks(
            samples,
            dataset_name=cfgs.DATASET_NAME,
            batch_size=cfgs.BATCH_SIZE,
            shortside_len=cfgs.SHORT_SIDE_LEN,
            is_training=True,
            save_name="test"
        )

    # ***********************************************************************************************
    # *                                         share net                                           *
    # ***********************************************************************************************
    _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                      inputs=img_batch,
                                      num_classes=None,
                                      is_training=True,
                                      output_stride=None,
                                      global_pool=False,
                                      spatial_squeeze=False)

    # ***********************************************************************************************
    # *                                            RPN                                              *
    # ***********************************************************************************************
    rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                        inputs=img_batch,
                        gtboxes_and_label=None,
                        is_training=False,
                        share_head=True,
                        share_net=share_net,
                        stride=cfgs.STRIDE,
                        anchor_ratios=cfgs.ANCHOR_RATIOS,
                        anchor_scales=cfgs.ANCHOR_SCALES,
                        scale_factors=cfgs.SCALE_FACTORS,
                        base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                        level=cfgs.LEVEL,
                        top_k_nms=cfgs.RPN_TOP_K_NMS,
                        rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                        max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                        rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                        rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                        rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                        rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                        remove_outside_anchors=False,  # whether remove anchors outside
                        rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

    # rpn predict proposals
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

    # ***********************************************************************************************
    # *                                         Fast RCNN                                           *
    # ***********************************************************************************************
    fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                         feature_pyramid=rpn.feature_pyramid,
                                         rpn_proposals_boxes=rpn_proposals_boxes,
                                         rpn_proposals_scores=rpn_proposals_scores,
                                         img_shape=tf.shape(img_batch),
                                         roi_size=cfgs.ROI_SIZE,
                                         scale_factors=cfgs.SCALE_FACTORS,
                                         roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                         gtboxes_and_label=None,
                                         fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                         fast_rcnn_maximum_boxes_per_img=100,
                                         fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                         show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,  # show detections which score >= 0.6
                                         num_classes=cfgs.CLASS_NUM,
                                         fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                         fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                         fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                         use_dropout=False,
                                         weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                         is_training=False,
                                         level=cfgs.LEVEL)

    fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
        fast_rcnn.fast_rcnn_predict()

    # train
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = restore_model.get_restorer(checkpoint_path=args.weights)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      if not restorer is None:
        restorer.restore(sess, restore_ckpt)
        print('restore model')

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess, coord)

      gtbox_dict = {}
      predict_dict = {}

      for i in range(data_num):
        start = time.time()

        _img_name_batch, _img_batch, _gtboxes_and_label_batch, _fast_rcnn_decode_boxes, \
            _fast_rcnn_score, _detection_category \
            = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, fast_rcnn_decode_boxes,
                        fast_rcnn_score, detection_category])
        end = time.time()

        # gtboxes convert dict
        gtbox_dict[str(_img_name_batch[0])] = []
        predict_dict[str(_img_name_batch[0])] = []

        for j, box in enumerate(_gtboxes_and_label_batch[0]):
          bbox_dict = {}
          bbox_dict['bbox'] = np.array(_gtboxes_and_label_batch[0][j, :-1], np.float64)
          bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label_batch[0][j, -1])]
          gtbox_dict[str(_img_name_batch[0])].append(bbox_dict)

        for label in NAME_LABEL_MAP.keys():
          if label == 'back_ground':
            continue
          else:
            temp_dict = {}
            temp_dict['name'] = label

            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
            temp_boxes = _fast_rcnn_decode_boxes[ind]
            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
            temp_dict['bbox'] = np.array(np.concatenate(
                [temp_boxes, temp_score], axis=1), np.float64)
            predict_dict[str(_img_name_batch[0])].append(temp_dict)

        view_bar('{} image cost {}s'.format(
            str(_img_name_batch[0]), (end - start)), i + 1, data_num)

      fw1 = open('./gtboxes_dict.pkl', 'wb')
      fw2 = open('./predict_dict.pkl', 'wb')
      pickle.dump(gtbox_dict, fw1)
      pickle.dump(predict_dict, fw2)
      fw1.close()
      fw2.close()
      coord.request_stop()
      coord.join(threads)


def voc_ap(rec, prec, use_07_metric=False):
  """
  average precision calculations
  [precision integrated to recall]
  :param rec: recall
  :param prec: precision
  :param use_07_metric: 2007 metric is 11-recall-point based AP
  :return: average precision
  """
  if use_07_metric:
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap += p / 11.
  else:
    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def get_single_label_dict(predict_dict, gtboxes_dict, label):
  rboxes = {}
  gboxes = {}
  rbox_images = list(predict_dict.keys())

  for i in range(len(rbox_images)):
    rbox_image = rbox_images[i]
    for pre_box in predict_dict[rbox_image]:
      # if pre_box['name'] == label and len(pre_box['bbox']) != 0:
      if pre_box['name'] == label:
        rboxes[rbox_image] = [pre_box]

        gboxes[rbox_image] = []

        for gt_box in gtboxes_dict[rbox_image]:
          if gt_box['name'] == label:
            gboxes[rbox_image].append(gt_box)
  return rboxes, gboxes


def eval(rboxes, gboxes, iou_th, use_07_metric):
  rbox_images = list(rboxes.keys())
  fp = np.zeros(len(rbox_images))
  tp = np.zeros(len(rbox_images))
  gt_num_dict = {}
  box_num = 0

  for i in range(len(rbox_images)):
    rbox_image = rbox_images[i]
    if len(rboxes[rbox_image][0]['bbox']) > 0:

      rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
      if len(gboxes[rbox_image]) > 0:
        gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
        box_num = box_num + len(gbox_list)
        gt_num_dict[rbox_image] = len(gbox_list)
        gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
        confidence = rbox_lists[:, 4]
        box_index = np.argsort(-confidence)

        rbox_lists = rbox_lists[box_index, :]
        for rbox_list in rbox_lists:

          ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
          iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
          ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
          iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
          iw = np.maximum(ixmax - ixmin + 1., 0.)
          ih = np.maximum(iymax - iymin + 1., 0.)
          inters = iw * ih

          # union
          uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                 (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                 (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)
          overlaps = inters / uni
          ovmax = np.max(overlaps)
          jmax = np.argmax(overlaps)
          if ovmax > iou_th:
            if gbox_list[jmax, -1] == 0:
              tp[i] += 1
              gbox_list[jmax, -1] = 1
            else:
              fp[i] += 1
          else:
            fp[i] += 1

      else:
        gt_num_dict[rbox_image] = 0
        fp[i] += len(rboxes[rbox_image][0]['bbox'])
    else:
      # if an image has no predicted boxes
      tp[i] = 0
      fp[i] = 0
      gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
      gt_num_dict[rbox_image] = len(gbox_list)

      continue

  rec = np.zeros(len(rbox_images))
  prec = np.zeros(len(rbox_images))

  if box_num == 0:
    for i in range(len(fp)):
      if fp[i] != 0:
        prec[i] = 0
      else:
        prec[i] = 1
  else:
    fp_sum = np.cumsum(fp)
    tp_sum = np.cumsum(tp)

    prec = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)
    rec = tp_sum / box_num

  ap = voc_ap(rec, prec, use_07_metric)

  tp_dict = {}
  fp_dict = {}
  for i in range(len(rbox_images)):
      tp_dict[rbox_images[i]] = tp[i]
      fp_dict[rbox_images[i]] = fp[i]

  return rec, prec, ap, box_num, tp_dict, fp_dict, gt_num_dict

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Evaluate a trained FPN model')
  parser.add_argument('--weights', dest='weights',
                      help='model path',
                      default='{}/output/layer/res101_trained_weights/v1_layer/layer_49999model.ckpt'.format(cfgs.ROOT_PATH),
                      type=str)

  args = parser.parse_args()
  return args

def output_tasks(f1_score, img_dir, xml_dir, def_thresh=0.8):

    known_task_x = [os.path.join(img_dir, k.split("'")[1]) for k, v in f1_score.items() if v["f1_score"] >= def_thresh]

    unknown_task_x = [os.path.join(img_dir, k.split("'")[1]) for k, v in f1_score.items() if v["f1_score"] < def_thresh]

    known_task_y = [os.path.join(xml_dir, os.path.split(img_name)[-1].replace("jpeg", "xml"))\
                    for img_name in known_task_x]
    unknown_task_y = [os.path.join(xml_dir, os.path.split(img_name)[-1].replace("jpeg", "xml"))\
                    for img_name in unknown_task_x]

    return (known_task_x, known_task_y), (unknown_task_x, unknown_task_y)

def main(samples, threshold):

    img_dir = os.path.split(samples.x[0])[0]
    xml_dir = os.path.split(samples.y[0])[0]

    args = parse_args()
    print('Called with args:')
    print(args)
    eval_dict_convert((samples.x, samples.y), args)

    fr1 = open('./predict_dict.pkl', 'rb')
    fr2 = open('./gtboxes_dict.pkl', 'rb')

    predict_dict = pickle.load(fr1)
    gtboxes_dict = pickle.load(fr2)

    R, P, AP, F, num = [], [], [], [], []

    f1_score = {}

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)
        # print('label',label)
        rec, prec, ap, box_num, tp_dict, fp_dict, gt_num_dict = eval(rboxes, gboxes, 0.3, False)
        # print("rec",rec)
        # print("prec", prec)
        recall = rec[-1]
        precision = prec[-1]
        F_measure = (2 * precision * recall) / (recall + precision)

        # calulate for each image
        for k, v in tp_dict.items():
            if fp_dict[k] != 0 and gt_num_dict[k] == 0:
                f1_k = 0
            elif fp_dict[k] == 0 and gt_num_dict[k] == 0:
                f1_k = 1
            elif v == 0:
                f1_k = 0
            else:
                prec_k = v / (v + fp_dict[k])
                recal_k = v / gt_num_dict[k]
                f1_k = 2 * prec_k * recal_k / (prec_k + recal_k)

            if not f1_score.get(k):
                f1_score[k] = {}

            f1_score[k][label] = f1_k

        print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall, precision, ap, F_measure))
        R.append(recall)
        P.append(precision)
        AP.append(ap)
        F.append(F_measure)
        num.append(box_num)

    # calulate avg f1_score for each image
    for img, v in f1_score.items():
        f1_scores = []
        for _, label_v in v.items():
            f1_scores.append(label_v)

        f1_score[img]['f1_score'] = np.mean(f1_scores)

    print("num:", num)
    R = np.array(R)
    P = np.array(P)
    AP = np.array(AP)
    F = np.array(F)
    num = np.array(num)
    weights = num / np.sum(num)
    Recall = np.sum(R) / 2
    Precision = np.sum(P) / 2
    mAP = np.sum(AP) / 2
    F_measure = np.sum(F) / 2
    print('\n{}\tR:{}\tP:{}\tmAP:{}\tF:{}'.format('Final', Recall, Precision, mAP, F_measure))

    fr1.close()
    fr2.close()

    return output_tasks(f1_score, img_dir, xml_dir, def_thresh=threshold)

if __name__ == "__main__":
    main(1.0)
