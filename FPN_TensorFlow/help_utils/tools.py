# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os

import numpy as np


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_single_label_dict(predict_dict, gtboxes_dict, label):
    rboxes = {}
    gboxes = {}
    rbox_images = predict_dict.keys()
    rbox_images = list(rbox_images)

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        for pre_box in predict_dict[rbox_image]:
            if pre_box['name'] == label and len(pre_box['bbox']) != 0:
                rboxes[rbox_image] = [pre_box]

                gboxes[rbox_image] = []

                for gt_box in gtboxes_dict[rbox_image]:
                    if gt_box['name'] == label:
                        gboxes[rbox_image].append(gt_box)
    return rboxes, gboxes


def single_label_eval(rboxes, gboxes, iou_th, use_07_metric):
    rbox_images = list(rboxes.keys())
    fp = np.zeros(len(rbox_images))
    tp = np.zeros(len(rbox_images))
    box_num = 0

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        if len(rboxes[rbox_image][0]['bbox']) > 0:

            rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
            if len(gboxes[rbox_image]) > 0:
                gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
                box_num = box_num + len(gbox_list)
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
                fp[i] += len(rboxes[rbox_image][0]['bbox'])
        else:
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

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / box_num

    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, box_num


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