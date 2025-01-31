# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
from FPN_TensorFlow.libs.label_name_dict.label_dict import LABEl_NAME_MAP


def show_boxes_in_img(img, boxes_and_label):
  '''
  :param img:
  :param boxes: must be int
  :return:
  '''
  boxes_and_label = boxes_and_label.astype(np.int64)
  img = np.array(img, np.float32)
  img = np.array(img*255/np.max(img), np.uint8)
  for box in boxes_and_label:
    ymin, xmin, ymax, xmax, label = box[0], box[1], box[2], box[3], box[4]

    category = LABEl_NAME_MAP[label]

    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
    cv2.rectangle(img,
                  pt1=(xmin, ymin),
                  pt2=(xmax, ymax),
                  color=color)
    cv2.putText(img,
                text=category,
                org=((xmin+xmax)//2, (ymin+ymax)//2),
                fontFace=1,
                fontScale=1,
                color=(0, 0, 255))

  cv2.imshow('img_', img)
  cv2.waitKey(0)


def draw_box_cv(img, boxes, labels, scores):
  img = img + np.array([103.939, 116.779, 123.68])
  boxes = boxes.astype(np.int64)
  labels = labels.astype(np.int32)
  img = np.array(img, np.float32)
  img = np.array(img*255/np.max(img), np.uint8)

  num_of_object = 0
  for i, box in enumerate(boxes):
    ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

    label = labels[i]
    if label != 0:
        if label==1:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 0, 255)
            cv2.rectangle(img,
                        pt1=(xmin, ymin),
                        pt2=(xmax, ymax),
                        color=color,
                        thickness=1)
            # category = LABEl_NAME_MAP[label]
            # cv2.putText(img,
            #             text=category + ": " + str(scores[i]),
            #             org=(xmin, ymin + 10),
            #             fontFace=1,
            #             fontScale=1,
            #             thickness=2,
            #             color=(color[1], color[2], color[0]))
        elif label==2:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            cv2.rectangle(img,
                        pt1=(xmin, ymin),
                        pt2=(xmax, ymax),
                        color=color,
                        thickness=1)
            # category = LABEl_NAME_MAP[label]
            # cv2.putText(img,
            #             text=category,
            #             org=(xmin, ymin + 10),
            #             fontFace=1,
            #             fontScale=1,
            #             thickness=2,
            #             color=(color[1], color[2], color[0]))


      # if scores is not None:
      #     cv2.rectangle(img,
      #                   pt1=(xmin, ymin),
      #                   pt2=(xmin + 120, ymin + 15),
      #                   color=color,
      #                   thickness=-1)
      #     cv2.putText(img,
      #                 text=category+": "+str(scores[i]),
      #                 org=(xmin, ymin+10),
      #                 fontFace=1,
      #                 fontScale=1,
      #                 thickness=2,
      #                 color=(color[1], color[2], color[0]))
      # else:
      #     cv2.rectangle(img,
      #                   pt1=(xmin, ymin),
      #                   pt2=(xmin + 50, ymin + 15),
      #                   color=color,
      #                   thickness=-1)
      #     cv2.putText(img,
      #                 text=category,
      #                 org=(xmin, ymin + 10),
      #                 fontFace=1,
      #                 fontScale=1,
      #                 thickness=2,
      #                 color=(color[1], color[2], color[0]))
  cv2.putText(img,
              text=str(num_of_object),
              org=((img.shape[1]) // 2, (img.shape[0]) // 2),
              fontFace=3,
              fontScale=3,
              color=(255, 255, 255))
  return img


def print_tensors(tensor, tensor_name):
  def np_print(ary):
    ary = ary + np.zeros_like(ary)
    print(tensor_name + ':', ary)

    print('shape is: ', ary.shape)
    print(10*"%%%%%")
    return ary
  result = tf.py_func(np_print,
                      [tensor],
                      [tensor.dtype])
  result = tf.reshape(result, tf.shape(tensor))
  result = tf.cast(result, tf.float32)
  sum_ = tf.reduce_sum(result)
  tf.summary.scalar('print_s/{}'.format(tensor_name), sum_)
