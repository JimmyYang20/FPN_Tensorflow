from __future__ import absolute_import, division, print_function

import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(0, '../../')

from FPN_TensorFlow.libs.networks.slim_nets import (inception_resnet_v2, mobilenet_v1,
                                     resnet_v1, vgg)

# FLAGS = get_flags_byname()


def get_flags_byname(net_name):
  if net_name not in ['resnet_v1_50', 'mobilenet_224', 'inception_resnet', 'vgg16', 'resnet_v1_101']:
    raise ValueError("not include network: {}, we allow resnet_v1_50, mobilenet_224, inception_resnet, "
                     "vgg16, resnet_v1_101"
                     "")

  if net_name == 'resnet_v1_50':
    from FPN_TensorFlow.configs import config_resnet_50
    return config_resnet_50.FLAGS
  if net_name == 'mobilenet_224':
    from FPN_TensorFlow.configs import config_mobilenet_224
    return config_mobilenet_224.FLAGS
  if net_name == 'inception_resnet':
    from FPN_TensorFlow.configs import config_inception_resnet
    return config_inception_resnet.FLAGS
  if net_name == 'vgg16':
    from FPN_TensorFlow.configs import config_vgg16
    return config_vgg16.FLAGS
  if net_name == 'resnet_v1_101':
    from FPN_TensorFlow.configs import config_res101
    return config_res101.FLAGS


def get_network_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       spatial_squeeze=True):
  if net_name == 'resnet_v1_50':
    FLAGS = get_flags_byname(net_name)
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=FLAGS.weight_decay)):
      logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                  num_classes=num_classes,
                                                  is_training=is_training,
                                                  global_pool=global_pool,
                                                  output_stride=output_stride,
                                                  spatial_squeeze=spatial_squeeze
                                                  )

    return logits, end_points
  if net_name == 'resnet_v1_101':
    FLAGS = get_flags_byname(net_name)
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=FLAGS.weight_decay)):
      logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                   num_classes=num_classes,
                                                   is_training=is_training,
                                                   global_pool=global_pool,
                                                   output_stride=output_stride,
                                                   spatial_squeeze=spatial_squeeze
                                                   )
    return logits, end_points

  # if net_name == 'inception_resnet':
  #     FLAGS = get_flags_byname(net_name)
  #     arg_sc = inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=FLAGS.weight_decay)
  #     with slim.arg_scope(arg_sc):
  #         logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs=inputs,
  #                                                                      num_classes=num_classes,
  #                                                                      is_training=is_training)
  #
  #     return logits, end_points
  #
  # if net_name == 'vgg16':
  #     FLAGS = get_flags_byname(net_name)
  #     arg_sc = vgg.vgg_arg_scope(weight_decay=FLAGS.weight_decay)
  #     with slim.arg_scope(arg_sc):
  #         logits, end_points = vgg.vgg_16(inputs=inputs,
  #                                         num_classes=num_classes,
  #                                         is_training=is_training)
  #     return logits, end_points
