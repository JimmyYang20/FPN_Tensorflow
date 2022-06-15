#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from FPN_TensorFlow.data.io import image_preprocess
from FPN_TensorFlow.libs.configs import cfgs
from FPN_TensorFlow.libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from FPN_TensorFlow.data.io.convert_to_tfrecord import convert_label_list, convert_pascal_to_tfrecord_from_list

def correct_decode_raw(data, dtype):
  # BUG: THERE WAS A BUG HERE, tf.decode_raw('', tf.float32) returns [0.] tensor not the [] tensor
  # So we use correct_decode_raw instead of tf.decode_raw
  result = tf.cond(tf.equal(data, tf.constant("")),
                   lambda: tf.constant([], dtype=dtype),
                   lambda: tf.decode_raw(data, dtype))

  return result


def read_single_example_and_decode(filename_queue):

  # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

  # reader = tf.TFRecordReader(options=tfrecord_options)
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized=serialized_example,
      features={
          'img_name': tf.FixedLenFeature([], tf.string),
          'img_height': tf.FixedLenFeature([], tf.int64),
          'img_width': tf.FixedLenFeature([], tf.int64),
          'img': tf.FixedLenFeature([], tf.string),
          'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
          'num_objects': tf.FixedLenFeature([], tf.int64)
      }
  )
  img_name = features['img_name']
  img_height = tf.cast(features['img_height'], tf.int32)
  img_width = tf.cast(features['img_width'], tf.int32)
  img = tf.decode_raw(features['img'], tf.uint8)

  img = tf.reshape(img, shape=[img_height, img_width, 3])
  # img.set_shape([None, None, 3])

  gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
  # gtboxes_and_label = correct_decode_raw(features['gtboxes_and_label'], tf.int32)
  gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
  # gtboxes_and_label.set_shape([None, 5])

  num_objects = tf.cast(features['num_objects'], tf.int32)
  return img_name, img, gtboxes_and_label, num_objects


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

  img_name, img, gtboxes_and_label, num_objects = read_single_example_and_decode(filename_queue)
  # img = tf.image.per_image_standardization(img)
  img = tf.cast(img, tf.float32)
  img = img - tf.constant([103.939, 116.779, 123.68])
  if is_training:
    img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                target_shortside_len=shortside_len)
    img, gtboxes_and_label = image_preprocess.random_flip_left_right(
        img_tensor=img, gtboxes_and_label=gtboxes_and_label)

  else:
    img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                target_shortside_len=shortside_len)

  return img_name, img, gtboxes_and_label, num_objects

def read_and_prepocess_single_label(filename_queue, target_shortside_len):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    num_objects = tf.cast(features['num_objects'], tf.int32)

    h, w = img_height, img_width
    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                           false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))


    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    xmin, xmax = xmin * new_w // w, xmax * new_w // w
    ymin, ymax = ymin * new_h // h, ymax * new_h // h

    gtboxes_and_label = tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))

    return img_name, gtboxes_and_label, num_objects

def next_batch(dataset_name, batch_size, shortside_len, is_training):
  if dataset_name not in ['cooler', 'airplane', 'SSDD', 'ship', 'pascal', 'coco', 'layer', 'shelf', 'user']:
    raise ValueError('dataSet name must be in cooler, pascal, coco, layer or shelf')

  if is_training:
    # pattern = os.path.join(os.path.join(cfgs.ROOT_PATH, 'data/tfrecords'), dataset_name + '_train*')
    pattern = os.path.join(os.path.join(cfgs.ROOT_PATH, 'data/tfrecords'), dataset_name + '_train.tfrecord')
  else:
    pattern = os.path.join(os.path.join(cfgs.ROOT_PATH, 'data/tfrecords'),
                           dataset_name + '_test.tfrecord')

  print('tfrecord path is -->', os.path.abspath(pattern))

  dataset_num = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.abspath(pattern)))
  # dataset_num = 0

  filename_tensorlist = tf.train.match_filenames_once(pattern)

  filename_queue = tf.train.string_input_producer(filename_tensorlist)

  img_name, img, gtboxes_and_label, num_obs = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                            is_training=is_training)
  img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch = \
      tf.train.batch(
          [img_name, img, gtboxes_and_label, num_obs],
          batch_size=batch_size,
          capacity=100,
          num_threads=1, # 16,
          dynamic_pad=True)
  return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, dataset_num

def next_batch_for_tasks(samples, dataset_name, batch_size, shortside_len, is_training=True, save_name=None):
  img_list, xml_list = samples
  pattern = convert_pascal_to_tfrecord_from_list(img_list, xml_list, save_name=save_name)

  if dataset_name not in ['cooler', 'airplane', 'SSDD', 'ship', 'pascal', 'coco', 'layer', 'shelf', 'user']:
    raise ValueError('dataSet name must be in cooler, pascal, coco, layer or shelf')

  print('tfrecord path is -->', os.path.abspath(pattern))

  dataset_num = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.abspath(pattern)))
  # dataset_num = 0

  filename_tensorlist = tf.train.match_filenames_once(pattern)

  filename_queue = tf.train.string_input_producer(filename_tensorlist)

  img_name, img, gtboxes_and_label, num_obs = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                            is_training=is_training)
  img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch = \
      tf.train.batch(
          [img_name, img, gtboxes_and_label, num_obs],
          batch_size=batch_size,
          capacity=100,
          num_threads=1, # 16,
          dynamic_pad=True)
  return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, dataset_num

def convert_labels(samples,
                   dataset_name=cfgs.DATASET_NAME,
                   shortside_len=cfgs.SHORT_SIDE_LEN,
                   save_name="test"):

    with tf.Graph().as_default():
        pattern= convert_label_list(samples, save_name=save_name)

        if dataset_name not in ['cooler', 'airplane', 'SSDD', 'ship', 'pascal', 'coco', 'layer', 'shelf', 'user']:
            raise ValueError('dataSet name must be in cooler, pascal, coco, layer or shelf')

        print('tfrecord path is -->', os.path.abspath(pattern))

        dataset_num = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.abspath(pattern)))

        filename_tensorlist = tf.train.match_filenames_once(pattern)

        filename_queue = tf.train.string_input_producer(filename_tensorlist)

        img_name, gtboxes_and_label, num_obs = read_and_prepocess_single_label(filename_queue, shortside_len)

        img_name_batch, gtboxes_and_label_batch, num_obs_batch = \
            tf.train.batch(
                [img_name, gtboxes_and_label, num_obs],
                batch_size=dataset_num,
                capacity=dataset_num,
                num_threads=1,  # 16,
                dynamic_pad=True)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            gtbox_dict = {}

            for i in range(1):
                _img_name_batch, _gtboxes_and_label_batch, _num_obs_batch = \
                    sess.run([img_name_batch, gtboxes_and_label_batch, num_obs_batch])

                for i in range(_gtboxes_and_label_batch.shape[0]):
                    gtboxes = _gtboxes_and_label_batch[i][:_num_obs_batch[i]]
                    gtbox_dict[str(_img_name_batch[i])] = []

                    for j, box in enumerate(gtboxes):
                        bbox_dict = {}
                        bbox_dict['bbox'] = np.array(gtboxes[j, :-1], np.float64)
                        bbox_dict['name'] = LABEl_NAME_MAP[int(gtboxes[j, -1])]
                        gtbox_dict[str(_img_name_batch[i])].append(bbox_dict)

            coord.request_stop()
            coord.join(threads)

    return gtbox_dict

if __name__ == '__main__':

  xml_dir = cfgs.ROOT_PATH + f"/data/increment_data/test_data/Annotations"

  # 传入jpeg和xml的地址
  xmls = [os.path.join(xml_dir, xml_name) for xml_name in os.listdir(xml_dir)]

  convert_labels(xmls)