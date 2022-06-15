# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import xml.etree.cElementTree as ET

import cv2
import numpy as np
import tempfile
import tensorflow.compat.v1 as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from FPN_TensorFlow.help_utils.tools import *
from FPN_TensorFlow.libs.configs import cfgs
from FPN_TensorFlow.libs.label_name_dict.label_dict import *

tf.app.flags.DEFINE_string('VOC_dir', '../{}/'.format(cfgs.DATASET_NAME), 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages/', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/data/tfrecords/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpeg', 'format of image')
FLAGS = tf.app.flags.FLAGS

split = int('200100')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  # print("value",type(value))
  if type(value)==str:
    value=value.encode()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
  """
  :param xml_path: the path of voc xml
  :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
         and has [xmin, ymin, xmax, ymax, label] in a per row
  """

  tree = ET.parse(xml_path)
  root = tree.getroot()
  img_width = None
  img_height = None
  box_list = []
  for child_of_root in root:
    # if child_of_root.tag == 'filename':
    #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
    #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

    if child_of_root.tag == 'size':
      for child_item in child_of_root:
        if child_item.tag == 'width':
          img_width = int(child_item.text)
        if child_item.tag == 'height':
          img_height = int(child_item.text)

    if child_of_root.tag == 'object':
      label = None
      for child_item in child_of_root:
        if child_item.tag == 'name':
          label = NAME_LABEL_MAP[child_item.text]
        if child_item.tag == 'bndbox':
          tmp_box = []
          for node in child_item:
            tmp_box.append(int(node.text))  # [x1, y1. x2, y2]
          assert label is not None, 'label is none, error'
          tmp_box.append(label)  # [x1, y1. x2, y2, label]
          box_list.append(tmp_box)

  gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, label]

  xmin, ymin, xmax, ymax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2],\
      gtbox_label[:, 3], gtbox_label[:, 4]

  xmin = np.where(xmin <= 0, 1, xmin)
  ymin = np.where(ymin <= 0, 1, ymin)
  xmax = np.where(xmax >= img_width, img_width-1, xmax)
  ymax = np.where(ymax >= img_height, img_height-1, ymax)

  # [ymin, xmin, ymax, xmax, label]
  gtbox_label = np.transpose(np.stack([ymin, xmin, ymax, xmax, label], axis=0))

  return img_height, img_width, gtbox_label


def read_image(path):
  # txt_path = path.replace('JPEGImages', 'Annotations').replace('jpeg', 'txt')
  # with open(txt_path) as f:
    # line = f.readlines()[0]
  # flag = True if line.startswith('rotate') else False
  im = cv2.imread(path)
  # if flag:
  #   rotate = int(line.strip().split()[1])
  #   rot_mat = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), rotate, 1)
  #   im = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]))

  return im

def convert_pascal_to_tfrecord(xml_path, image_path):

  # xml_path = FLAGS.VOC_dir + FLAGS.xml_dir
  # image_path = FLAGS.VOC_dir + FLAGS.image_dir
  save_path = FLAGS.save_dir + cfgs.DATASET_NAME + '_' + FLAGS.save_name + '.tfrecord'
  mkdir(FLAGS.save_dir)

  writer = tf.python_io.TFRecordWriter(path=save_path)

  xmls = [os.path.join(xml_path, f).replace('jpeg', 'xml')
          for f in os.listdir(image_path) if not f.startswith('.')]

  print('{} in train...'.format(len(xmls)))

  for count, xml in enumerate(xmls):
    # to avoid path error in different development platform
    xml = xml.replace('\\', '/')

    img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
    img_path = image_path + '/' + img_name

    if not os.path.exists(img_path):
      print('{} is not exist!'.format(img_path))
      continue

    if os.path.isfile(xml):

      img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

      # img = np.array(Image.open(img_path))
      # img = cv2.imread(img_path)
      img = read_image(img_path)

      feature = tf.train.Features(feature={
          # maybe do not need encode() in linux
          'img_name': _bytes_feature(img_name),
          'img_height': _int64_feature(img_height),
          'img_width': _int64_feature(img_width),
          'img': _bytes_feature(img.tostring()),
          'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
          'num_objects': _int64_feature(gtbox_label.shape[0])
      })

      example = tf.train.Example(features=feature)

      writer.write(example.SerializeToString())

      view_bar('Conversion progress', count + 1, len(xmls))

  print('\nConversion is completed!')
  return save_path

def convert_pascal_to_test_tfrecord(xml_path, image_path):
  temp_dir = tempfile.mkdtemp()
  if not os.path.isdir(temp_dir):
    mkdir(temp_dir)

  save_path = temp_dir + "/" + cfgs.DATASET_NAME + '_' + 'test' + '.tfrecord'

  # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
  # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
  writer = tf.python_io.TFRecordWriter(path=save_path)

  xmls = [os.path.join(xml_path, f).replace('jpeg', 'xml')
          for f in os.listdir(image_path) if not f.startswith('.')]
          # for f in os.listdir(image_path) if not f.startswith('.') and int(f[:6]) <= split]

  print('{} in test...'.format(len(xmls)))

  for count, xml in enumerate(xmls):
    # to avoid path error in different development platform
    xml = xml.replace('\\', '/')

    img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
    img_path = image_path + '/' + img_name

    if not os.path.exists(img_path):
      print('{} is not exist!'.format(img_path))
      continue

    if os.path.isfile(xml):

      img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

      # img = np.array(Image.open(img_path))
      # img = cv2.imread(img_path)
      img = read_image(img_path)

      feature = tf.train.Features(feature={
          # maybe do not need encode() in linux
          'img_name': _bytes_feature(img_name),
          'img_height': _int64_feature(img_height),
          'img_width': _int64_feature(img_width),
          'img': _bytes_feature(img.tostring()),
          'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
          'num_objects': _int64_feature(gtbox_label.shape[0])
      })

      example = tf.train.Example(features=feature)

      writer.write(example.SerializeToString())

      view_bar('Conversion progress', count + 1, len(xmls))

  print('\nConversion is completed!')
  return save_path

def convert_pascal_to_tfrecord_from_list(image_list, xml_list, save_name="test"):
  # save_path = FLAGS.save_dir + cfgs.DATASET_NAME + '_' + "temporary" + "_" + save_name + '.tfrecord'
  # mkdir(FLAGS.save_dir)

  temp_dir = tempfile.mkdtemp()
  if not os.path.isdir(temp_dir):
    mkdir(temp_dir)
  save_path = temp_dir + "/" + cfgs.DATASET_NAME + '_' + "temporary" + "_" + save_name + '.tfrecord'

  writer = tf.python_io.TFRecordWriter(path=save_path)

  print('{} in train...'.format(len(xml_list)))

  for count, xml in enumerate(xml_list):
    # to avoid path error in different development platform
    xml = xml.replace('\\', '/')

    img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
    img_path = [img for img in image_list if img_name in img][0]

    if not os.path.exists(img_path):
      print('{} is not exist!'.format(img_path))
      continue

    if os.path.isfile(xml):

      img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
      img = read_image(img_path)

      feature = tf.train.Features(feature={
          # maybe do not need encode() in linux
          'img_name': _bytes_feature(img_name),
          'img_height': _int64_feature(img_height),
          'img_width': _int64_feature(img_width),
          'img': _bytes_feature(img.tostring()),
          'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
          'num_objects': _int64_feature(gtbox_label.shape[0])
      })

      example = tf.train.Example(features=feature)

      writer.write(example.SerializeToString())

      view_bar('Conversion progress', count + 1, len(xml_list))

  print('\nConversion is completed!')
  return save_path

def convert_label_list(lable_list, save_name="label"):
  temp_dir = tempfile.mkdtemp()
  if not os.path.isdir(temp_dir):
    mkdir(temp_dir)
  save_path = temp_dir + "/" + cfgs.DATASET_NAME + '_' + "temporary" + "_" + save_name + '.tfrecord'

  writer = tf.python_io.TFRecordWriter(path=save_path)
  print('{} in train...'.format(len(lable_list)))

  for count, xml in enumerate(lable_list):
    # to avoid path error in different development platform
    xml = xml.replace('\\', '/')
    img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format

    if os.path.isfile(xml):
      img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
      feature = tf.train.Features(feature={
          # maybe do not need encode() in linux
          'img_name': _bytes_feature(img_name),
          'img_height': _int64_feature(img_height),
          'img_width': _int64_feature(img_width),
          # 'img': _bytes_feature(img.tostring()),
          'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
          'num_objects': _int64_feature(gtbox_label.shape[0])
      })
      example = tf.train.Example(features=feature)
      writer.write(example.SerializeToString())
      view_bar('Conversion progress', count + 1, len(lable_list))

  print('\nConversion is completed!')
  return save_path

if __name__ == '__main__':
  # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
  # read_xml_gtbox_and_label(xml_path)
  # convert_pascal_to_tfrecord()
  # convert_pascal_to_test_tfrecord()

  xml_dir = cfgs.ROOT_PATH + f"/data/increment_data/test_data/Annotations"

  # 传入jpeg和xml的地址
  xmls = [os.path.join(xml_dir, xml_name) for xml_name in os.listdir(xml_dir)]

  convert_label_list(xmls)

  # lis=os.listdir('../layer/JPEGImages/')
  # print(lis)