#!/usr/bin/python2

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math

import rospy

def load_image(infilename):
  img = PIL.Image.open(infilename)
  img.load()
  data = np.asarray(img, dtype="int32")
  return data

def load_two_channel_image(infilename):
  img = PIL.Image.open(infilename)
  img.load()
  data = np.asarray(img, dtype="int32")
  data = np.transpose(data, [2, 0, 1])
  data = [data[2], data[1]]
  data = np.transpose(data, axes=[1, 2, 0])
  return data

def load_three_channel_image(infilename):
  img = PIL.Image.open(infilename)
  img.load()
  data = np.asarray(img, dtype="int32")
  return data

def get_inria_dataset_next_kernel(source_file_name_prefix, x_empty_prefix,
                                  x_frontier_prefix, y_prefix, y_channels, enable_augmentation,
                                  start, end):
  image_load_ok = True
  counter = start
  augmentation_counter = 0
  while (image_load_ok):
    empty_filename = source_file_name_prefix + str(counter) + x_empty_prefix
    frontier_filename = source_file_name_prefix + str(counter) + x_frontier_prefix
    gt_filename = source_file_name_prefix + str(counter) + y_prefix

    try:
      #rospy.loginfo("nbv_3d_cnn: loading empty '%s'" % frontier_filename)
      empty = load_image(empty_filename)
      #rospy.loginfo("nbv_3d_cnn: loading frontier '%s'" % frontier_filename)
      frontier = load_image(frontier_filename)
      #rospy.loginfo("nbv_3d_cnn: loading gt '%s'" % gt_filename)
      if (y_channels == 1):
        gt = load_image(gt_filename)
      elif (y_channels == 2):
        gt = load_two_channel_image(gt_filename)
      elif (y_channels == 3):
        gt = load_three_channel_image(gt_filename)

      if ('rotate_channels' in enable_augmentation):
        chan0 = np.copy(np.array(gt[:,:,0]))
        chan1 = np.copy(np.array(gt[:,:,1]))
        for chan_rot_n in range(0, augmentation_counter):
          (chan0, chan1) = (chan1, np.subtract(empty, chan0))
        gt[:,:,0] = chan0
        gt[:,:,1] = chan1
      if ('rotation' in enable_augmentation):
        empty = np.rot90(np.array(empty), k=augmentation_counter)
        frontier = np.rot90(np.array(frontier), k=augmentation_counter)
        gt = np.rot90(np.array(gt), k=augmentation_counter)

      x = [empty, frontier]
      x = np.transpose(x, [1, 2, 0])
      x = np.array(x)

      #y = np.reshape(environment, [len(environment), len(environment[0]), 1])
      if (y_channels == 1):
        gt = np.reshape(gt, [len(gt), len(gt[0]), 1])
      y = np.array(gt)

      x = x.astype('float32') / 255.0
      y = y.astype('float32') / 255.0

      batch = ((x, ), (y, ))

      yield batch

    except IOError as e:
      rospy.logerr('nbv_3d_cnn: could not load image, error is ' + str(e))
      image_load_ok = False
      pass

    if ('rotation' in enable_augmentation):
      augmentation_counter += 1
      if (augmentation_counter >= 4):
        augmentation_counter = 0
        counter += 1
    else:
      counter += 1

    if (counter >= end):
      return

    if (rospy.is_shutdown()):
      exit()
    pass
  pass

# returns: dataset, x_image_width, x_image_height, y_image_width, y_image_height
def get_inria_dataset(source_file_name_prefix, start, end, mode, enable_augmentation):
  empty_prefix = '_empty.png';
  frontier_prefix = '_frontier.png'

  if (mode == 'smooth_directional'):
    output_prefix = '_smooth_directional_gt.png'
    y_channels = 1
  elif (mode == 'directional'):
    output_prefix = '_directional_gt.png'
    y_channels = 1
  elif (mode == 'quat'):
    output_prefix = '_scoreangle_gt.png'
    y_channels = 2
  elif (mode == 'autocomplete'):
    output_prefix = '_environment.png'
    frontier_prefix = '_occupied.png'
    y_channels = 1
  else:
    rospy.logfatal('Invalid mode: ' + mode)
    exit(1)

  generator = get_inria_dataset_next_kernel(source_file_name_prefix,
                                            empty_prefix,
                                            frontier_prefix,
                                            output_prefix,
                                            y_channels,
                                            enable_augmentation,
                                            start, end)
  for ((x, ), (y, )) in generator:
    x_image_width = len(x[0])
    x_image_height = len(x)
    y_image_width = len(y[0])
    y_image_height = len(y)
    break

  dataset = tf.data.Dataset.from_generator(lambda: get_inria_dataset_next_kernel(source_file_name_prefix,
                                                                                 empty_prefix,
                                                                                 frontier_prefix,
                                                                                 output_prefix,
                                                                                 y_channels,
                                                                                 enable_augmentation,
                                                                                 start, end),
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(tf.TensorShape([1, x_image_height, x_image_width, 2]),
                                                          tf.TensorShape([1, y_image_height, y_image_width, y_channels])))
  return dataset, x_image_width, x_image_height, y_image_width, y_image_height
