#!/usr/bin/python2

import nbv_2d_model
import nbv_3d_model
import inria_dataset
import scene_3d_dataset

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math
import datetime

import rospy

def load_image(infilename):
  img = PIL.Image.open(infilename)
  img.load()
  data = np.asarray(img, dtype="int32")
  return data

rospy.init_node('nbv_3d_cnn', anonymous=True)

source_file_name_prefix = rospy.get_param('~source_images_prefix', '')

dest_file_name_prefix = rospy.get_param('~dest_images_prefix', '')

tensorboard_directory = rospy.get_param('~tensorboard_directory', '')

sensor_range_voxels = rospy.get_param('~sensor_range_voxels', 150)

num_epochs = rospy.get_param('~num_epochs', 300)

load_checkpoint = rospy.get_param('~load_checkpoint', '')

evaluation_only = rospy.get_param('~evaluation_only', False)

model_type = rospy.get_param('~model_type', '')

is_3d = rospy.get_param('~is_3d', False)
if_3d_3d_str = ('_3d' if is_3d else '')

sub_image_expand_pow = rospy.get_param('~sub_image_expand_pow', 2)
sub_image_expand = 2**sub_image_expand_pow

# enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

final_val_dataset_augmentation = []
dataset_augmentation = []
batch_size = 1
if (model_type == '' or model_type == '2'):
  if (not is_3d) and (model_type == '2'):
    dataset_mode = 'smooth_directional'
  else:
    dataset_mode = 'directional'
  dataset_augmentation = ['rotation', ]
elif (model_type == 'flat'):
  dataset_mode = 'smooth_directional'
  if not is_3d:
    dataset_augmentation = ['rotation', ]
  if is_3d:
    batch_size = 2
    dataset_augmentation = ['files8', ]
elif (model_type == 'quat'):
  dataset_mode = 'quat'
  if not is_3d:
    dataset_augmentation = ['rotation', 'rotate_channels']
  if is_3d:
    batch_size = 5
    dataset_augmentation = ['files8', ]
elif (model_type == 'autocomplete'):
  dataset_mode = 'autocomplete'
  if not is_3d:
    dataset_augmentation = ['rotation', ]
elif (model_type == 'circular'):
  if not is_3d:
    dataset_mode = 'smooth_directional'
  else:
    dataset_mode = 'directional'
else:
  rospy.logfatal('Unknown model type: ' + model_type)
  exit(1)

if not is_3d:
  dataset, image_width, image_height, y_image_width, y_image_height = (
    inria_dataset.get_inria_dataset(source_file_name_prefix, 0, 120, dataset_mode, dataset_augmentation))
  val_dataset, val_image_width, val_image_height, val_y_image_width, val_y_image_height = (
    inria_dataset.get_inria_dataset(source_file_name_prefix, 120, 180, dataset_mode, dataset_augmentation))
  pass
else:
  dataset, image_width, image_height, image_depth, y_image_width, y_image_height, y_image_depth = (
    scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, 0, 120, dataset_mode, dataset_augmentation))
  val_dataset, val_image_width, val_image_height, val_image_depth, val_y_image_width, val_y_image_height, val_y_image_depth = (
    scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, 120, 180, dataset_mode, dataset_augmentation))
  pass

dataset.batch(batch_size)
val_dataset.batch(batch_size)

if not is_3d:
  input_shape = (image_height, image_width, 2)
else:
  input_shape = (image_depth, image_height, image_width, 2)
batch_size = 1

if (model_type == '' or model_type == '2'):
  if not is_3d:
    model = nbv_2d_model.get_2d_model(sensor_range_voxels, input_shape, sub_image_expand_pow)
  else:
    model = nbv_3d_model.get_3d_model(sensor_range_voxels, input_shape, sub_image_expand_pow)
  model_file_prefix = model_type
  output_channels = 1
elif (model_type == 'flat'):
  if not is_3d:
    model = nbv_2d_model.get_flat_2d_model(sensor_range_voxels, input_shape, sub_image_expand_pow)
    output_channels = 1
  else:
    model = nbv_3d_model.get_flat_3d_model(sensor_range_voxels, input_shape, 52)
    output_channels = 52
  model_file_prefix = 'flat'
  sub_image_expand_pow = 0
  sub_image_expand = 1
elif (model_type == 'quat'):
  if not is_3d:
    model = nbv_2d_model.get_quat_2d_model(sensor_range_voxels, input_shape)
    output_channels = 2
  else:
    model = nbv_3d_model.get_quat_3d_model(sensor_range_voxels, input_shape)
    output_channels = 4
  model_file_prefix = 'scoreangle'
  sub_image_expand_pow = 0
  sub_image_expand = 1
elif (model_type == 'autocomplete'):
  if not is_3d:
    model = nbv_2d_model.get_autocomplete_2d_model(sensor_range_voxels, input_shape)
  else:
    model = nbv_3d_model.get_autocomplete_3d_model(sensor_range_voxels, input_shape)
  model_file_prefix = 'autocomplete'
  output_channels = 1
  sub_image_expand_pow = 0
  sub_image_expand = 1
elif (model_type == 'circular'):
  model = nbv_2d_model.get_circular_2d_model(sensor_range_voxels, input_shape, sub_image_expand_pow)
  model_file_prefix = 'circular'
  output_channels = 1
else:
  rospy.logfatal('Unknown model type: ' + model_type)
  exit(1)

model.summary()

if (load_checkpoint != ''):
  rospy.loginfo("loading weights from " + dest_file_name_prefix + load_checkpoint)
  model.load_weights(dest_file_name_prefix + load_checkpoint)

if (not evaluation_only):

  class TerminationCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
      pass
    def on_epoch_end(self, epoch, logs=None):
      if (rospy.is_shutdown()):
        exit()
      pass
    pass

  class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
      self.model = model
      pass
    def on_epoch_end(self, epoch, logs=None):
      if (epoch % 20 == 0):
        self.model.save_weights(dest_file_name_prefix + model_file_prefix + if_3d_3d_str +
                                '_epoch_' + str(epoch) + '.chkpt')
      pass
    pass

  str_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_directory = (tensorboard_directory + str_now + '_' + model_file_prefix + if_3d_3d_str)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_directory, histogram_freq=1)

  model.fit(dataset,
            epochs=num_epochs,
            callbacks=[TerminationCallback(), SaveModelCallback(model), tensorboard_callback],
            validation_data=val_dataset
            )

  model_filename = dest_file_name_prefix + model_file_prefix + if_3d_3d_str + '_final.chkpt'
  rospy.loginfo("nbv_3d_cnn: saving model '%s'" % model_filename)
  model.save_weights(model_filename)
  pass

rospy.loginfo('nbv_3d_cnn: predicting and saving images...')
counter = 0
if is_3d:
  dataset, image_width, image_height, image_depth, y_image_width, y_image_height, y_image_depth = (
    scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, 0, 120, dataset_mode, final_val_dataset_augmentation))
  val_dataset, val_image_width, val_image_height, val_image_depth, val_y_image_width, val_y_image_height, val_y_image_depth = (
    scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, 120, 180, dataset_mode, final_val_dataset_augmentation))
  pass

for i,x in enumerate(dataset.concatenate(val_dataset)):
  img = model.predict(x)
  result_filename = dest_file_name_prefix + str(counter) + '_result'

  final_img_shape = [image_width * sub_image_expand, image_height * sub_image_expand]
  if (is_3d):
    final_img_shape.append(image_depth * sub_image_expand)
  if (output_channels == 4):
    final_img_shape.append(output_channels)
  if (is_3d and output_channels == 52):
    final_img_shape[2] *= 52
  img = np.reshape(img, final_img_shape)

  if ((not is_3d) and output_channels == 2):
    img = np.transpose(img, [2, 0, 1])
    img = np.asarray([np.full((image_height, image_width), 0.0), img[1], img[0]])
    img = np.transpose(img, [1, 2, 0])
    pass

  img = np.clip(img, 0.0, 1.0)
  if (not is_3d):
    img = (img * 255).astype(np.uint8)
    img = PIL.Image.fromarray(img, 'L' if (output_channels == 1) else 'RGB')
    img.save(result_filename + ".png")
    pass

  if (is_3d and output_channels == 1 or output_channels == 52):
    scene_3d_dataset.save_voxelgrid(result_filename, img)
  if (is_3d and output_channels == 4):
    scene_3d_dataset.save_voxelgrid_nchannels(result_filename, img)
  
  rospy.loginfo('nbv_3d_cnn: saved image %s' % result_filename)
  
  counter += 1
  pass

rospy.loginfo('nbv_3d_cnn: saved %s images' % str(counter))

