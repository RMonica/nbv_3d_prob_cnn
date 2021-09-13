#!/usr/bin/python2

import nbv_2d_model
import nbv_3d_model
import nbv_3d_cnn_msgs.msg as nbv_3d_cnn_msgs
import std_msgs.msg as std_msgs
from rospy.numpy_msg import numpy_msg
import nbv_3d_cnn_msgs.msg as nbv_3d_cnn_msgs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math

import rospy
import actionlib
import cv_bridge

class PredictAction(object):
  def __init__(self):

    self.last_input_shape = [0, 0]
    self.model = None

    self.sensor_range_voxels = rospy.get_param('~sensor_range_voxels', 150)
    self.checkpoint_file = rospy.get_param('~checkpoint_file', '')
    self.action_name = rospy.get_param('~action_name', '~predict')
    self.model_type = rospy.get_param('~model_type', '')
    self.sub_image_expand_pow = rospy.get_param('~sub_image_expand_pow', 2)
    self.is_3d = rospy.get_param('~is_3d', False)
    self.sub_image_expand = 2**self.sub_image_expand_pow

    self.log_accuracy_skip_voxels = rospy.get_param('~log_accuracy_skip_voxels', 0)

    self.raw_pub = rospy.Publisher(self.action_name + "raw_data", numpy_msg(nbv_3d_cnn_msgs.Floats), queue_size=1)

    if self.is_3d:
      self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_msgs.Predict3dAction,
                                               execute_cb=self.on_predict_3d, auto_start=False)
    else:
      self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_msgs.PredictAction,
                                               execute_cb=self.on_predict, auto_start=False)
    self.action_server.start()
    rospy.loginfo('nbv_3d_cnn_predict: action \'%s\' started.' % self.action_name)
    pass

  def on_predict_3d(self, goal):
    rospy.loginfo('nbv_3d_cnn_predict: action start.')

    output_size_mult = 1
    if (self.model_type == ''):
      model_file_prefix = ''
      output_channels = 1
      output_size_mult = 2**self.sub_image_expand_pow
    elif (self.model_type == 'flat'):
      model_file_prefix = 'flat'
      output_channels = 52
    elif (self.model_type == 'circular'):
      model_file_prefix = 'circular'
      output_channels = 1
    elif (self.model_type == 'quat'):
      model_file_prefix = 'quat'
      output_channels = 4
    elif (self.model_type == 'autocomplete'):
      model_file_prefix = 'autocomplete'
      output_channels = 1
    else:
      rospy.logfatal('Unknown model type: ' + self.model_type)
      exit(1)

    image_width = goal.empty.layout.dim[2].size
    image_height = goal.empty.layout.dim[1].size
    image_depth = goal.empty.layout.dim[0].size

    frontier_image_width = goal.frontier.layout.dim[2].size
    frontier_image_height = goal.frontier.layout.dim[1].size
    frontier_image_depth = goal.frontier.layout.dim[0].size

    if (frontier_image_width != image_width or frontier_image_height != image_height or
        frontier_image_depth != image_depth):
      rospy.error('nbv_3d_cnn_predict: empty image has size %d %d %d, but frontier image has size %d %d %d, aborted.' %
                  (image_width, image_height, image_depth, frontier_image_width, frontier_image_height, frontier_image_depth))
      self.action_server.set_aborted()
      return

    np_empty_image = np.reshape(np.array(goal.empty.data).astype('float32'), [image_depth, image_height, image_width])
    np_frontier_image = np.reshape(np.array(goal.frontier.data).astype('float32'), [image_depth, image_height, image_width])
    input_image = [np_empty_image, np_frontier_image]
    input_image = np.transpose(input_image, [1, 2, 3, 0])

    input_shape = [image_depth, image_height, image_width]
    load_input_shape = [image_depth, image_height, image_width, 2]

    if (self.model == None or self.last_input_shape != input_shape):
      rospy.loginfo('nbv_3d_cnn_predict: last input shape does not match, reloading model (type "%s").' % self.model_type)

      if (self.model_type == ''):
        model = nbv_3d_model.get_3d_model(self.sensor_range_voxels, load_input_shape,
                                          self.sub_image_expand_pow, self.log_accuracy_skip_voxels)
      elif (self.model_type == 'flat'):
        model = nbv_3d_model.get_flat_3d_model(self.sensor_range_voxels, load_input_shape, 52,
                                               self.log_accuracy_skip_voxels)
      elif (self.model_type == 'quat'):
        model = nbv_3d_model.get_quat_3d_model(self.sensor_range_voxels, load_input_shape, self.log_accuracy_skip_voxels)
      elif (self.model_type == 'autocomplete'):
        model = nbv_3d_model.get_autocomplete_3d_model(self.sensor_range_voxels, load_input_shape)

      model.summary()
      model.load_weights(self.checkpoint_file)

      self.model = model
      self.last_input_shape = input_shape
    else:
      model = self.model

    rospy.loginfo('nbv_3d_cnn_predict: predicting.')
    prediction = model.predict(np.array([input_image, ]))

    rospy.loginfo('nbv_3d_cnn_predict: sending result.')
    prediction = prediction[0]

    if (output_channels == 1):
      prediction = np.transpose(prediction, [3, 0, 1, 2])
      prediction = prediction[0]
      pass

    out_image_depth = len(prediction)
    out_image_height = len(prediction[0])
    out_image_width = len(prediction[0][0])

    prediction = np.reshape(prediction.astype('float32'), [out_image_width * out_image_height * out_image_depth *
                                                           output_channels, ])

    if (len(prediction) > 10000):
      rospy.loginfo('nbv_3d_cnn_predict: response is big, using raw publisher.')
      self.raw_pub.publish(prediction)

    result = nbv_3d_cnn_msgs.Predict3dResult()
    if (len(prediction) <= 10000):
      result.scores.data = prediction.tolist()

    dim_x = std_msgs.MultiArrayDimension()
    dim_x.label = "x"
    dim_x.size = out_image_width
    dim_x.stride = output_channels
    dim_y = std_msgs.MultiArrayDimension()
    dim_y.label = "y"
    dim_y.size = out_image_height
    dim_y.stride = out_image_width * output_channels
    dim_z = std_msgs.MultiArrayDimension()
    dim_z.label = "z"
    dim_z.size = out_image_depth
    dim_z.stride = out_image_width * out_image_height * output_channels
    dim_c = std_msgs.MultiArrayDimension()
    dim_c.label = "channels"
    dim_c.size = output_channels
    dim_c.stride = 1
    layout_dims = [dim_z, dim_y, dim_x]
    if (output_channels != 1):
      layout_dims.append(dim_c)
    result.scores.layout.dim = layout_dims

    self.action_server.set_succeeded(result)

    rospy.loginfo('nbv_3d_cnn_predict: action succeeded.')
    pass

  def on_predict(self, goal):
    rospy.loginfo('nbv_3d_cnn_predict: action start.')

    if (self.model_type == ''):
      model_file_prefix = ''
      output_channels = 1
    elif (self.model_type == 'flat'):
      model_file_prefix = 'flat'
      output_channels = 1
    elif (self.model_type == 'circular'):
      model_file_prefix = 'circular'
      output_channels = 1
    elif (self.model_type == 'quat'):
      model_file_prefix = 'scoreangle'
      output_channels = 2
    elif (self.model_type == 'autocomplete'):
      model_file_prefix = 'autocomplete'
      output_channels = 1
    else:
      rospy.logfatal('Unknown model type: ' + self.model_type)
      exit(1)

    image_width = goal.empty.width
    image_height = goal.empty.height

    if (goal.frontier.width != image_width or goal.frontier.height != image_height):
      rospy.error('nbv_3d_cnn_predict: empty image has size %d %d, but frontier image has size %d %d, aborted.' %
                  (image_width, image_height, goal.frontier.width, goal.frontier.height))
      self.action_server.set_aborted()
      return

    bridge = cv_bridge.CvBridge()
    empty_image = bridge.imgmsg_to_cv2(goal.empty, desired_encoding='mono8')
    frontier_image = bridge.imgmsg_to_cv2(goal.frontier, desired_encoding='mono8')

    np_empty_image = empty_image
    np_frontier_image = frontier_image
    input_image = [np_empty_image, np_frontier_image]
    input_image = np.transpose(input_image, [1, 2, 0])

    input_image = input_image.astype('float32') / 255.0 # convert to zeros and ones

    input_shape = [image_height, image_width]

    if (self.model == None or self.last_input_shape != input_shape):
      load_input_shape = [image_height, image_width, 2]
      rospy.loginfo('nbv_3d_cnn_predict: last input shape does not match, reloading model (type "%s").' % self.model_type)
      #model = nbv_2d_model.get_2d_model(self.sensor_range_voxels, load_input_shape, 2)

      if (self.model_type == ''):
        model = nbv_2d_model.get_2d_model(self.sensor_range_voxels, load_input_shape, self.sub_image_expand_pow)
      elif (self.model_type == 'flat'):
        model = nbv_2d_model.get_flat_2d_model(self.sensor_range_voxels, load_input_shape, self.sub_image_expand_pow)
      elif (self.model_type == 'quat'):
        model = nbv_2d_model.get_quat_2d_model(self.sensor_range_voxels, load_input_shape)
      elif (self.model_type == 'autocomplete'):
        model = nbv_2d_model.get_autocomplete_2d_model(self.sensor_range_voxels, load_input_shape)
      elif (self.model_type == 'circular'):
        model = nbv_2d_model.get_circular_2d_model(self.sensor_range_voxels, load_input_shape, self.sub_image_expand_pow)

      model.summary()
      model.load_weights(self.checkpoint_file)

      self.model = model
      self.last_input_shape = input_shape
    else:
      model = self.model

    rospy.loginfo('nbv_3d_cnn_predict: predicting.')
    prediction = model.predict(np.array([input_image, ]))

    rospy.loginfo('nbv_3d_cnn_predict: sending result.')
    prediction = prediction[0]

    if (output_channels == 1):
      prediction = np.transpose(prediction, [2, 0, 1])
      prediction = prediction[0]
    elif (output_channels == 2):
      prediction = np.transpose(prediction, [2, 0, 1])
      prediction = np.asarray([prediction[0], prediction[1],
                               np.full((image_height * self.sub_image_expand, image_width * self.sub_image_expand), 0.0)])
      prediction = np.transpose(prediction, [1, 2, 0])
      pass

    if (output_channels == 1):
      encoding = "32FC1"
    else:
      encoding = "32FC3"
      pass

    prediction = prediction.astype('float32')

    result = nbv_3d_cnn_msgs.PredictResult()
    result.scores = bridge.cv2_to_imgmsg(prediction, encoding=encoding)

    self.action_server.set_succeeded(result)

    rospy.loginfo('nbv_3d_cnn_predict: action succeeded.')
    pass
  pass

rospy.init_node('nbv_3d_cnn_predict', anonymous=True)

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
# limit GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
      gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_mb)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print("Exception while limiting GPU memory:")
    print(e)
    exit()

server = PredictAction()
rospy.spin()
