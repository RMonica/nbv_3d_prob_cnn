#!/usr/bin/python2

import tensorflow as tf

import numpy as np
import math
import sys

def get_quat_2d_model(sensor_range_voxels, input_shape):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [8*(2**k) for k in range(0, num_layers)]
  print("get_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some obscure reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  x = tf.keras.layers.Conv2DTranspose(filters=2,
                                      kernel_size=1,
                                      strides=1,
                                      activation='linear',
                                      #activation='sigmoid',
                                      padding='same')(x)

  # extract empty from input
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[3, 0, 1, 2])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.concatenate([empty_only_matrix, empty_only_matrix], axis=3)

  outputs = x * empty_only_matrix # and multiply it by output

  model = tf.keras.models.Model(inputs, outputs, name='quat_2d_model')

  model.compile(loss='mse', optimizer='adam')

  return model

def get_autocomplete_2d_model(sensor_range_voxels, input_shape):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [8*(2**k) for k in range(0, num_layers)]
  print("get_autocomplete_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some obscure reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  x = tf.keras.layers.Conv2DTranspose(filters=1,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid', # tested: sigmoid is better
                                      padding='same')(x)

  # extract empty from input
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[3, 0, 1, 2])
  empty_only_matrix, occupied_only_matrix = empty_only_matrix[0], empty_only_matrix[1]
  unknown_only_matrix = tf.subtract(1.0, empty_only_matrix + occupied_only_matrix)
  unknown_only_matrix = tf.expand_dims(unknown_only_matrix, -1)
  occupied_only_matrix = tf.expand_dims(occupied_only_matrix, -1)

  outputs = x * unknown_only_matrix + occupied_only_matrix # and multiply it by output

  model = tf.keras.models.Model(inputs, outputs, name='get_autocomplete_2d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def get_2d_model(sensor_range_voxels, input_shape, output_expand_layers):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [8*(2**k) for k in range(0, num_layers)]
  print("get_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))
  expand_layer_filters = [4*(2**k) for k in range(0, output_expand_layers)]
  print("get_2d_model: expand layer filters are %d with layers %s:" %
        (int(output_expand_layers), str(expand_layer_filters)))

  # this generates a mask with the central part of each pixel set to zero
  # like this: [[1 1 1 1] [1 0 0 1] [1 0 0 1] [1 1 1 1]] for each pixel
  output_mask = np.zeros(shape=(2**output_expand_layers, 2**output_expand_layers))
  for x in range(0, 2**output_expand_layers):
    for y in range(0, 2**output_expand_layers):
      if (x == 0 or y == 0 or x == (2**output_expand_layers - 1) or y == (2**output_expand_layers - 1)):
        output_mask[x, y] = 1.0
  output_mask = tf.keras.backend.constant(output_mask, dtype=tf.float32)
  output_mask = tf.tile(output_mask, input_shape[0:2])
  output_mask = tf.expand_dims(output_mask, -1)

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some obscure reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  reverse_expand_layer_filters = reversed(expand_layer_filters)
  for filters in reverse_expand_layer_filters:
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  x = tf.keras.layers.Conv2DTranspose(filters=1,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid',
                                      padding='same')(x)

  # extract empty from input
  # this masks output so that only empty cells are relevant (where the sensor may be placed)
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[3, 0, 1, 2])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.UpSampling2D(size=2**output_expand_layers)(empty_only_matrix)

  outputs = x * empty_only_matrix * output_mask

  model = tf.keras.models.Model(inputs, outputs, name='nbv_2d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def is_in_outer_circle(x, y, outer_circle_side):
  return (x == outer_circle_side - 1 or
          y == outer_circle_side - 1 or
          x == 0 or y == 0)
  pass

def get_flat_2d_model(sensor_range_voxels, input_shape, output_expand_layers):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [8*(2**k) for k in range(0, num_layers)]
  print("get_flat_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))
  outer_circle_side = 2**output_expand_layers
  flat_output_channels = (outer_circle_side - 1) * 4
  print("get_flat_2d_model: flat output channels are %d" %
        (int(flat_output_channels)))

  # this generates a mask with the central part of each pixel set to zero
  # like this: [[1 1 1 1] [1 0 0 1] [1 0 0 1] [1 1 1 1]] for each pixel
  output_mask = np.zeros(shape=(2**output_expand_layers, 2**output_expand_layers))
  for x in range(0, 2**output_expand_layers):
    for y in range(0, 2**output_expand_layers):
      if (x == 0 or y == 0 or x == (2**output_expand_layers - 1) or y == (2**output_expand_layers - 1)):
        output_mask[x, y] = 1.0
  output_mask = tf.keras.backend.constant(output_mask, dtype=tf.float32)
  output_mask = tf.tile(output_mask, input_shape[0:2])
  output_mask = tf.expand_dims(output_mask, -1)

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  image_width = input_shape[1]
  image_height = input_shape[0]

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some obscure reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  x = tf.keras.layers.Conv2DTranspose(filters=flat_output_channels,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid',
                                      padding='same')(x)

  # extract empty from input
  # this masks output so that only empty cells are relevant (where the sensor may be placed)
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[3, 0, 1, 2])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.UpSampling2D(size=2**output_expand_layers)(empty_only_matrix)

  # interleave to create the final image
  if True:
    x = tf.transpose(x, perm=[3, 0, 1, 2])
    x_counter = 0

    mat = []
    for a in range(0, outer_circle_side):
      new_list = []
      for b in range(0, outer_circle_side):
        new_list.append(None)
      mat.append(new_list)
      pass

    for ocy in range(0, outer_circle_side):
      for ocx in range(0, outer_circle_side):
        if (not is_in_outer_circle(ocx, ocy, outer_circle_side)):
          continue
        mat[ocy][ocx] = x[x_counter]
        mat[ocy][ocx] = tf.expand_dims(mat[ocy][ocx], -1)
        x_counter += 1

    zero_image = tf.fill(tf.shape(mat[0][0]), 0.0)
    for ocy in range(0, outer_circle_side):
      for ocx in range(0, outer_circle_side):
        if (not is_in_outer_circle(ocx, ocy, outer_circle_side)):
          mat[ocy][ocx] = zero_image

    for ocy in range(0, outer_circle_side):
      concatenate_name = "fin_concatenate_y" + str(ocy)
      mat[ocy] = tf.keras.layers.concatenate(mat[ocy], axis=3, name=concatenate_name)
      mat[ocy] = tf.reshape(mat[ocy], [-1, image_height, image_width * outer_circle_side, 1])
      mat[ocy] = tf.transpose(mat[ocy], perm=[0, 2, 1, 3])
    concatenate_name = "fin_concatenate_last"
    mat = tf.keras.layers.concatenate(mat, axis=3, name=concatenate_name)
    mat = tf.reshape(mat, [-1, image_width * outer_circle_side, image_height * outer_circle_side, 1])
    x = tf.transpose(mat, perm=[0, 2, 1, 3])
    pass

  outputs = x * empty_only_matrix * output_mask

  model = tf.keras.models.Model(inputs, outputs, name='nbv_flat_2d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def get_next_in_outer_circle(y, x, outer_circle_side):
  ocx = x;
  ocy = y;

  maxxy = outer_circle_side - 1;

  if (ocy == 0 and ocx < maxxy):
    return (ocy, ocx + 1)

  if (ocx == maxxy and ocy < maxxy):
    return (ocy + 1, ocx)

  if (ocy == maxxy and ocx > 0):
    return (ocy, ocx - 1)

  if (ocx == 0 and ocy > 0):
    return (ocy - 1, ocx)

  pass

def get_prev_in_outer_circle(y, x, outer_circle_side):
  ocx = x;
  ocy = y;

  maxxy = outer_circle_side - 1;

  if (ocy == 0 and ocx > 0):
    return (ocy, ocx - 1)

  if (ocx == 0 and ocy < maxxy):
    return (ocy + 1, ocx)

  if (ocy == maxxy and ocx < maxxy):
    return (ocy, ocx + 1)

  if (ocx == maxxy and ocy > 0):
    return (ocy - 1, ocx)
  pass

def get_circular_2d_model(sensor_range_voxels, input_shape, output_expand_layers):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [2*(2**k) for k in range(0, num_layers)]
  print("get_circular_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))
  print("get_circular_2d_model: input shape: " + str(input_shape))

  outer_circle_side = 2**output_expand_layers
  outer_circle_center = outer_circle_side / 2.0 - 0.5
  image_width = input_shape[1]
  image_height = input_shape[0]

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')

  x = []
  for a in range(0, outer_circle_side):
    new_list = []
    for b in range(0, outer_circle_side):
      new_list.append(None)
    x.append(new_list)
    pass
  skip_connections_padding = []

  for ocy in range(0, outer_circle_side):
    for ocx in range(0, outer_circle_side):
      if (is_in_outer_circle(ocx, ocy, outer_circle_side)):
        x[ocy][ocx] = inputs

  for filter_i, filters in enumerate(layer_filters):

    predicted_padding = [1 - (x[0][0].get_shape().as_list()[1] % 2), 1 - (x[0][0].get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)

    px = x
    px = [list(i) for i in px]

    shared_conv2d_1 = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=1,
                                             activation='linear',
                                             padding='same',
                                             name=('conv2d_1_' + str(filter_i)))
    shared_conv2d_2 = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=2,
                                             activation='linear',
                                             padding='same',
                                             name=('conv2d_2_' + str(filter_i)))

    for ocy in range(0, outer_circle_side):
      for ocx in range(0, outer_circle_side):
        if (is_in_outer_circle(ocx, ocy, outer_circle_side)):
          bearing_angle = math.atan2(ocy - outer_circle_center, ocx - outer_circle_center)
          if (bearing_angle < 0.0):
            bearing_angle += 2.0 * math.pi
          bearing_angle_i = int(round(bearing_angle / (math.pi / 2.0)))
          rotation_name_suffix = str(filter_i) + '_' + str(ocx) + '_' + str(ocy) + '_' + str(bearing_angle_i * 90) + 'deg'

          next_ocy, next_ocx = get_next_in_outer_circle(ocy, ocx, outer_circle_side)
          prev_ocy, prev_ocx = get_prev_in_outer_circle(ocy, ocx, outer_circle_side)

          concatenate_name = "concatenate_f" + str(filter_i) + "_x" + str(ocx) + "_y" + str(ocy)
          x[ocy][ocx] = tf.keras.layers.concatenate([px[prev_ocy][prev_ocx], px[ocy][ocx], px[next_ocy][next_ocx]],
                                                    axis=3, name=concatenate_name)

          # clockwise
          x[ocy][ocx] = tf.image.rot90(x[ocy][ocx], k=bearing_angle_i, name=('conv_rot' + rotation_name_suffix))
          x[ocy][ocx] = shared_conv2d_1(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.BatchNormalization()(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.LeakyReLU()(x[ocy][ocx])
          x[ocy][ocx] = shared_conv2d_2(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.BatchNormalization()(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.LeakyReLU()(x[ocy][ocx])
          x[ocy][ocx] = tf.image.rot90(x[ocy][ocx], k=(4 - bearing_angle_i), name=('conv_derot' + rotation_name_suffix))
          pass
        pass
      pass
    pass

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]

    px = x
    px = [list(l) for l in px]

    shared_deconv2d = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                      kernel_size=kernel_size,
                                                      strides=2,
                                                      output_padding=output_padding,
                                                      activation='linear',
                                                      padding='same',
                                                      name=('deconv2d_' + str(i)))

    for ocy in range(0, outer_circle_side):
      for ocx in range(0, outer_circle_side):
        if (is_in_outer_circle(ocx, ocy, outer_circle_side)):
          bearing_angle = math.atan2(ocy - outer_circle_center, ocx - outer_circle_center)
          if (bearing_angle < 0.0):
            bearing_angle += 2.0 * math.pi
          bearing_angle_i = int(round(bearing_angle / (math.pi / 2.0)))
          rotation_name_suffix = str(i) + '_' + str(ocx) + '_' + str(ocy) + '_' + str(bearing_angle_i * 90) + 'deg'

          next_ocy, next_ocx = get_next_in_outer_circle(ocy, ocx, outer_circle_side)
          prev_ocy, prev_ocx = get_prev_in_outer_circle(ocy, ocx, outer_circle_side)

          concatenate_name = "de_concatenate_f" + str(i) + "_x" + str(ocx) + "_y" + str(ocy)
          x[ocy][ocx] = tf.keras.layers.concatenate([px[prev_ocy][prev_ocx], px[ocy][ocx], px[next_ocy][next_ocx]],
                                                    axis=3, name=concatenate_name)

          x[ocy][ocx] = tf.image.rot90(x[ocy][ocx], k=bearing_angle_i, name=('deconv_rot' + rotation_name_suffix))
          x[ocy][ocx] = shared_deconv2d(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.BatchNormalization()(x[ocy][ocx])
          x[ocy][ocx] = tf.keras.layers.LeakyReLU()(x[ocy][ocx])
          x[ocy][ocx] = tf.image.rot90(x[ocy][ocx], k=(4 - bearing_angle_i), name=('deconv_derot' + rotation_name_suffix))
          pass
        pass
      pass
    pass

  for ocy in range(0, outer_circle_side):
    for ocx in range(0, outer_circle_side):
      if (is_in_outer_circle(ocx, ocy, outer_circle_side)):
        x[ocy][ocx] = tf.keras.layers.Conv2DTranspose(filters=1,
                                                      kernel_size=1,
                                                      strides=1,
                                                      #activation='linear',
                                                      activation='sigmoid',
                                                      padding='same')(x[ocy][ocx])
        pass
      pass
    pass

  zero_image = tf.fill(tf.shape(x[0][0]), 0.0)
  for ocy in range(0, outer_circle_side):
    for ocx in range(0, outer_circle_side):
      if (not is_in_outer_circle(ocx, ocy, outer_circle_side)):
        x[ocy][ocx] = zero_image

  # interleave to create the final image
  if True:
    for ocy in range(0, outer_circle_side):
      concatenate_name = "fin_concatenate_y" + str(ocy)
      x[ocy] = tf.keras.layers.concatenate(x[ocy], axis=3, name=concatenate_name)
      x[ocy] = tf.reshape(x[ocy], [-1, image_height, image_width * outer_circle_side, 1])
      x[ocy] = tf.transpose(x[ocy], perm=[0, 2, 1, 3])
    concatenate_name = "fin_concatenate_last"
    x = tf.keras.layers.concatenate(x, axis=3, name=concatenate_name)
    x = tf.reshape(x, [-1, image_width * outer_circle_side, image_height * outer_circle_side, 1])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    pass

  # extract empty from input
  # this masks output so that only empty cells are relevant (where the sensor may be placed)
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[3, 0, 1, 2])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.UpSampling2D(size=2**output_expand_layers)(empty_only_matrix)

  outputs = x * empty_only_matrix

  model = tf.keras.models.Model(inputs, outputs, name='nbv_2d_model_circular')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model
