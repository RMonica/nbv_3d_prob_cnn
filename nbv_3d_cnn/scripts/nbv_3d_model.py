#!/usr/bin/python2

import tensorflow as tf

import numpy as np
import math
import sys

def get_quat_3d_model(sensor_range_voxels, input_shape):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [16*(2**k) for k in range(0, num_layers)]
  print("get_3d_model: creating network with %d layers: %s for range %f" %
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
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2),
                         1 - (x.get_shape().as_list()[2] % 2),
                         1 - (x.get_shape().as_list()[3] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=4)

  x = tf.keras.layers.Conv3D(filters=4,
                             kernel_size=1,
                             strides=1,
                             activation='linear',
                             #activation='sigmoid',
                             padding='same')(x)

  # extract empty from input
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[4, 0, 1, 2, 3])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.concatenate([empty_only_matrix, empty_only_matrix,
                                                   empty_only_matrix, empty_only_matrix], axis=4)

  outputs = x * empty_only_matrix # and multiply it by output

  model = tf.keras.models.Model(inputs, outputs, name='quat_3d_model')

  model.compile(loss='mse', optimizer='adam')

  return model

def get_autocomplete_3d_model(sensor_range_voxels, input_shape):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [16*(2**k) for k in range(0, num_layers)]
  print("get_autocomplete_3d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv3Dtranspose below
    #or the output size will differ
    #for some obscure reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2),
                         1 - (x.get_shape().as_list()[3] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=4)

  x = tf.keras.layers.Conv3DTranspose(filters=1,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid', # tested: sigmoid is better
                                      padding='same')(x)

  # extract empty from input
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[4, 0, 1, 2, 3])
  empty_only_matrix, occupied_only_matrix = empty_only_matrix[0], empty_only_matrix[1]
  unknown_only_matrix = tf.subtract(1.0, empty_only_matrix + occupied_only_matrix)
  unknown_only_matrix = tf.expand_dims(unknown_only_matrix, -1)
  occupied_only_matrix = tf.expand_dims(occupied_only_matrix, -1)

  outputs = x * unknown_only_matrix + occupied_only_matrix # and multiply it by output

  model = tf.keras.models.Model(inputs, outputs, name='get_autocomplete_3d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def get_3d_model(sensor_range_voxels, input_shape, output_expand_layers):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [16*(2**k) for k in range(0, num_layers)]
  print("get_3d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))
  expand_layer_filters = [4*(2**k) for k in range(0, output_expand_layers)]
  print("get_3d_model: expand layer filters are %d with layers %s:" %
        (int(output_expand_layers), str(expand_layer_filters)))

  # this generates a mask with the central part of each pixel set to zero
  # like this: [[1 1 1 1] [1 0 0 1] [1 0 0 1] [1 1 1 1]] for each pixel
  output_mask = np.zeros(shape=(2**output_expand_layers, 2**output_expand_layers, 2**output_expand_layers))
  for x in range(0, 2**output_expand_layers):
    for y in range(0, 2**output_expand_layers):
      for z in range(0, 2**output_expand_layers):
        if (x == 0 or y == 0 or z == 0 or
            x == (2**output_expand_layers - 1) or y == (2**output_expand_layers - 1) or z == (2**output_expand_layers - 1)):
          output_mask[x][y][z] = 1.0
  output_mask = tf.keras.backend.constant(output_mask, dtype=tf.float32)
  output_mask = tf.tile(output_mask, input_shape[0:3])
  output_mask = tf.expand_dims(output_mask, -1)

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some unknown reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2),
                         1 - (x.get_shape().as_list()[2] % 2),
                         1 - (x.get_shape().as_list()[3] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=4)

  reverse_expand_layer_filters = reversed(expand_layer_filters)
  for filters in reverse_expand_layer_filters:
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  x = tf.keras.layers.Conv3DTranspose(filters=1,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid',
                                      padding='same')(x)

  # extract empty from input
  # this masks output so that only empty cells are relevant (where the sensor may be placed)
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[4, 0, 1, 2, 3])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.UpSampling3D(size=2**output_expand_layers)(empty_only_matrix)

  outputs = x * empty_only_matrix * output_mask

  model = tf.keras.models.Model(inputs, outputs, name='nbv_3d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def get_flat_3d_model(sensor_range_voxels, input_shape, flat_output_channels):

  kernel_size = 3

  sensor_range = sensor_range_voxels
  num_layers = int(math.floor(math.log(sensor_range, 2))) - 1
  layer_filters = [16*(2**k) for k in range(0, num_layers)]
  print("get_flat_2d_model: creating network with %d layers: %s for range %f" %
        (int(num_layers), str(layer_filters), float(sensor_range)))
  print("get_flat_2d_model: flat output channels are %d" %
        (int(flat_output_channels)))

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
    #for some unknown reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2),
                         1 - (x.get_shape().as_list()[2] % 2),
                         1 - (x.get_shape().as_list()[3] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=4)

  x = tf.keras.layers.Conv3DTranspose(filters=flat_output_channels,
                                      kernel_size=1,
                                      strides=1,
                                      #activation='linear',
                                      activation='sigmoid',
                                      padding='same')(x)

  # extract empty from input
  # this masks output so that only empty cells are relevant (where the sensor may be placed)
  empty_only_matrix = inputs
  empty_only_matrix = tf.transpose(empty_only_matrix, perm=[4, 0, 1, 2, 3])
  empty_only_matrix = empty_only_matrix[0]
  empty_only_matrix = tf.expand_dims(empty_only_matrix, -1)
  empty_only_matrix = tf.keras.layers.concatenate([empty_only_matrix, ] * flat_output_channels, axis=4)

  outputs = x * empty_only_matrix

  model = tf.keras.models.Model(inputs, outputs, name='nbv_flat_3d_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model
