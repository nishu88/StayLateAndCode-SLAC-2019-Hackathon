from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def _next_power_of_two(x):
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  elif preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):

  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'tiny_conv':
    return create_tiny_conv_model(fingerprint_input, model_settings,
                                  is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, "low_latency_svdf",' +
                    ' or "tiny_conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):

  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.get_variable(
      name='weights',
      initializer=tf.truncated_normal_initializer(stddev=0.001),
      shape=[fingerprint_size, label_count])
  bias = tf.get_variable(
      name='bias', initializer=tf.zeros_initializer, shape=[label_count])
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable(
      name='second_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.get_variable(
      name='second_bias',
      initializer=tf.zeros_initializer,
      shape=[second_filter_count])
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.get_variable(
      name='first_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_conv_element_count, first_fc_output_channels])
  first_fc_bias = tf.get_variable(
      name='first_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.get_variable(
      name='second_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.get_variable(
      name='second_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  rank = 2
  num_units = 1280
  num_filters = rank * num_units
  batch = 1
  memory = tf.get_variable(
      initializer=tf.zeros_initializer,
      shape=[num_filters, batch, input_time_size],
      trainable=False,
      name='runtime-memory')
  first_time_flag = tf.get_variable(
      name="first_time_flag",
      dtype=tf.int32,
      initializer=1)
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(first_time_flag, 1),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  first_time_flag = 0
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  weights_frequency = tf.get_variable(
      name='weights_frequency',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[input_frequency_size, num_filters])
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  if not is_training:
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  weights_time = tf.get_variable(
      name='weights_time',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[num_filters, input_time_size])
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  units_output = tf.reduce_sum(outputs, axis=1)
  units_output = tf.transpose(units_output)

  bias = tf.get_variable(
      name='bias', initializer=tf.zeros_initializer, shape=[num_units])
  first_bias = tf.nn.bias_add(units_output, bias)

  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.get_variable(
      name='first_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[num_units, first_fc_output_channels])
  first_fc_bias = tf.get_variable(
      name='first_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.get_variable(
      name='second_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.get_variable(
      name='second_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_tiny_conv_model(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 10
  first_filter_count = 8
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv_stride_x = 2
  first_conv_stride_y = 2
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                            [1, first_conv_stride_y, first_conv_stride_x, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_dropout_shape = first_dropout.get_shape()
  first_dropout_output_width = first_dropout_shape[2]
  first_dropout_output_height = first_dropout_shape[1]
  first_dropout_element_count = int(
      first_dropout_output_width * first_dropout_output_height *
      first_filter_count)
  flattened_first_dropout = tf.reshape(first_dropout,
                                       [-1, first_dropout_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_dropout_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = (
      tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
