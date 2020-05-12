# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle

class Model():
    def __init__(self, weight_file_path):
      """Overlay bounding boxes of face on images.
        Args:
          weight_file_path: 
              A pretrained weight file in the pickle format 
              generated by matconvnet_hr101_to_tf.py.
        Returns:
          None.
      """
      self.dtype = tf.float32
      self.weight_file_path = weight_file_path
      with open(self.weight_file_path, "rb") as f:
        self.mat_blocks_dict, self.mat_params_dict = pickle.load(f,encoding='latin1')

    def get_data_by_key(self, key):
      """Helper to access a pretrained model data through a key."""
      assert key in self.mat_params_dict, "key: " + key + " not found."
      return self.mat_params_dict[key]

    def _weight_variable_on_cpu(self, name, shape):
      """Helper to create a weight Variable stored on CPU memory.

      Args:
        name: name of the variable.
        shape: list of ints: (height, width, channel, filter).

      Returns:
        initializer for Variable.
      """
      assert len(shape) == 4

      weights = self.get_data_by_key(name + "_filter")  # (h, w, channel, filter)
      assert list(weights.shape) == shape
      initializer = tf.constant_initializer(weights)

      with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name + "_w", shape, initializer=initializer, dtype=self.dtype)
      return var

    def _bias_variable_on_cpu(self, name, shape):
      """Helper to create a bias Variable stored on CPU memory.

      Args:
        name: name of the variable.
        shape: int, filter size.

      Returns:
        initializer for Variable.
      """
      assert isinstance(shape, int)
      bias = self.get_data_by_key(name + "_bias")
      assert len(bias) == shape
      initializer = tf.constant_initializer(bias)

      with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name + "_b", shape, initializer=initializer, dtype=self.dtype)
      return var


    def _bn_variable_on_cpu(self, name, shape):
      """Helper to create a batch normalization Variable stored on CPU memory.

      Args:
        name: name of the variable.
        shape: int, filter size.

      Returns:
        initializer for Variable.
      """
      assert isinstance(shape, int)

      name2 = "bn" + name[3:]
      if name.startswith("conv"):
        name2 = "bn_" + name

      scale = self.get_data_by_key(name2 + '_scale')
      offset = self.get_data_by_key(name2 + '_offset')
      mean = self.get_data_by_key(name2 + '_mean')
      variance = self.get_data_by_key(name2 + '_variance')

      with tf.device('/cpu:0'):
        initializer = tf.constant_initializer(scale)
        scale = tf.compat.v1.get_variable(name2 + "_scale", shape, initializer=initializer, dtype=self.dtype)
        initializer = tf.constant_initializer(offset)
        offset = tf.compat.v1.get_variable(name2 + "_offset", shape, initializer=initializer, dtype=self.dtype)
        initializer = tf.constant_initializer(mean)
        mean = tf.compat.v1.get_variable(name2 + "_mean", shape, initializer=initializer, dtype=self.dtype)
        initializer = tf.constant_initializer(variance)
        variance = tf.compat.v1.get_variable(name2 + "_variance", shape, initializer=initializer, dtype=self.dtype)

      return scale, offset, mean, variance


    def conv_block(self, bottom, name, shape, strides=[1,1,1,1], padding="SAME",
                   has_bias=False, add_relu=True, add_bn=True, eps=1.0e-5):
      """Create a block composed of multiple layers:
            a conv layer
            a batch normalization layer
            an activation layer

      Args:
        bottom: A layer before this block.
        name: Name of the block.
        shape: List of ints: (height, width, channel, filter).
        strides: Strides of conv layer.
        padding: Padding of conv layer.
        has_bias: Whether a bias term is added.
        add_relu: Whether a ReLU layer is added.
        add_bn: Whether a batch normalization layer is added.
        eps: A small float number to avoid dividing by 0, used in a batch normalization layer.
      Returns:
        a block of layers
      """
      assert len(shape) == 4

      weight = self._weight_variable_on_cpu(name, shape)
      conv = tf.nn.conv2d(bottom, weight, strides, padding=padding)
      if has_bias:
        bias = self._bias_variable_on_cpu(name, shape[3])

      pre_activation = tf.nn.bias_add(conv, bias) if has_bias else conv

      if add_bn:
        # scale, offset, mean, variance = self._bn_variable_on_cpu("bn_" + name, shape[-1])
        scale, offset, mean, variance = self._bn_variable_on_cpu(name, shape[-1])
        pre_activation = tf.nn.batch_normalization(pre_activation, mean, variance, offset, scale, variance_epsilon=eps)

      relu = tf.nn.relu(pre_activation) if add_relu else pre_activation

      return relu


    def conv_trans_layer(self, bottom, name, shape, strides=[1,1,1,1], padding="SAME", has_bias=False):
      """Create a block composed of multiple layers:
            a transpose of conv layer
            an activation layer

      Args:
        bottom: A layer before this block.
        name: Name of the block.
        shape: List of ints: (height, width, channel, filter).
        strides: Strides of conv layer.
        padding: Padding of conv layer.
        has_bias: Whether a bias term is added.
        add_relu: Whether a ReLU layer is added.
      Returns:
        a block of layers
      """
      assert len(shape) == 4

      weight = self._weight_variable_on_cpu(name, shape)
      nb, h, w, nc = tf.split(tf.shape(bottom), num_or_size_splits=4)
      output_shape = tf.stack([nb, (h - 1) * strides[1] - 3 + shape[0], (w - 1) * strides[2] - 3 + shape[1], nc])[:, 0]
      conv = tf.nn.conv2d_transpose(bottom, weight, output_shape, strides, padding=padding)
      if has_bias:
        bias = self._bias_variable_on_cpu(name, shape[3])

      conv = tf.nn.bias_add(conv, bias) if has_bias else conv

      return conv

    def residual_block(self, bottom, name, in_channel, neck_channel, out_channel, trunk):
      """Create a residual block.

      Args:
        bottom: A layer before this block.
        name: Name of the block.
        in_channel: number of channels in a input tensor.
        neck_channel: number of channels in a bottleneck block.
        out_channel: number of channels in a output tensor.
        trunk: a tensor in a identity path.
      Returns:
        a block of layers
      """
      _strides = [1, 2, 2, 1] if name.startswith("res3a") or name.startswith("res4a") else [1, 1, 1, 1]
      res = self.conv_block(bottom, name + '_branch2a', shape=[1, 1, in_channel, neck_channel],
                            strides=_strides, padding="VALID", add_relu=True)
      res = self.conv_block(res, name + '_branch2b', shape=[3, 3, neck_channel, neck_channel],
                            padding="SAME", add_relu=True)
      res = self.conv_block(res, name + '_branch2c', shape=[1, 1, neck_channel, out_channel],
                            padding="VALID", add_relu=False)

      res = trunk + res
      res = tf.nn.relu(res)

      return res

    def tiny_face(self, image):
        """Create a tiny face model.
  
        Args:
          image: an input image.
        Returns:
          a score tensor
        """
        img = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        conv = self.conv_block(img, 'conv1', shape=[7, 7, 3, 64], strides=[1, 2, 2, 1], padding="VALID", add_relu=True)
        pool1 = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        res2a_branch1 = self.conv_block(pool1, 'res2a_branch1', shape=[1, 1, 64, 256], padding="VALID", add_relu=False)
        res2a = self.residual_block(pool1, 'res2a', 64, 64, 256, res2a_branch1)
        res2b = self.residual_block(res2a, 'res2b', 256, 64, 256, res2a)
        res2c = self.residual_block(res2b, 'res2c', 256, 64, 256, res2b)

        res3a_branch1 = self.conv_block(res2c, 'res3a_branch1', shape=[1, 1, 256, 512], strides=[1, 2, 2, 1], padding="VALID", add_relu=False)
        res3a = self.residual_block(res2c, 'res3a', 256, 128, 512, res3a_branch1)

        res3b1 = self.residual_block(res3a, 'res3b1', 512, 128, 512, res3a)
        res3b2 = self.residual_block(res3b1, 'res3b2', 512, 128, 512, res3b1)
        res3b3 = self.residual_block(res3b2, 'res3b3', 512, 128, 512, res3b2)

        res4a_branch1 = self.conv_block(res3b3, 'res4a_branch1', shape=[1, 1, 512, 1024], strides=[1, 2, 2, 1], padding="VALID", add_relu=False)
        res4a = self.residual_block(res3b3, 'res4a', 512, 256, 1024, res4a_branch1)

        res4b = res4a
        for i in range(1, 23):
          res4b = self.residual_block(res4b, 'res4b' + str(i), 1024, 256, 1024, res4b)

        score_res4 = self.conv_block(res4b, 'score_res4', shape=[1, 1, 1024, 125], padding="VALID",
                                     has_bias=True, add_relu=False, add_bn=False)
        score4 = self.conv_trans_layer(score_res4, 'score4', shape=[4, 4, 125, 125], strides=[1, 2, 2, 1], padding="SAME")
        score_res3 = self.conv_block(res3b3, 'score_res3', shape=[1, 1, 512, 125], padding="VALID",
                                     has_bias=True, add_bn=False, add_relu=False)

        bs, height, width = tf.split(tf.shape(score4), num_or_size_splits=4)[0:3]
        _size = tf.convert_to_tensor([height[0], width[0]])
        _offsets = tf.zeros([bs[0], 2])
        score_res3c = tf.image.extract_glimpse(score_res3, _size, _offsets, centered=True, normalized=False)

        score_final = score4 + score_res3c
        return score_final
