import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow.keras.utils as conv_utils

GROUP_NUM = 1

class MixNetConvInitializer(initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(MixNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random.normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class Swish(layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)

import tensorflow as tf
from tensorflow.keras.layers import Layer, DepthwiseConv2D, Conv2D, BatchNormalization, ReLU

class MixNetIBNBlock(Layer):
    def __init__(self, input_filters, output_filters, expand_ratio, stride, kernel_size):
        super(MixNetIBNBlock, self).__init__()
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.stride = stride

        self.use_res_connect = self.stride == 1 and self.input_filters == self.output_filters

        self.expand_conv = Conv2D(filters=self.input_filters * self.expand_ratio,
                                  kernel_size=1,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=MixNetConvInitializer(),
                                  use_bias=False)
        self.expand_bn = BatchNormalization()

        self.depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size,
                                              strides=self.stride,
                                              padding='same',
                                              kernel_initializer=MixNetConvInitializer(),
                                              use_bias=False)
        self.depthwise_bn = BatchNormalization()

        self.project_conv = Conv2D(filters=self.output_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=MixNetConvInitializer(),
                                   use_bias=False)
        self.project_bn = BatchNormalization()

        self.relu = ReLU()

    def call(self, inputs, training=None):
        x = inputs

        # Expand
        x = self.expand_conv(x)
        x = self.expand_bn(x, training=training)
        x = self.relu(x)

        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.relu(x)

        # Project
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        if self.use_res_connect:
            x = tf.keras.layers.add([x, inputs])

        return x


class GroupIBN(layers.Layer):

    def __init__(self, filters, kernels, groups,
                 type='conv', conv_kwargs=None,
                 **kwargs):
        super(GroupIBN, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }

        self.filters = filters
        self.kernels = kernels
        self.groups = groups
        self.type = type
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        splits = self._split_channels(filters, self.groups) #splits is filter_num
        self._channel_axis = -1
        self._layers = []
        for i in range(groups):
            self._layers.append(MixNetIBNBlock(splits[i], splits[i], expand_ratio=1, stride=self.strides, kernel_size=kernels[i]))
        self.data_format = 'channels_last'

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        splits = self._split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        #x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
        x_outputs = []
        for x, c in zip(x_splits, self._layers):
            output = c(x)
            x_outputs.append(output)
        x = layers.concatenate(x_outputs, axis=self._channel_axis)
        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                filter_size=1,
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'type': self.type,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(GroupIBN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(depth_multiplier) if depth_multiplier is not None else None
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_repeats(repeats):
    """Round number of filters based on depth multiplier."""
    return int(repeats)


# Ontained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
class GroupedIBN2D(object):
    """Groupped convolution.
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel_size size.
    """

    def __init__(self, filters, kernel_size, type, **kwargs):
        """Initialize the layer.
        Args:
          filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel_size for each group.
          **kwargs: other parameters passed to the original conv2d layer.
        """

        global GROUP_NUM
        self._groups = len(kernel_size)
        self._channel_axis = -1
        self.filters = filters
        self.kernels = kernel_size
        self.type = type

        self._conv_kwargs = {
            'strides': kwargs.get('strides', (1, 1)),
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
            'type': kwargs.get('type', False),
        }

        GROUP_NUM += 1

    def __call__(self, inputs):
        grouped_op = GroupIBN(self.filters, self.kernels, groups=self._groups,
                                      type=self.type, conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


def _inverted_res_block(inputs, stride, filters, prefix, kernel_size):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    alpha = 1.0
    expansion = 1.0
    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = pointwise_conv_filters
    x = inputs
    prefix = 'block_{}_'.format(prefix)
    x = layers.Conv2D(expansion * in_channels,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      kernel_initializer=MixNetConvInitializer(),
                      name=prefix + 'expand')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'expand_BN')(x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               kernel_initializer=MixNetConvInitializer(),
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x
