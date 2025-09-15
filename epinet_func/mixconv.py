import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow.keras.utils as conv_utils
import random, string

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


class GroupConvolution(layers.Layer):

    def __init__(self, filters, kernels, groups,
                 type='conv', split_type='equ', conv_kwargs=None,
                 **kwargs):
        super(GroupConvolution, self).__init__(**kwargs)

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
        self.split_type = split_type
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}
        res = ''.join(random.choices(string.ascii_lowercase +
                                     string.digits, k=8))
        self.nme = conv_kwargs.get('name', res)

        assert type in ['conv', 'depthwise_conv', 'dwsc']
        if self.type == 'conv':
            if split_type == "equ":
                splits = self._split_channels(filters, self.groups)
            else:
                splits = self._exp_split_channels(filters, self.groups)
            self._layers = [layers.Conv2D(splits[i], kernels[i],
                                          strides=self.strides,
                                          padding=self.padding,
                                          dilation_rate=self.dilation_rate,
                                          use_bias=self.use_bias,
                                          kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        elif self.type == 'depthwise_conv':
            self._layers = [layers.DepthwiseConv2D(kernels[i],
                                                   strides=self.strides,
                                                   padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]
        else:
            if self.split_type == "equ":
                splits = self._split_channels(filters, self.groups)
            else:
                splits = self._exp_split_channels(filters, self.groups)
            self._layers = [layers.SeparableConv2D(splits[i], kernels[i],
                                                   strides=self.strides,
                                                   padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]
        self.data_format = 'channels_last'
        self._channel_axis = -1

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        if self.split_type == "equ":
            splits = self._split_channels(filters, self.groups)
        else:
            splits = self._exp_split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
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

    def _exp_split_channels(self, total_filters, num_groups):
        scaling_factor = 2 ** (1 / num_groups)
        relative_scaling_factors = [scaling_factor ** i for i in range(num_groups)]
        scaling_factor_sum = sum(relative_scaling_factors)
        split = [int(round(total_filters * s / scaling_factor_sum)) for s in relative_scaling_factors]
        remaining_frame = sum(split) - total_filters
        if remaining_frame > 0:
            if split[0] - remaining_frame > 1:
                split[0] = split[0] - remaining_frame
        elif remaining_frame < 0:
            split[-1] = split[-1] - remaining_frame
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
        base_config = super(GroupConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
# def _split_channels(total_filters, num_groups):
#    split = [total_filters // num_groups for _ in range(num_groups)]
#    split[0] += total_filters - sum(split)
#    return split


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


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
class GroupedConv2D(object):
    """Groupped convolution.
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel_size size.
    """

    def __init__(self, filters, kernel_size, type, split_type, **kwargs):
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
        self.split_type = split_type

        self._conv_kwargs = {
            'strides': kwargs.get('strides', (1, 1)),
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }



        GROUP_NUM += 1

    def __call__(self, inputs):
        grouped_op = GroupConvolution(self.filters, self.kernels, groups=self._groups,
                                      type=self.type, split_type=self.split_type, conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x
