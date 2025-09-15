import tensorflow as tf
from tensorflow.keras.layers import Layer, DepthwiseConv2D, BatchNormalization, ReLU, Conv2D, Add, Lambda

MIN_VAL = -1.0
MAX_VAL = 1.0

class FECellA(Layer):
    def __init__(self, out_filter, res, prefix):
        super(FECellA, self).__init__()
        self.prefix = prefix
        self._add = None
        self._pconv3 = None
        self._pconv2 = None
        self._pconv1 = None
        self._dconv2x2_1 = None
        self._dconv2x2_2 = None
        self._dconv2x2_3 = None
        self._dconv3x3_8 = None
        self._dconv3x3_7 = None
        self._dconv3x3_6 = None
        self._dconv3x3_5 = None
        self._dconv3x3_4 = None
        self._dconv3x3_3 = None
        self._dconv3x3_2 = None
        self._dconv3x3_1 = None
        self.out_filter = out_filter
        self.res = res
        self.strides = 1
        self.padding = 'same'
        self.dilation_rate = 1
        self.use_bias = True

    def build(self, input_shape):
        # Define layers here using input_shape
        self._dconv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_1'),
            BatchNormalization(name=self.prefix + '_dconv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # seq.add(Activation('relu', name=prefix + '_act'))
            # ReLU(name=self.prefix + '_dconv3x3_1_af')
        ])

        self._dconv3x3_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_2'),
            BatchNormalization(name=self.prefix + '_dconv3x3_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_2_af')
        ])

        self._dconv3x3_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_3'),
            BatchNormalization(name=self.prefix + '_dconv3x3_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_3_af')
        ])

        self._dconv3x3_4 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_4'),
            BatchNormalization(name=self.prefix + '_dconv3x3_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_4_af')
        ])

        self._dconv3x3_5 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_5'),
            BatchNormalization(name=self.prefix + '_dconv3x3_5_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_5_af')
        ])

        self._dconv3x3_6 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_6'),
            BatchNormalization(name=self.prefix + '_dconv3x3_6_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_6_af')
        ])

        self._dconv3x3_7 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_7'),
            BatchNormalization(name=self.prefix + '_dconv3x3_7_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_7_af')
        ])

        self._dconv3x3_8 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_8'),
            BatchNormalization(name=self.prefix + '_dconv3x3_8_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_8_af')
        ])

        self._dconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        self._dconv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_2'),
            BatchNormalization(name=self.prefix + '_dconv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_2_af')
        ])

        self._dconv2x2_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_3'),
            BatchNormalization(name=self.prefix + '_dconv2x2_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_3_af')
        ])

        self._pconv1 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_1',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=False),
            BatchNormalization(name=self.prefix + '_pconv_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_pconv_1_af')
        ])

        self._pconv2 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_2',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=False),
            BatchNormalization(name=self.prefix + '_pconv_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_pconv_2_af')
        ])

        self._pconv3 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_3',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=False),
            BatchNormalization(name=self.prefix + '_pconv_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_pconv_3_af')
        ])

        self._add1 = Add(name=self.prefix + '_add1')
        self._add2 = Add(name=self.prefix + '_add2')
        self._add3 = Add(name=self.prefix + '_add3')
        self._add4 = Add(name=self.prefix + '_add4')
        self._add5 = Add(name=self.prefix + '_add5')
        self._add6 = Add(name=self.prefix + '_add6')
        self._add7 = Add(name=self.prefix + '_add7')
        self._add8 = Add(name=self.prefix + '_add8')

    def call(self, input1, input2):
        o1 = self._dconv2x2_1(input1)
        o2 = self._pconv1(input1)
        o3 = self._dconv3x3_1(input1)
        o4 = self._dconv3x3_2(input1)

        o5 = self._dconv3x3_3(input2)
        o6 = self._pconv2(input2)
        o7 = self._dconv3x3_4(input2)
        o8 = self._dconv3x3_5(input2)

        i0 = self._add1([o3, o5])
        i1 = self._add2([o4, o7])

        o9 = self._dconv2x2_2(i0)
        o10 = self._dconv3x3_6(i0)
        o11 = self._pconv3(i0)
        o12 = self._dconv3x3_7(i1)
        o13 = self._dconv3x3_8(i1)

        i2 = self._add3([o11, o6])
        o14 = self._dconv2x2_3(i2)

        i3 = self._add4([o10, o2])
        i4 = self._add5([o13, o8])
        i5 = self._add6([o1, o9])
        i6 = self._add7([o14, o12])

        output = self._add8([i5, i0, i3, i2, i6, i1, i4])
        return output, input1

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_filter': self.out_filter,
            'res': self.res,
        })
        return config


class FECellB(Layer):
    def __init__(self, out_filter, res, prefix):
        super(FECellB, self).__init__()
        self.prefix = prefix
        self._add = None
        self._pconv2 = None
        self._pconv1 = None
        self._dconv3x3_8 = None
        self._dconv3x3_7 = None
        self._dconv3x3_6 = None
        self._dconv3x3_5 = None
        self._dconv3x3_4 = None
        self._dconv3x3_3 = None
        self._dconv3x3_2 = None
        self._dconv3x3_1 = None
        self._dconv2x2_4 = None
        self._dconv2x2_3 = None
        self._dconv2x2_2 = None
        self._dconv2x2_1 = None
        self.out_filter = out_filter
        self.res = res
        self.strides = 1
        self.padding = 'same'
        self.dilation_rate = 0
        self.use_bias = True

    def build(self, input_shape):
        # Define layers here using input_shape
        self._dconv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_1'),
            BatchNormalization(name=self.prefix + '_dconv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_1_af')
        ])

        self._dconv3x3_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_2'),
            BatchNormalization(name=self.prefix + '_dconv3x3_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_2_af')
        ])

        self._dconv3x3_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_3'),
            BatchNormalization(name=self.prefix + '_dconv3x3_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_3_af')
        ])

        self._dconv3x3_4 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_4'),
            BatchNormalization(name=self.prefix + '_dconv3x3_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_4_af')
        ])

        self._dconv3x3_5 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_5'),
            BatchNormalization(name=self.prefix + '_dconv3x3_5_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_5_af')
        ])

        self._dconv3x3_6 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_6'),
            BatchNormalization(name=self.prefix + '_dconv3x3_6_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_6_af')
        ])

        self._dconv3x3_7 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_7'),
            BatchNormalization(name=self.prefix + '_dconv3x3_7_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_7_af')
        ])

        self._dconv3x3_8 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_8'),
            BatchNormalization(name=self.prefix + '_dconv3x3_8_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv3x3_8_af')
        ])

        self._dconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        self._dconv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_2'),
            BatchNormalization(name=self.prefix + '_dconv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_2_af')
        ])

        self._dconv2x2_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_3'),
            BatchNormalization(name=self.prefix + '_dconv2x2_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_3_af')
        ])

        self._dconv2x2_4 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_4'),
            BatchNormalization(name=self.prefix + '_dconv2x2_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_dconv2x2_4_af')
        ])

        self._pconv1 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_1',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_pconv_1_af')
        ])

        self._pconv2 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_2',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            # ReLU(name=self.prefix + '_pconv_2_af')
        ])

        self._add1 = Add(name=self.prefix + '_add1')
        self._add2 = Add(name=self.prefix + '_add2')
        self._add3 = Add(name=self.prefix + '_add3')
        self._add4 = Add(name=self.prefix + '_add4')
        self._add5 = Add(name=self.prefix + '_add5')
        self._add6 = Add(name=self.prefix + '_add6')
        self._add7 = Add(name=self.prefix + '_add7')
        self._add8 = Add(name=self.prefix + '_add8')

    def call(self, input1, input2):
        o1 = self._dconv3x3_1(input1)
        o2 = self._dconv2x2_1(input1)
        o3 = self._dconv3x3_2(input1)
        o4 = self._dconv3x3_3(input1)

        o5 = self._dconv3x3_4(input2)
        o6 = self._pconv1(input2)
        o7 = self._dconv3x3_5(input2)
        o8 = self._dconv3x3_6(input2)

        i0 = self._add1([o3, o5])
        i1 = self._add2([o4, o7])

        o9 = self._dconv2x2_2(i0)
        o10 = self._dconv3x3_7(i0)
        o11 = self._dconv3x3_8(i0)
        o12 = self._dconv2x2_3(i1)
        o13 = self._pconv2(i1)

        i2 = self._add3([o11, o6])
        o14 = self._dconv2x2_4(i2)

        i3 = self._add4([o10, o2])
        i4 = self._add5([o13, o8])
        i5 = self._add6([o1, o9])
        i6 = self._add7([o14, o12])

        output = self._add8([i5, i0, i3, i2, i6, i1, i4])
        return output, input1

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_filter': self.out_filter,
            'res': self.res,
        })
        return config


class FECell_DARTS(Layer):
    def __init__(self, out_filter, res, prefix):
        super(FECell_DARTS, self).__init__()
        self.prefix = prefix
        self._add = None

        # PCONV = 6
        self._pconv6 = None
        self._pconv5 = None
        self._pconv4 = None
        self._pconv3 = None
        self._pconv2 = None
        self._pconv1 = None

        # DWCONV2x2 = 3
        self._dconv2x2_3 = None
        self._dconv2x2_2 = None
        self._dconv2x2_1 = None

        # DILCONV2x2 = 2
        self._dilconv2x2_2 = None
        self._dilconv2x2_1 = None

        # DILCONV3x3 = 1
        self._dilconv3x3_1 = None

        # CONV2x2 = 2
        self._conv2x2_2 = None
        self._conv2x2_1 = None

        self.out_filter = out_filter
        self.res = res
        self.strides = 1
        self.padding = 'same'
        self.dilation_rate = 0
        self.use_bias = True

    def build(self, input_shape):
        # Define layers here using input_shape
        # PCONV = 6

        self._pconv1 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_1',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv_1_af')
        ])

        self._pconv2 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_2',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv2_af')
        ])

        self._pconv3 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_3',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv3_af')
        ])

        self._pconv4 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_4',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv4_af')
        ])

        self._pconv5 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_5',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_5_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv5_af')
        ])

        self._pconv6 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_6',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_6_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv6_af')
        ])

        # DWCONV2x2 = 3
        self._dconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        self._dconv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_2'),
            BatchNormalization(name=self.prefix + '_dconv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_2_af')
        ])

        self._dconv2x2_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_3'),
            BatchNormalization(name=self.prefix + '_dconv2x2_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_3_af')
        ])

        # DILCONV2x2 = 2
        self._dilconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding='same', dilation_rate=2,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dilconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dilconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dilconv2x2_1_af')
        ])

        self._dilconv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding='same', dilation_rate=2,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dilconv2x2_2'),
            BatchNormalization(name=self.prefix + '_dilconv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dilconv2x2_2_af')
        ])

        # DILCONV3x3 = 1
        self._dilconv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding, dilation_rate=3,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dilconv3x3_1'),
            BatchNormalization(name=self.prefix + '_dilconv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dilconv3x3_1_af')
        ])

        # CONV2x2 = 2
        self._conv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_1'),
            BatchNormalization(name=self.prefix + '_conv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_1_af')
        ])

        self._conv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_2'),
            BatchNormalization(name=self.prefix + '_conv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_2_af')
        ])

        self._add1 = Add(name=self.prefix + '_add1')
        self._add2 = Add(name=self.prefix + '_add2')
        self._add3 = Add(name=self.prefix + '_add3')
        self._add4 = Add(name=self.prefix + '_add4')
        self._add5 = Add(name=self.prefix + '_add5')
        self._add6 = Add(name=self.prefix + '_add6')
        self._add7 = Add(name=self.prefix + '_add7')
        self._add8 = Add(name=self.prefix + '_add8')

    def call(self, input1, input2):
        o1 = self._pconv1(input1)
        o2 = self._pconv2(input1)
        o3 = self._dconv2x2_1(input1)
        o4 = self._dilconv2x2_1(input1)

        o5 = self._pconv3(input2)
        o6 = self._dilconv2x2_2(input2)
        o7 = self._conv2x2_1(input2)
        o8 = self._dconv2x2_2(input2)

        i0 = self._add1([o3, o5])
        i1 = self._add2([o4, o7])

        o9 = self._dilconv3x3_1(i0)
        o10 = self._pconv4(i0)
        o11 = self._pconv5(i0)
        o12 = self._pconv6(i1)
        o13 = self._dconv2x2_3(i1)

        i2 = self._add3([o11, o6])
        o14 = self._conv2x2_2(i2)

        i3 = self._add4([o10, o2])
        i4 = self._add5([o13, o8])
        i5 = self._add6([o1, o9])
        i6 = self._add7([o14, o12])

        output = self._add8([i5, i0, i3, i2, i6, i1, i4])
        return output, input1

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_filter': self.out_filter,
            'res': self.res,
        })
        return config

class FECell_PDARTS(Layer):
    def __init__(self, out_filter, res, prefix):
        super(FECell_PDARTS, self).__init__()
        self.prefix = prefix
        self._add = None

        # PCONV = 9
        self._pconv9 = None
        self._pconv8 = None
        self._pconv7 = None
        self._pconv6 = None
        self._pconv5 = None
        self._pconv4 = None
        self._pconv3 = None
        self._pconv2 = None
        self._pconv1 = None

        # CONV2x2 = 3
        self._conv2x2_3 = None
        self._conv2x2_2 = None
        self._conv2x2_1 = None

        # CONV3x3 = 1
        self._conv3x3_1 = None

        # DWCONV2x2 = 1
        self._dconv2x2_1 = None

        self.out_filter = out_filter
        self.res = res
        self.strides = 1
        self.padding = 'same'
        self.dilation_rate = 0
        self.use_bias = True

    def build(self, input_shape):
        # Define layers here using input_shape
        # PCONV = 6

        self._pconv1 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_1',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv1_af')
        ])

        self._pconv2 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_2',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv2_af')
        ])

        self._pconv3 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_3',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv3_af')
        ])

        self._pconv4 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_4',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv4_af')
        ])

        self._pconv5 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_5',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_5_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv5_af')
        ])

        self._pconv6 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_6',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_6_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv6_af')
        ])

        self._pconv7 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_7',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_7_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv7_af')
        ])

        self._pconv8 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_8',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_8_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv8_af')
        ])

        self._pconv9 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_9',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_9_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv9_af')
        ])

        # DWCONV2x2 = 1
        self._dconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        # CONV2x2 = 3
        self._conv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_1'),
            BatchNormalization(name=self.prefix + '_conv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_1_af')
        ])

        self._conv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_2'),
            BatchNormalization(name=self.prefix + '_conv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_2_af')
        ])

        self._conv2x2_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_3'),
            BatchNormalization(name=self.prefix + '_conv2x2_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_3_af')
        ])

        # Conv3x3 = 1
        self._conv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv3x3_1'),
            BatchNormalization(name=self.prefix + '_conv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv3x3_1_af')
        ])

        self._add1 = Add(name=self.prefix + '_add1')
        self._add2 = Add(name=self.prefix + '_add2')
        self._add3 = Add(name=self.prefix + '_add3')
        self._add4 = Add(name=self.prefix + '_add4')
        self._add5 = Add(name=self.prefix + '_add5')
        self._add6 = Add(name=self.prefix + '_add6')
        self._add7 = Add(name=self.prefix + '_add7')
        self._add8 = Add(name=self.prefix + '_add8')

    def call(self, input1, input2):
        o1 = self._pconv1(input1)
        o2 = self._conv2x2_1(input1)
        o3 = self._pconv2(input1)
        o4 = self._conv2x2_2(input1)

        o5 = self._pconv3(input2)
        o6 = self._pconv4(input2)
        o7 = self._dconv2x2_1(input2)
        o8 = self._pconv5(input2)

        i0 = self._add1([o3, o5])
        i1 = self._add2([o4, o7])

        o9 = self._pconv6(i0)
        o10 = self._conv2x2_3(i0)
        o11 = self._conv3x3_1(i0)
        o12 = self._pconv7(i1)
        o13 = self._pconv8(i1)

        i2 = self._add3([o11, o6])
        o14 = self._pconv9(i2)

        i3 = self._add4([o10, o2])
        i4 = self._add5([o13, o8])
        i5 = self._add6([o1, o9])
        i6 = self._add7([o14, o12])

        output = self._add8([i5, i0, i3, i2, i6, i1, i4])
        return output, input1

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_filter': self.out_filter,
            'res': self.res,
        })
        return config


class FECell_PDDARTS(Layer):
    def __init__(self, out_filter, res, prefix):
        super(FECell_PDDARTS, self).__init__()
        self.prefix = prefix
        self._add = None

        # PCONV = 4
        self._pconv4 = None
        self._pconv3 = None
        self._pconv2 = None
        self._pconv1 = None

        # CONV2x2 = 1
        self._conv2x2_1 = None

        # CONV3x3 = 1
        self._conv3x3_1 = None

        # DWCONV2x2 = 6
        self._dconv2x2_6 = None
        self._dconv2x2_5 = None
        self._dconv2x2_4 = None
        self._dconv2x2_3 = None
        self._dconv2x2_2 = None
        self._dconv2x2_1 = None

        # DWCONV3x3 = 2
        self._dconv3x3_2 = None
        self._dconv3x3_1 = None

        self.out_filter = out_filter
        self.res = res
        self.strides = 1
        self.padding = 'same'
        self.dilation_rate = 0
        self.use_bias = True

    def build(self, input_shape):
        # Define layers here using input_shape
        # PCONV = 6

        self._pconv1 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_1',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv1_af')
        ])

        self._pconv2 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_2',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv2_af')
        ])

        self._pconv3 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_3',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv3_af')
        ])

        self._pconv4 = tf.keras.Sequential([
            Conv2D(filters=self.out_filter, kernel_size=1, strides=self.strides, padding=self.padding,
                   name=self.prefix + '_pconv_4',
                   input_shape=(self.res, self.res, self.out_filter), use_bias=self.use_bias),
            BatchNormalization(name=self.prefix + '_pconv_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_pconv4_af')
        ])

        # DWCONV2x2 = 6
        self._dconv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_1'),
            BatchNormalization(name=self.prefix + '_dconv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        self._dconv2x2_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_2'),
            BatchNormalization(name=self.prefix + '_dconv2x2_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_2_af')
        ])

        self._dconv2x2_3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_3'),
            BatchNormalization(name=self.prefix + '_dconv2x2_3_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_3_af')
        ])

        self._dconv2x2_4 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_4'),
            BatchNormalization(name=self.prefix + '_dconv2x2_4_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_4_af')
        ])

        self._dconv2x2_5 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_5'),
            BatchNormalization(name=self.prefix + '_dconv2x2_5_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_5_af')
        ])

        self._dconv2x2_6 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv2x2_6'),
            BatchNormalization(name=self.prefix + '_dconv2x2_6_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv2x2_1_af')
        ])

        # DWCONV3x3 = 2
        self._dconv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_1'),
            BatchNormalization(name=self.prefix + '_dconv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv3x3_1_af')
        ])

        self._dconv3x3_2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_dconv3x3_2'),
            BatchNormalization(name=self.prefix + '_dconv3x3_2_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_dconv3x3_2_af')
        ])

        # CONV2x2 = 1
        self._conv2x2_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=2, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv2x2_1'),
            BatchNormalization(name=self.prefix + '_conv2x2_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv2x2_1_af')
        ])

        # Conv3x3 = 1
        self._conv3x3_1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=self.strides, padding=self.padding,
                            input_shape=(self.res, self.res, self.out_filter), name=self.prefix + '_conv3x3_1'),
            BatchNormalization(name=self.prefix + '_conv3x3_1_bn'),
            # Lambda(lambda x: tf.nn.relu6(x)),
            # Lambda(lambda x: tf.clip_by_value(x, clip_value_min=MIN_VAL, clip_value_max=MAX_VAL)),
            ReLU(name=self.prefix + '_conv3x3_1_af')
        ])

        self._add1 = Add(name=self.prefix + '_add1')
        self._add2 = Add(name=self.prefix + '_add2')
        self._add3 = Add(name=self.prefix + '_add3')
        self._add4 = Add(name=self.prefix + '_add4')
        self._add5 = Add(name=self.prefix + '_add5')
        self._add6 = Add(name=self.prefix + '_add6')
        self._add7 = Add(name=self.prefix + '_add7')
        self._add8 = Add(name=self.prefix + '_add8')

    def call(self, input1, input2):
        o1 = self._conv3x3_1(input1)
        o2 = self._pconv1(input1)
        o3 = self._dconv3x3_1(input1)
        o4 = self._dconv2x2_1(input1)

        o5 = self._pconv2(input2)
        o6 = self._dconv2x2_2(input2)
        o7 = self._dconv2x2_3(input2)
        o8 = self._pconv3(input2)

        i0 = self._add1([o3, o5])
        i1 = self._add2([o4, o7])

        o9 = self._conv2x2_1(i0)
        o10 = self._dconv2x2_4(i0)
        o11 = self._pconv4(i0)
        o12 = self._dconv3x3_2(i1)
        o13 = self._dconv2x2_5(i1)

        i2 = self._add3([o11, o6])
        o14 = self._dconv2x2_6(i2)

        i3 = self._add4([o10, o2])
        i4 = self._add5([o13, o8])
        i5 = self._add6([o1, o9])
        i6 = self._add7([o14, o12])

        output = self._add8([i5, i0, i3, i2, i6, i1, i4])
        return output, input1

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_filter': self.out_filter,
            'res': self.res,
        })
        return config
