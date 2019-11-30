import tensorflow as tf
from collections import OrderedDict


# -------------------------------------------------------------
#    Models
# -------------------------------------------------------------

class LeNet:
    def __init__(self):
        super().__init__()
        self.nb_classes = 10
        self.input_shape = [28, 28, 3]
        self.weights_init = 'He'
        self.filters = 32  # 32 is the default here for all our pre-trained models
        self.is_training = False
        self.bn = False
        self.bn_scale = False
        self.bn_bias = False
        self.parameters = 0

        # Create variables
        with tf.variable_scope('conv1_vars'):
            self.W_conv1 = create_conv2d_weights(kernel_size=3, filter_in=1, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.input_shape[-1] * self.filters

            self.b_conv1 = create_biases(size=self.filters)
            self.parameters += self.filters

        with tf.variable_scope('conv2_vars'):
            self.W_conv2 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters * 2,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * (self.filters * 2)

            self.b_conv2 = create_biases(size=self.filters * 2)
            self.parameters += self.filters * 2

        with tf.variable_scope('fc1_vars'):
            self.W_fc1 = create_weights(units_in=7 * 7 * self.filters * 2, units_out=1024, init=self.weights_init)
            self.parameters += (7 * 7 * self.filters * 2) * 1024

            self.b_fc1 = create_biases(size=1024)
            self.parameters += 1024

        with tf.variable_scope('fc2_vars'):
            self.W_fc2 = create_weights(units_in=1024, units_out=self.nb_classes, init=self.weights_init)
            self.parameters += 1024 * self.nb_classes

            self.b_fc2 = create_biases(size=self.nb_classes)
            self.parameters += self.nb_classes

        self.x_input = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_input = tf.placeholder(tf.int64, shape=[None])

        x = tf.reshape(self.x_input, [-1, 28, 28, 1])

        with tf.name_scope('conv-block-1'):
            conv1 = conv_layer(x, self.is_training, self.W_conv1, stride=1, padding='SAME', bn=self.bn,
                               bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv1', bias=self.b_conv1)

        with tf.name_scope('max-pool-1'):
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('conv-block-2'):
            conv2 = conv_layer(conv1, self.is_training, self.W_conv2, stride=1, padding='SAME', bn=self.bn,
                               bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv2', bias=self.b_conv2)

        with tf.name_scope('max-pool-2'):
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('fc-block'):
            conv2 = tf.layers.flatten(conv2)
            fc1 = fc_layer(conv2, self.is_training, self.W_fc1, bn=self.bn, bn_scale=self.bn_scale,
                           bn_bias=self.bn_bias, name='fc1', non_linearity='relu', bias=self.b_fc1)

            logits = fc_layer(fc1, self.is_training, self.W_fc2, bn=self.bn, bn_scale=self.bn_scale,
                              bn_bias=self.bn_bias, name='fc2', non_linearity='linear', bias=self.b_fc2)

        self.summaries = False
        self.logits = logits


class ResNet20_v2:
    def __init__(self):
        super().__init__()
        self.nb_classes = 10
        self.input_shape = [32, 32, 3]
        self.weights_init = 'He'
        self.filters = 64  # 64 is the default here for all our pre-trained models
        self.is_training = False
        self.bn = True
        self.bn_scale = True
        self.bn_bias = True
        self.parameters = 0

        # Create variables
        with tf.variable_scope('conv1_vars'):
            self.W_conv1 = create_conv2d_weights(kernel_size=3, filter_in=self.input_shape[-1], filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.input_shape[-1] * self.filters

        with tf.variable_scope('conv2_vars'):
            self.W_conv2 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv3_vars'):
            self.W_conv3 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv4_vars'):
            self.W_conv4 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv5_vars'):
            self.W_conv5 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv6_vars'):
            self.W_conv6 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv7_vars'):
            self.W_conv7 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * self.filters

        with tf.variable_scope('conv8_vars'):
            self.W_conv8 = create_conv2d_weights(kernel_size=3, filter_in=self.filters, filter_out=self.filters * 2,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * self.filters * (self.filters * 2)

        with tf.variable_scope('conv9_vars'):
            self.W_conv9 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2, filter_out=self.filters * 2,
                                                 init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 2)

        with tf.variable_scope('conv10_vars'):
            self.W_conv10 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2,
                                                  filter_out=self.filters * 2, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 2)

        with tf.variable_scope('conv11_vars'):
            self.W_conv11 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2,
                                                  filter_out=self.filters * 2, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 2)

        with tf.variable_scope('conv12_vars'):
            self.W_conv12 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2,
                                                  filter_out=self.filters * 2, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 2)

        with tf.variable_scope('conv13_vars'):
            self.W_conv13 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2,
                                                  filter_out=self.filters * 2, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 2)

        with tf.variable_scope('conv14_vars'):
            self.W_conv14 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 2,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 2) * (self.filters * 4)

        with tf.variable_scope('conv15_vars'):
            self.W_conv15 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 4,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 4) * (self.filters * 4)

        with tf.variable_scope('conv16_vars'):
            self.W_conv16 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 4,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 4) * (self.filters * 4)

        with tf.variable_scope('conv17_vars'):
            self.W_conv17 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 4,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 4) * (self.filters * 4)

        with tf.variable_scope('conv18_vars'):
            self.W_conv18 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 4,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 4) * (self.filters * 4)

        with tf.variable_scope('conv19_vars'):
            self.W_conv19 = create_conv2d_weights(kernel_size=3, filter_in=self.filters * 4,
                                                  filter_out=self.filters * 4, init=self.weights_init)
            self.parameters += 3 * 3 * (self.filters * 4) * (self.filters * 4)

        with tf.variable_scope('fc1_vars'):
            self.W_fc1 = create_weights(units_in=self.filters * 4, units_out=self.nb_classes, init=self.weights_init)
            self.parameters += (self.filters * 4) * self.nb_classes

            self.b_fc1 = create_biases(size=self.nb_classes)
            self.parameters += self.nb_classes

        with tf.variable_scope('scip1_vars'):
            self.W_scip1 = create_conv2d_weights(kernel_size=1, filter_in=self.filters, filter_out=self.filters,
                                                 init=self.weights_init)
            self.parameters += 1 * 1 * self.filters * self.filters

        with tf.variable_scope('scip2_vars'):
            self.W_scip2 = create_conv2d_weights(kernel_size=1, filter_in=self.filters, filter_out=self.filters * 2,
                                                 init=self.weights_init)
            self.parameters += 1 * 1 * self.filters * (self.filters * 2)

        with tf.variable_scope('scip3_vars'):
            self.W_scip3 = create_conv2d_weights(kernel_size=1, filter_in=self.filters * 2, filter_out=self.filters * 4,
                                                 init=self.weights_init)
            self.parameters += 1 * 1 * (self.filters * 2) * (self.filters * 4)

        self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_input = tf.placeholder(tf.int64, shape=None)
        x = self.x_input / 255.0

        # Specify forward pass
        with tf.name_scope('input-block'):
            conv1 = conv_layer(x, self.is_training, self.W_conv1, stride=1, padding='SAME',
                               bn=False, bn_scale=self.bn_scale, bn_bias=self.bn_bias,
                               name='conv1',
                               non_linearity='linear')

        with tf.name_scope('conv-block-1'):
            conv2 = pre_act_conv_layer(conv1, self.is_training, self.W_conv2, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv2')

            conv3 = pre_act_conv_layer(conv2, self.is_training, self.W_conv3, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv3')

            # skip connection
            conv3 += tf.nn.conv2d(conv1, self.W_scip1, strides=[1, 1, 1, 1], padding='SAME', name='conv-skip1')

            conv4 = pre_act_conv_layer(conv3, self.is_training, self.W_conv4, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv4')

            conv5 = pre_act_conv_layer(conv4, self.is_training, self.W_conv5, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv5')

            # skip connection
            conv5 += conv3

            conv6 = pre_act_conv_layer(conv5, self.is_training, self.W_conv6, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv6')

            conv7 = pre_act_conv_layer(conv6, self.is_training, self.W_conv7, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv7')

            # skip connection
            conv7 += conv5

        with tf.name_scope('conv-block-2'):
            conv8 = pre_act_conv_layer(conv7, self.is_training, self.W_conv8, stride=2, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv8')

            conv9 = pre_act_conv_layer(conv8, self.is_training, self.W_conv9, stride=1, padding='SAME',
                                       bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv9')

            # skip connection
            conv9 += tf.nn.conv2d(conv7, self.W_scip2, strides=[1, 2, 2, 1], padding='SAME', name='conv-skip2')

            conv10 = pre_act_conv_layer(conv9, self.is_training, self.W_conv10, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv10')

            conv11 = pre_act_conv_layer(conv10, self.is_training, self.W_conv11, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv11')

            # skip connection
            conv11 += conv9

            conv12 = pre_act_conv_layer(conv11, self.is_training, self.W_conv12, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv12')

            conv13 = pre_act_conv_layer(conv12, self.is_training, self.W_conv13, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv13')

            # skip connection
            conv13 += conv11

        with tf.name_scope('conv-block-3'):
            conv14 = pre_act_conv_layer(conv13, self.is_training, self.W_conv14, stride=2, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv14')

            conv15 = pre_act_conv_layer(conv14, self.is_training, self.W_conv15, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv15')

            # skip connection
            conv15 += tf.nn.conv2d(conv13, self.W_scip3, strides=[1, 2, 2, 1], padding='SAME', name='conv-skip3')

            conv16 = pre_act_conv_layer(conv15, self.is_training, self.W_conv16, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv16')

            conv17 = pre_act_conv_layer(conv16, self.is_training, self.W_conv17, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv17')

            # skip connection
            conv17 += conv15

            conv18 = pre_act_conv_layer(conv17, self.is_training, self.W_conv18, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv18')

            conv19 = pre_act_conv_layer(conv18, self.is_training, self.W_conv19, stride=1, padding='SAME',
                                        bn=self.bn, bn_scale=self.bn_scale, bn_bias=self.bn_bias, name='conv19')

            # skip connection
            conv19 += conv17
            conv19 = nonlinearity(conv19)

        with tf.name_scope('output-block'):
            with tf.name_scope('global-average-pooling'):
                fc1 = tf.reduce_mean(conv19, axis=[1, 2])

            logits = fc_layer(fc1, self.is_training, self.W_fc1, bn=False, bn_scale=self.bn_scale, bn_bias=self.bn_bias,
                              name='fc1',
                              non_linearity='linear', bias=self.b_fc1)

        self.summaries = False
        self.logits = logits


# -------------------------------------------------------------
#    Helpers
# -------------------------------------------------------------

def create_weights(units_in, units_out, init='Xavier', seed=None):
    if init == 'Xavier':
        initializer = tf.variance_scaling_initializer(scale=1.0,
                                                      mode='fan_in',
                                                      distribution='normal',
                                                      seed=None,
                                                      dtype=tf.float32)
    elif init == 'He':
        initializer = tf.variance_scaling_initializer(scale=2.0,
                                                      mode='fan_in',
                                                      distribution='normal',
                                                      seed=None,
                                                      dtype=tf.float32)
    else:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, seed=seed, dtype=tf.float32)

    weights = tf.get_variable(name='weights',
                              shape=[units_in, units_out],
                              dtype=tf.float32,
                              initializer=initializer)
    return weights


def create_conv2d_weights(kernel_size, filter_in, filter_out, init='Xavier', seed=None):
    if init == 'Xavier':
        initializer = tf.variance_scaling_initializer(scale=1.0,
                                                      mode='fan_in',
                                                      distribution='normal',
                                                      seed=None,
                                                      dtype=tf.float32)
    elif init == 'He':
        initializer = tf.variance_scaling_initializer(scale=2.0,
                                                      mode='fan_in',
                                                      distribution='normal',
                                                      seed=None,
                                                      dtype=tf.float32)
    else:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, seed=seed, dtype=tf.float32)

    weights = tf.get_variable(name='weights',
                              shape=[kernel_size, kernel_size, filter_in, filter_out],
                              dtype=tf.float32,
                              initializer=initializer)
    return weights


def create_biases(size):
    return tf.get_variable(name='biases', shape=[size], dtype=tf.float32, initializer=tf.zeros_initializer())


def batch_norm(x, is_training, scale, bias, name, reuse):
    return tf.contrib.layers.batch_norm(
        x,
        decay=0.999,
        center=bias,
        scale=scale,
        epsilon=0.001,
        param_initializers=None,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=is_training,
        reuse=reuse,
        variables_collections=['batch-norm'],
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        fused=False,
        zero_debias_moving_mean=False,
        scope=name,
        renorm=False,
        renorm_clipping=None,
        renorm_decay=0.99
    )


def nonlinearity(x, non_linearity='relu'):
    if non_linearity == 'linear':
        return tf.identity(x)
    if non_linearity == 'sigmoid':
        return tf.nn.sigmoid(x)
    if non_linearity == 'tanh':
        return tf.nn.tanh(x)
    if non_linearity == 'relu':
        return tf.nn.relu(x)
    if non_linearity == 'elu':
        return tf.nn.elu(x)
    if non_linearity == 'selu':
        return tf.nn.selu(x)


def conv_layer(inputs, is_training, weights, stride, padding, bn, bn_scale, bn_bias, name,
               non_linearity='relu', bias=None):
    if bias is not None:
        inputs = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding) + bias
    else:
        inputs = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)

    if bn:
        inputs = batch_norm(inputs, is_training=is_training, scale=bn_scale, bias=bn_bias,
                            name='batch-norm-{:s}'.format(name),
                            reuse=tf.AUTO_REUSE)

    activations = nonlinearity(inputs, non_linearity=non_linearity)

    return activations


def pre_act_conv_layer(inputs, is_training, weights, stride, padding, bn, bn_scale, bn_bias, name,
                       non_linearity='relu'):
    if bn:
        inputs = batch_norm(inputs, is_training=is_training, scale=bn_scale, bias=bn_bias,
                            name='batch-norm-{:s}'.format(name),
                            reuse=tf.AUTO_REUSE)

    activations = nonlinearity(inputs, non_linearity=non_linearity)

    outputs = tf.nn.conv2d(activations, weights, strides=[1, stride, stride, 1], padding=padding)

    return outputs


def fc_layer(inputs, is_training, weights, bn, bn_scale, bn_bias, name, non_linearity='relu', bias=None):
    if bias is not None:
        inputs = tf.matmul(inputs, weights) + bias
    else:
        inputs = tf.matmul(inputs, weights)

    if bn:
        inputs = batch_norm(inputs, is_training=is_training, scale=bn_scale, bias=bn_bias,
                            name='batch-norm-{:s}'.format(name),
                            reuse=tf.AUTO_REUSE)

    activations = nonlinearity(inputs, non_linearity)

    return activations
