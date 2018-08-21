import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def downsampling_block(inputs, num_internal_outputs, final, is_training, scope):
    print(scope)
    conv = slim.conv2d(inputs, num_outputs=num_internal_outputs, kernel_size=[3, 3], stride=2, activation_fn=None,
                       scope=scope + '_conv')
    bn = slim.batch_norm(conv, is_training=is_training, scope=scope + '_bn')
    pool = slim.max_pool2d(inputs, kernel_size=[2, 2], stride=2, padding='SAME', scope=scope + '_pool')

    if not final:
        net = tf.concat([bn, pool], axis=3, name=scope + '_concat')
        net = tf.nn.crelu(net, name=scope + '_crelu')
    else:
        net = tf.add(bn, pool, name=scope + '_relu')
        net = tf.nn.relu(net, name=scope + '_relu')

    print(net.get_shape().as_list())  # 1/2 resolution
    return net


@slim.add_arg_scope
def convolution_block(inputs, num_outputs, internal_depth_multiplier, dilation_rate, is_training, scope):
    print(scope)
    previous_block = inputs

    net = slim.conv2d(inputs, num_outputs=internal_depth_multiplier * num_outputs, kernel_size=[3, 1],
                      rate=dilation_rate,
                      activation_fn=tf.nn.relu, scope=scope + '_conv1_1')
    net = slim.conv2d(net, num_outputs=internal_depth_multiplier * num_outputs, kernel_size=[1, 3], rate=dilation_rate,
                      activation_fn=tf.nn.relu, scope=scope + '_conv1_2')
    net = slim.batch_norm(net, is_training=is_training, scope=scope + '_bn_1')

    net = slim.conv2d(net, num_outputs=num_outputs, kernel_size=[3, 1], activation_fn=tf.nn.relu, rate=dilation_rate,
                      scope=scope + '_conv2_1')
    net = slim.conv2d(net, num_outputs=num_outputs, kernel_size=[1, 3], activation_fn=None, rate=dilation_rate,
                      scope=scope + '_conv2_2')
    net = slim.batch_norm(net, is_training=is_training, scope=scope + '_bn_2')

    net = tf.add(previous_block, net, name=scope + '_add')
    net = tf.nn.relu(net, name=scope + '_final_relu')

    print(net.get_shape().as_list())
    return net


@slim.add_arg_scope
def upsampling_block(inputs, num_outputs, scope):
    print(scope)

    net = slim.conv2d_transpose(inputs, num_outputs=num_outputs, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu,
                                scope=scope + '_conv')
    print(net.get_shape().as_list())

    return net


def ShallowSeg(inputs, num_classes, reuse=None, is_training=True, scope='ShallowSeg'):
    """
    The ShallowSeg model, implemented in TensorFlow Slim.
    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one
                        batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output
                        channels as the answer.
    - batch_size(int): the batch size to explicitly set the shape of the inputs in order for operations to work
                        properly.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel
                        has a one-hot encoded vector determining the label of the pixel.
    """

    # Set the shape of the inputs first to get the batch_size information
    # inputs_shape = inputs.get_shape().as_list()
    # inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    with tf.variable_scope(scope, reuse=reuse):
        # inputs_shape = inputs.get_shape().as_list()
        # net = tf.reshape(tensor=inputs, shape=[batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]],
        #                  name='input')
        print(inputs.get_shape().as_list())

        # Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([downsampling_block, convolution_block], is_training=is_training), \
             slim.arg_scope([slim.batch_norm], fused=True):
            net = downsampling_block(inputs, num_internal_outputs=29, final=False, scope='downsampling_block_1')
            net = convolution_block(net, num_outputs=64, internal_depth_multiplier=2, dilation_rate=1,
                                    scope='convolution_block_1')

            net = downsampling_block(net, num_internal_outputs=64, final=False, scope='downsampling_block_2')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=2, dilation_rate=1,
                                    scope='convolution_block_2')

            net = downsampling_block(net, num_internal_outputs=256, final=True, scope='downsampling_block_3')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=1, dilation_rate=2,
                                    scope='convolution_block_3_dilated_2')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=1, dilation_rate=4,
                                    scope='convolution_block_4_dilated_4')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=1, dilation_rate=8,
                                    scope='convolution_block_5_dilated_8')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=1, dilation_rate=2,
                                    scope='convolution_block_6_dilated_2')
            net = convolution_block(net, num_outputs=256, internal_depth_multiplier=1, dilation_rate=4,
                                    scope='convolution_block_7_dilated_4')

            net = upsampling_block(net, num_outputs=128, scope='deconvolution_1')
            net = convolution_block(net, num_outputs=128, internal_depth_multiplier=2, dilation_rate=1,
                                    scope='convolution_block_8')

            net = upsampling_block(net, num_outputs=64, scope='deconvolution_2')
            net = convolution_block(net, num_outputs=64, internal_depth_multiplier=2, dilation_rate=1,
                                    scope='convolution_block_9')

        logits = slim.conv2d_transpose(net, num_outputs=num_classes, kernel_size=[3, 3], stride=2, activation_fn=None,
                                       scope='deconvolution_3' + '_conv')
        probabilities = tf.nn.softmax(logits, name='deconvolution_3' + '_probabilities')

        print(logits.get_shape().as_list())
        print(probabilities.get_shape().as_list())

        return logits, probabilities


def ShallowSeg_arg_scope(weight_decay=2e-4,
                         batch_norm_decay=0.1,
                         batch_norm_epsilon=0.001):
    """
    The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
    Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.
    INPUTS:
    - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
    - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
    - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.
    OUTPUTS:
    - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
    """

    # Set weight_decay for weights in conv2d and separable_conv2d layers.
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        # Set parameters for batch_norm.
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            return scope
