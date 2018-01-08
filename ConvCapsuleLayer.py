import tensorflow as tf
from utils import squash


class SimpleConvOutput(object):
    def get_conv_output(self, input, n_caps, n_dims):
        conv1 = tf.layers.conv2d(input, filters=256, kernel_size=9, strides=1,
                                 padding="valid", activation=tf.nn.relu)

        # printShape(conv1)  # (?, 20, 20, 256)

        # because of the kernel size and strides the output grid will be 6 x 6 x ?
        # 6x6 = 36
        n_filters = (n_caps * n_dims) / 36
        # stride of 2!
        conv2 = tf.layers.conv2d(conv1, filters=n_filters, kernel_size=9, strides=2,
                                 padding="valid", activation=tf.nn.relu)
        # printShape(conv2)  # (?, 6, 6, 256)
        return conv2


class ConvCapsuleLayer(object):
    '''
    Convolutional Capsule Layer
    '''

    def __init__(self, n_capsules, capsule_length, conv_output=SimpleConvOutput()):
        '''

        :param n_capsules: number of capsules
        :param capsule_length: integer, length of the capsule vector
        '''
        self.n_caps = n_capsules
        self.n_dims = capsule_length
        self.conv_output = conv_output

    def __call__(self, input):
        '''

        :param input: 4D tensor
        :return:
        '''
        conv_output = self.conv_output.get_conv_output(input, self.n_caps, self.n_dims)
        # flatten
        flatten_caps = tf.reshape(conv_output, [-1, self.n_caps, self.n_dims])
        final_capsules = squash(flatten_caps)
        # squash to keep the vectors under 1
        return final_capsules
