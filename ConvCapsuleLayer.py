import tensorflow as tf
from utils import squash


class ConvCapsuleLayer(object):
    '''
    Convolutional Capsule Layer
    '''

    def __init__(self, n_capsules, capsule_length):
        '''

        :param n_capsules: number of capsules
        :param capsule_length: integer, length of the capsule vector
        '''
        self.n_capsules = n_capsules
        self.capsule_length = capsule_length

    def __call__(self, input):
        '''

        :param input: 4D tensor
        :return:
        '''
        conv_output = self._get_conv_output(input)
        # flatten
        flatten_caps = tf.reshape(conv_output, [-1, self.n_capsules, self.capsule_length])
        final_capsules = squash(flatten_caps)
        # squash to keep the vectors under 1
        return final_capsules

    def _get_conv_output(self, input):
        conv1 = tf.layers.conv2d(input, filters=256, kernel_size=9, strides=1,
                                 padding="valid", activation=tf.nn.relu)

        # printShape(conv1)  # (?, 20, 20, 256)

        # stride of 2!
        conv2 = tf.layers.conv2d(conv1, filters=256, kernel_size=9, strides=2,
                                 padding="valid", activation=tf.nn.relu)
        # printShape(conv2)  # (?, 6, 6, 256)
        return conv2

