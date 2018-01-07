import tensorflow as tf
from utils import squash


class ConvCapsuleLayer(object):
    '''
    Convolutional Capsule Layer
    '''

    def __init__(self, n_capsules, capsule_length, n_next_layer_capsules, n_next_layer_capsule_length):
        '''

        :param n_capsules: number of capsules
        :param capsule_length: integer, length of the capsule vector
        '''
        self.n_capsules = n_capsules
        self.capsule_length = capsule_length
        self.n_next_layer_capsules = n_next_layer_capsules
        self.n_next_layer_capsule_length = n_next_layer_capsule_length

    def __call__(self, input, batch_size):
        '''

        :param input: 4D tensor
        :return:
        '''
        conv_output = self._get_conv_output(input)
        # flatten
        flatten_caps = tf.reshape(conv_output, [-1, self.n_capsules, self.capsule_length])
        final_capsules = squash(flatten_caps)
        # squash to keep the vectors under 1
        return final_capsules, self._transform_capsule_prediction_for_layer(batch_size, final_capsules,
                                                                            self.n_capsules, self.capsule_length,
                                                                            self.n_next_layer_capsules,
                                                                            self.n_next_layer_capsule_length)

    def _get_conv_output(self, input):
        conv1 = tf.layers.conv2d(input, filters=256, kernel_size=9, strides=1,
                                 padding="valid", activation=tf.nn.relu)

        # printShape(conv1)  # (?, 20, 20, 256)

        # stride of 2!
        conv2 = tf.layers.conv2d(conv1, filters=256, kernel_size=9, strides=2,
                                 padding="valid", activation=tf.nn.relu)
        # printShape(conv2)  # (?, 6, 6, 256)
        return conv2

    def _transform_capsule_prediction_for_layer(self, batch_size, last_layer_capsules,
                                                n_last_layer_capsules,
                                                n_last_layer_capsule_length,
                                                n_next_layer_capsules,
                                                n_next_layer_capsule_length):
        # The way to go from 8 dimensions to 16 dimensions is to use a **transformation matrix** for each pair (i, j)
        # For each capsule in the first layer *foreach (1, 8) in (1152, 8)* we want to predict the output of every capsule in this layer

        # since we will go from a primary-layer-capsule(1, 8) to a digit-capsule(1, 16)
        # we need the transformation matrix to be (16, 8)
        # ** (16, 8) * (8, 1) = (16, 1)**

        # Transformation matrix for one primary capsule: want [1152, 1] with [16, 8] vectors
        # EFFICIENT: let's do it for all capsules in this layer so we want [1152, 10] with [16, 8] vectors
        # what we want: (?, 1152, 10, 16, 8) "an 1152 by 10 matrix of 16x8 matrix"

        # transformation matrix weights
        init_sigma = 0.01

        W_init = tf.random_normal(
            shape=(1, n_last_layer_capsules, n_next_layer_capsules,
                   n_next_layer_capsule_length, n_last_layer_capsule_length),
            stddev=init_sigma, dtype=tf.float32)
        W = tf.Variable(W_init)

        # copy paste for batch size to get (BATCH_SIZE, 1152, 10, 16, 8)
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
        # (?, 1152, 10, 16, 8)


        # what we have: Transformation_matrix(BATCH_SIZE, 1152, 10, 16, 8)
        # what we need: Second matrix from last layer (BATCH_SIZE, 1152, 10, 8, 1)
        #   last layer matrix: (BATCH_SIZE, 1152, 8)

        # (BATCH_SIZE, 1152, 8) -> (BATCH_SIZE, 1152, 8, 1)
        caps1_output_expanded = tf.expand_dims(last_layer_capsules, -1)

        # (BATCH_SIZE, 1152, 8, 1) -> (BATCH_SIZE, 1152, 1, 8, 1)
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)

        # copy paste for digit_n_caps: (BATCH_SIZE, 1152, 1, 8, 1) -> (BATCH_SIZE, 1152, 10, 8, 1)
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, n_next_layer_capsules, 1, 1])

        next_layer_predictions = tf.matmul(W_tiled, caps1_output_tiled)
        # (? , 1152, 10, 16, 1)


        return next_layer_predictions
