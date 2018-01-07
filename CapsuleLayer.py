import tensorflow as tf
import RoutingByAgreement


class CapsuleLayer(object):
    '''

    '''

    def __init__(self, input_n_caps, input_n_dims,
                 output_n_caps, output_n_dims):
        self.input_n_caps = input_n_caps
        self.input_n_dims = input_n_dims
        self.output_n_caps = output_n_caps
        self.output_n_dims = output_n_dims

    def __call__(self, input_caps, batch_size):
        last_layer_prediction = self._transform_capsule_prediction_for_layer(batch_size, input_caps,
                                                                             self.input_n_caps, self.input_n_dims,
                                                                             self.output_n_caps, self.output_n_dims)
        routing_output = RoutingByAgreement.routing_by_agreement(last_layer_prediction, batch_size)

        return routing_output

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
