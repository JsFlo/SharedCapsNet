import tensorflow as tf
import RoutingByAgreement


class CapsuleLayer(object):
    '''

    '''

    def __init__(self, input_n_caps, input_n_dims,
                 output_n_caps, output_n_dims, init_sigma=0.01):
        self.input_n_caps = input_n_caps
        self.input_n_dims = input_n_dims
        self.output_n_caps = output_n_caps
        self.output_n_dims = output_n_dims
        self.init_sigma = init_sigma

    def __call__(self, input_caps, batch_size):
        # assert (input_caps.shape == [batch_size, 1, self.input_n_caps, 1, self.input_n_dims])
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
        # transformation matrix weights
        W_init = tf.random_normal(
            shape=(1, n_last_layer_capsules, n_next_layer_capsules,
                   n_next_layer_capsule_length, n_last_layer_capsule_length),
            stddev=self.init_sigma, dtype=tf.float32)
        W = tf.Variable(W_init)

        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
        caps1_output_tiled = tf.tile(last_layer_capsules, [1, 1, n_next_layer_capsules, 1, 1])

        next_layer_predictions = tf.matmul(W_tiled, caps1_output_tiled)

        return next_layer_predictions
