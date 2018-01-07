import tensorflow as tf
import RoutingByAgreement


class ConvAdapter(object):
    def __call__(self, conv_caps):
        # (?, caps, dims)
        conv_caps_expanded = tf.expand_dims(conv_caps, -1)
        # (?, caps, dims, 1)
        conv_caps_transformed = tf.expand_dims(conv_caps_expanded, 2)
        # (?, caps, 1, dims, 1)
        return conv_caps_transformed


class CapsAdapter(object):
    def __call__(self, caps):
        # (?, 1, caps, dims, 1)
        # (0, 1, 2, 3, 4)

        # (?, caps, 1, dims, 1)
        # (0, 2, 1, 3, 4)
        caps_transformed = tf.transpose(caps, [0, 2, 1, 3, 4])
        print("shape transformed: {}".format(caps_transformed.shape))
        return caps_transformed


class CapsuleLayer(object):
    '''

    '''

    def __init__(self, input_n_caps, input_n_dims,
                 output_n_caps, output_n_dims, caps_adapter, init_sigma=0.01):
        self.input_n_caps = input_n_caps
        self.input_n_dims = input_n_dims
        self.output_n_caps = output_n_caps
        self.output_n_dims = output_n_dims
        self.init_sigma = init_sigma
        self.caps_adapter = caps_adapter

    def __call__(self, input_caps, batch_size):
        input_caps = self.caps_adapter(input_caps)
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
        last_layer_tiled_next_layer = tf.tile(last_layer_capsules, [1, 1, n_next_layer_capsules, 1, 1])
        print("LAST_LAYER_WITH_NEW_LAYER_COPY: {}".format(last_layer_tiled_next_layer.shape))

        # transformation matrix weights
        W_init = tf.random_normal(
            shape=(1, n_last_layer_capsules, n_next_layer_capsules,
                   n_next_layer_capsule_length, n_last_layer_capsule_length),
            stddev=self.init_sigma, dtype=tf.float32)
        W = tf.Variable(W_init)
        print("WEIGHT: {}".format(W.shape))

        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
        print("WEIGHT TILED: {}".format(W_tiled.shape))

        next_layer_predictions = tf.matmul(W_tiled, last_layer_tiled_next_layer)

        return next_layer_predictions
