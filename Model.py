import tensorflow as tf

from CapsuleLayer import CapsuleLayer
from ConvCapsuleLayer import ConvCapsuleLayer
from Loss import get_margin_loss
from Loss import get_reconstruction_loss
from utils import safe_norm
from CapsuleLayer import ConvAdapter
from CapsuleLayer import CapsAdapter

class Model(object):
    '''

    '''

    def __init__(self, input_image_batch, batch_size):

        conv_caps_layer = ConvCapsuleLayer(1152, 3)
        conv_caps = conv_caps_layer(input_image_batch)

        # conv_caps_layer = ConvCapsuleLayer(1152, 3)
        # conv_caps = conv_caps_layer(input_image_batch)

        digit_caps_layer = CapsuleLayer(1152, 3, 10, 16, ConvAdapter())
        routing_output1 = digit_caps_layer(conv_caps, batch_size)
        # (?, 1, caps, dims, 1)

        # digit_caps2_layer = CapsuleLayer(20, 16, 10, 16, CapsAdapter())
        # routing_output2 = digit_caps2_layer(routing_output1, batch_size)

        final_model_output = routing_output1

        # single digit prediction
        single_digit_prediction = self._transform_model_output_to_a_single_digit(final_model_output)
        # (?, )

        # labels
        correct_labels_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)

        # loss
        margin_loss = get_margin_loss(correct_labels_placeholder, final_model_output)
        mask_with_labels = tf.placeholder_with_default(False, shape=())

        reconstruction_loss, decoder_output, masked_out = _get_reconstruction_loss(mask_with_labels,
                                                                                   correct_labels_placeholder,
                                                                                   single_digit_prediction,
                                                                                   final_model_output,
                                                                                   input_image_batch)

        # keep it small
        reconstruction_alpha = 0.0005
        # favor the margin loss with a small weight for reconstruction loss
        final_loss = tf.add(margin_loss, reconstruction_alpha * reconstruction_loss)

        correct = tf.equal(correct_labels_placeholder, single_digit_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(final_loss)

        self.digit_caps_routing_output = final_model_output
        self.final_loss = final_loss
        self.correct = correct
        self.accuracy = accuracy
        self.mask_with_labels = mask_with_labels
        self.decoder_output = decoder_output
        self.single_digit_prediction = single_digit_prediction
        self.correct_labels_placeholder = correct_labels_placeholder
        self.masked_out = masked_out

    def _transform_model_output_to_a_single_digit(self, digitCaps_postRouting):
        # what we have: 10 16-dimensional vectors
        # what we want: which digit are you predicting ?

        # normalize to to get 10 scalars (length of the vectors)
        y_prob = safe_norm(digitCaps_postRouting, axis=-2)
        # (", 1, 10, 1)

        # get index of longest output vector
        y_prob_argmax = tf.argmax(y_prob, axis=2)
        # (?, 1, 1)

        # we have a 1 x 1 matrix , lets just say 1
        y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2])
        return y_pred


def _get_reconstruction_loss(mask_with_labels, y, y_pred, digitCaps_postRouting, input_image_batch):
    # first take the 10 16-dimension vectors output and pulls out the [predicted digit vector|correct_label digit vector)
    # (ex. prediction: digit 3 so take the 16-dimension vector and pass it to the decoder)

    # make a tensorflow placeholder for choosing based on label or prediction
    # during training pass the correct label digit
    # during inference pass what the model guessed
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred)  # if False)

    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=10)

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, 10, 1, 1])

    # mask it! (10, 16) * [0, 0, 1, 0, 0, ...]
    masked_out = tf.multiply(digitCaps_postRouting, reconstruction_mask_reshaped)
    # print("shape!!!!!!: {}".format(masked_out))
    # masked out
    # (10, 16) but only (1, 16) has values because of the above


    # Decoder will use the 16 dimension vector to reconstruct the image (28 x 28)
    reconstruction_loss, decoder_output = get_reconstruction_loss(masked_out, input_image_batch)
    return reconstruction_loss, decoder_output, masked_out
