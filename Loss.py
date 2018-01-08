import tensorflow as tf
from utils import safe_norm


# paper used special margin loss to detect more than 1 digit in an image (overachievers)
def get_margin_loss(predicted_digit, digitCaps_postRouting):
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    predicted_one_hot_digit = tf.one_hot(predicted_digit, depth=10)

    digitCaps_postRouting_safeNorm = safe_norm(digitCaps_postRouting, axis=-2, keep_dims=True)

    present_error_raw = tf.square(tf.maximum(0., m_plus - digitCaps_postRouting_safeNorm))
    present_error = tf.reshape(present_error_raw, shape=(-1, 10))
    absent_error_raw = tf.square(tf.maximum(0., digitCaps_postRouting_safeNorm - m_minus))
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))

    loss = tf.add(predicted_one_hot_digit * present_error, lambda_ * (1.0 - predicted_one_hot_digit) * absent_error)
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return margin_loss


def get_reconstruction_loss(capsules, target_images):
    """

    :param capsules: array of 10 16 dimension vectors
    :param target_images: (? , 28, 28, 1)
    :return: loss
    """
    n_output = 28 * 28
    # flatten - reshape capsules to an array
    decoder_input = tf.reshape(capsules, [-1, capsules.shape[2].value * capsules.shape[3].value])

    # get prediction array
    decoder_output = _get_tf_layers_impl(decoder_input, n_output)
    # decoder_output = _get_tf_nn_impl(decoder_input, n_output)

    # flatten - reshape target images from (28 x 28) to (784)
    X_flat = tf.reshape(target_images, [-1, n_output])
    squared_difference = tf.square(X_flat - decoder_output)
    reconstruction_loss = tf.reduce_sum(squared_difference)

    return reconstruction_loss, decoder_output


def _get_tf_layers_impl(decoder_input, target,
                        n_hidden1=512, n_hidden2=1024):
    """
    3 fc layers
    512 -> relu ->
    1024 -> relu ->
    target -> sigmoid ->

    :param decoder_input:
    :return:
    """
    with tf.variable_scope('decoder') as scope:
        n_output = target

        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu)
        decoder_output = tf.layers.dense(hidden2, n_output,
                                         activation=tf.nn.sigmoid)
        return decoder_output


def _get_tf_nn_impl(decoder_input, target,
                    n_hidden1=512, n_hidden2=1024):
    with tf.variable_scope('decoder') as scope:
        # 160 was flattened above (10 x 16)
        fc1 = getFullyConnectedLayer_relu(decoder_input, 160, n_hidden1)
        fc2 = getFullyConnectedLayer_relu(fc1, n_hidden1, n_hidden2)
        fc3 = getFullyConnectedLayer_sigmoid(fc2, n_hidden2, target)
        return fc3


# fully connected with relu
def getFullyConnectedLayer_relu(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.relu(tf.matmul(lastLayer, W_fc1) + b_fc1)


# fully connected with sigmoid
def getFullyConnectedLayer_sigmoid(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.sigmoid(tf.matmul(lastLayer, W_fc1) + b_fc1)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
