import tensorflow as tf
import numpy as np
from utils import squash


def routing_by_agreement(digit_caps, batch_size, n_rounds=3):
    # weight for every pair
    raw_weights = tf.zeros([batch_size,
                            digit_caps.shape[1].value,
                            digit_caps.shape[2].value, 1, 1],
                           dtype=np.float32)
    # (?, 1152, 10, 1, 1)
    last_output, last_round_weights = _round_and_agreement(raw_weights, digit_caps, digit_caps.shape[1].value, True)
    for i in range(1, n_rounds):
        last_output, last_round_weights = _round_and_agreement(last_round_weights, digit_caps, digit_caps.shape[1].value, True)
    return last_output


def _round_and_agreement(weights, capsules, n_capsules, get_round_agreement=True):
    routing_output = _routing_round(weights, capsules)
    if (get_round_agreement):
        agreement = _round_agreement(routing_output, capsules, n_capsules)
        updated_weights = tf.add(weights, agreement)
        return routing_output, updated_weights
    else:
        return routing_output


def _routing_round(previous_weights, digit_caps_prediction):
    # print(": routing weights = softmax on previous weights")
    routing_weights = tf.nn.softmax(previous_weights, dim=2)
    # (?, 1152, 10, 1, 1)

    # print(": weighted predictions = routing weights x digit caps prediction")
    weighted_predictions = tf.multiply(routing_weights, digit_caps_prediction)
    # (?, 1152, 10, 16, 1)

    # Q: When getting weighted predictions why is there no bias ?

    # print(": reduce sum of all of them (collapse `rows`)")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True)
    # (?, 1 , 10, 16, 1)

    # print(": squash to keep below 1")
    round_output = squash(weighted_sum, axis=-2)
    # (?, 1 , 10, 16, 1)
    return round_output


def _round_agreement(round_output, digit_capsule_output, primary_n_caps):
    """
    Measure how close the digit capsule output is compared to a round guess.

    How to measure for **1** capsule:
        A = From digit-capsule_output (?, 1152, 10, 16, 1) take one(1, 10, 16, 1) take (10, 16)
        B = From round output (1, 10, 16, 1) take (10, 16)

        SCALAR = Perform a scalar product A(transpose) dot B

        The SCALAR value is the "agreement" value

        WAIT! Lets do all these scalar products in one run:
        digit-capsule_output (?, 1152, 10, 16, 1)
        round output (1, 10, 16, 1)

        COPY & PASTE the round output to match the digit_capsule_output
        so we want round output to be (1152, 10, 16, 1)

    :param round_output: (1, 10, 16, 1)
    :param digit_capsule_output: (?, 1152, 10, 16, 1)
    :param primary_n_caps: 1152
    :return: (?, 1152, 10,1, 1)
    """

    # the copy&paste
    round_output_tiled = tf.tile(
        round_output, [1, primary_n_caps, 1, 1, 1])
    print("NORMAL tiled {}".format(round_output_tiled.shape))
    # that scalar product we talked about above
    agreement = tf.matmul(digit_capsule_output, round_output_tiled, transpose_a=True)
    # (?, 1152, 10, 1, 1)
    print("NORMAL AGREEMENT: {}".format(agreement.shape))
    return agreement
