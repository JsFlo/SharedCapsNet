import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model

MNIST = input_data.read_data_sets("/tmp/data/")
def writeToFile(string):
    print(string)
    with open('singleTestOut', 'a') as out:
        out.write(string + '\n')

class SingleTestTrain(object):
    '''

    '''

    def __init__(self, n_duo_conv_caps_36_x, n_duo_conv_dims, n_digit_caps, n_digit_dims, reconstruction_alpha, training_batch_size=50):
        tf.reset_default_graph()
        writeToFile("\n\n SINGLE TEST TRAIN\n")
        writeToFile("conv caps: {} dims: {}".format(n_duo_conv_caps_36_x * 36, n_duo_conv_dims))
        writeToFile("conv caps: {} dims: {}".format(n_duo_conv_caps_36_x * 36, n_duo_conv_dims))
        writeToFile("digit caps: {} dims: {}".format(n_digit_caps, n_digit_dims))
        writeToFile("reconstruction loss: {}".format(reconstruction_alpha))


        N_ITERATIONS_PER_EPOCH = MNIST.train.num_examples // training_batch_size
        N_ITERATIONS_VALIDATION = MNIST.validation.num_examples // training_batch_size

        input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        batch_size = tf.shape(input_image_batch)[0]

        model = Model(input_image_batch, batch_size, n_duo_conv_caps_36_x, n_duo_conv_dims,n_digit_caps, n_digit_dims, reconstruction_alpha )

        training_op = model.training_op
        final_loss = model.final_loss
        correct_labels_placeholder = model.correct_labels_placeholder
        mask_with_labels = model.mask_with_labels
        accuracy = model.accuracy
        margin_loss = model.margin_loss
        reconstruction_loss = model.reconstruction_loss

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()

            for epoch in range(1):
                for iteration in range(1, N_ITERATIONS_PER_EPOCH + 1):
                    X_batch, y_batch = MNIST.train.next_batch(training_batch_size)
                    # train and get loss to log
                    _, loss_train, margin_loss_val, reconstruction_loss_val = sess.run(
                        [training_op, final_loss, margin_loss, reconstruction_loss],
                        feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                                   correct_labels_placeholder: y_batch,
                                   mask_with_labels: True})  # use labels during training for the decoder

                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f} Margin Loss: {:.5f} ReconLoss: {:.5f}".format(
                        iteration, N_ITERATIONS_PER_EPOCH,
                        iteration * 100 / N_ITERATIONS_PER_EPOCH,
                        loss_train, margin_loss_val, reconstruction_loss_val),
                          end="")

                # check against validation set and log it
                loss_vals = []
                acc_vals = []
                marg_loss_vals = []
                recons_loss_vals = []
                for iteration in range(1, N_ITERATIONS_VALIDATION + 1):
                    X_batch, y_batch = MNIST.validation.next_batch(training_batch_size)
                    loss_val, acc_val, marg_loss_val, recon_loss_val = sess.run(
                        [final_loss, accuracy, margin_loss, reconstruction_loss],
                        feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                                   correct_labels_placeholder: y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    marg_loss_vals.append(marg_loss_val)
                    recons_loss_vals.append(recon_loss_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, N_ITERATIONS_VALIDATION,
                        iteration * 100 / N_ITERATIONS_VALIDATION),
                          end=" " * 10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                final_marg_loss = np.mean(marg_loss_vals)
                final_recons_loss = np.mean(recons_loss_vals)
                writeToFile("\rEpoch: {}  Val accuracy: {:.3f}%  Loss: {:.5f} Margin Loss: {:.5f} ReconLoss: {:.5f}".format(
                    epoch + 1, acc_val * 100, loss_val, final_marg_loss, final_recons_loss))
