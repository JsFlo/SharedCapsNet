import tensorflow as tf
import numpy as np
from Model import Model
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from utils import create_dirs_if_not_exists

MNIST = input_data.read_data_sets("/tmp/data/")

parser = argparse.ArgumentParser()
# REQUIRED
parser.add_argument('--checkpoint_dir', type=str, required=False, default="test_checkPoint",
                    help='Directory where the checkpoints will be saved')
parser.add_argument('--checkpoint_name', type=str, required=False, default="my_model",
                    help='Checkpoint name')
# OPTIONAL
parser.add_argument('--restore_checkpoint', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--debug', type=bool, default=False)
FLAGS = parser.parse_args()

if (FLAGS.debug):
    FLAGS.batch_size = 1
    FLAGS.n_epochs = 1
    N_ITERATIONS_PER_EPOCH = 1
    N_ITERATIONS_VALIDATION = 1
else:
    N_ITERATIONS_PER_EPOCH = MNIST.train.num_examples // FLAGS.batch_size
    N_ITERATIONS_VALIDATION = MNIST.validation.num_examples // FLAGS.batch_size

CHECKPOINT_PATH = FLAGS.checkpoint_dir + "/" + FLAGS.checkpoint_name

BEST_LOSS_VAL = np.infty

create_dirs_if_not_exists(FLAGS.checkpoint_dir)

input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
batch_size = tf.shape(input_image_batch)[0]

model = Model(input_image_batch, batch_size)

training_op = model.training_op
final_loss = model.final_loss
correct_labels_placeholder = model.correct_labels_placeholder
mask_with_labels = model.mask_with_labels
accuracy = model.accuracy
margin_loss = model.margin_loss
reconstruction_loss = model.reconstruction_loss

init = tf.global_variables_initializer()
saver = tf.train.Saver()
best_loss_val = BEST_LOSS_VAL
with tf.Session() as sess:
    if FLAGS.restore_checkpoint and tf.train.checkpoint_exists(CHECKPOINT_PATH):
        saver.restore(sess, CHECKPOINT_PATH)
    else:
        init.run()

    for epoch in range(FLAGS.n_epochs):
        for iteration in range(1, N_ITERATIONS_PER_EPOCH + 1):
            X_batch, y_batch = MNIST.train.next_batch(FLAGS.batch_size)
            # train and get loss to log
            _, loss_train, margin_loss_val, reconstruction_loss_val = sess.run([training_op, final_loss, margin_loss, reconstruction_loss],
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
            X_batch, y_batch = MNIST.validation.next_batch(FLAGS.batch_size)
            loss_val, acc_val, marg_loss_val, recon_loss_val= sess.run(
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
        print("\rEpoch: {}  Val accuracy: {:.3f}%  Loss: {:.5f} Margin Loss: {:.5f} ReconLoss: {:.5f}".format(
            epoch + 1, acc_val * 100, loss_val, final_marg_loss, final_recons_loss))

        # save if improved
        if loss_val < best_loss_val:
            print("(improved)")
            save_path = saver.save(sess, CHECKPOINT_PATH)
            best_loss_val = loss_val
