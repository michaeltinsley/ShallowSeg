import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import tf_logging as logging
from ShallowSeg import ShallowSeg, ShallowSeg_arg_scope
from preprocessing import preprocess
from get_class_weights import ENet_weighting, MFB_weighting
import os
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============INPUT ARGUMENTS==================
flags = tf.app.flags

# Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/original', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')

# Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 15, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 20, 'The batch size used for validation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 1600, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 900, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 2.5e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')

FLAGS = flags.FLAGS

# ==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size  # Can be larger than train_batch as no need to backpropagate gradients.
combine_dataset = FLAGS.combine_dataset

# Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs = FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

# Use median frequency balancing or not
weighting = FLAGS.weighting

# Visualization and where to save images
save_images = FLAGS.save_images
photo_dir = os.path.join(FLAGS.logdir, "images")

# Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

# ===============PREPARATION FOR TRAINING==================
# Get the images into a list
image_files = sorted(
    [os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.png')])
annotation_files = sorted(
    [os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if
     file.endswith('.png')])

image_val_files = sorted(
    [os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.png')])
annotation_val_files = sorted(
    [os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if
     file.endswith('.png')])

if combine_dataset:
    image_files += image_val_files
    annotation_files += annotation_val_files

# Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = len(image_files) / batch_size
decay_steps = int(num_epochs_before_decay * num_batches_per_epoch)

num_val_batch_per_epoch = len(image_val_files) / eval_batch_size

# =================CLASS WEIGHTS===============================
# Median frequency balancing class_weights
if weighting == "MFB":
    class_weights = MFB_weighting()
    print("========= Median Frequency Balancing Class Weights =========\n", class_weights)

# Inverse weighing probability class weights
elif weighting == "ENET":
    class_weights = ENet_weighting()
    print("========= ENet Class Weights =========\n", class_weights)


# ============= TRAINING =================
def weighted_cross_entropy(onehot_labels, logits, class_weights):
    """
    A quick wrapper to compute weighted cross entropy.
    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.
    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------
    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.
    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    """
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss


def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        # ===================TRAINING BRANCH=======================
        # Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        input_queue = tf.train.slice_input_producer(
            [images, annotations])  # Slice_input producer shuffles the data by default.

        # Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(input_queue[1])
        annotation = tf.image.decode_image(annotation)

        # Pre-process and batch up the image and annotation
        preprocessed_image, preprocessed_annotation = preprocess(image, annotation, image_height, image_width)
        images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation],
                                             batch_size=batch_size,
                                             allow_smaller_final_batch=True)

        # Create the model inference
        with slim.arg_scope(ShallowSeg_arg_scope(weight_decay=weight_decay)):
            logits, probabilities = ShallowSeg(images,
                                               num_classes=num_classes,
                                               reuse=None,
                                               is_training=True)

        # Perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)

        # Actually compute the loss
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=learning_rate_decay_factor,
                                        staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        # accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        accuracy, accuracy_update = tf.metrics.accuracy(labels=annotations, predictions=predictions)

        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions,
                                                                          labels=annotations,
                                                                          num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update)

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            """
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            """
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val, _ = sess.run([train_op,
                                                                                     global_step,
                                                                                     accuracy,
                                                                                     mean_IOU,
                                                                                     metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to show some results
            logging.info(
                'Epoch %d / %d: Batch %d / %d: Loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f',
                epoch + 1, num_epochs, batch + 1, num_batches_per_epoch, total_loss, time_elapsed, accuracy_val,
                mean_IOU_val)

            return total_loss, accuracy_val, mean_IOU_val

        # ================VALIDATION BRANCH========================
        # Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])

        # Decode the image and annotation raw content
        image_val = tf.read_file(input_queue_val[0])
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        annotation_val = tf.read_file(input_queue_val[1])
        annotation_val = tf.image.decode_png(annotation_val)

        # Pre-process and batch up the image and annotation
        preprocessed_image_val, preprocessed_annotation_val = preprocess(image_val,
                                                                         annotation_val,
                                                                         image_height,
                                                                         image_width)
        images_val, annotations_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val],
                                                     batch_size=eval_batch_size,
                                                     allow_smaller_final_batch=True)

        with slim.arg_scope(ShallowSeg_arg_scope(weight_decay=weight_decay)):
            logits_val, probabilities_val = ShallowSeg(images_val,
                                                       num_classes=num_classes,
                                                       is_training=True,
                                                       reuse=True)

        # Perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)

        # State the metrics that you want to predict.
        # We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        # accuracy_val, accuracy_val_update = tf.contrib.metrics.streaming_accuracy(predictions_val, annotations_val)
        accuracy_val, accuracy_val_update = tf.metrics.accuracy(labels=annotations_val, predictions=predictions_val)
        mean_IOU_val, mean_IOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val,
                                                                                  labels=annotations_val,
                                                                                  num_classes=num_classes)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update)

        # Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val,
                                                   shape=[-1, image_height, image_width, 1])

        def eval_step(sess, metrics_op):
            """
            Simply takes in a session, runs the metrics op and some logging information.
            """
            start_time = time.time()
            _, accuracy_value, mean_IOU_value = sess.run([metrics_op, accuracy_val, mean_IOU_val])
            time_elapsed = time.time() - start_time

            # Log some information
            sec_per_image = time_elapsed / eval_batch_size
            fps = int(1 / sec_per_image)

            logging.info(
                '---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    (%.4f sec/per image, %d FPS)',
                accuracy_value, mean_IOU_value, sec_per_image, fps)

            return accuracy_value, mean_IOU_value

        # ===================================================
        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val,
                         max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        # Define your supervisor for running a managed session.
        # Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=None)

        # Run the managed session
        with sv.managed_session() as sess:

            for epoch in range(num_epochs):
                # At the start of every epoch, show the vital information:
                print('---- Running Training Step ----')
                logging.info('Epoch %s/%s', epoch + 1, num_epochs)
                logging.info('Current Learning Rate: %s', sess.run([lr]))

                for batch in range(int(num_batches_per_epoch)):
                    loss, training_accuracy, training_mean_IOU = train_step(sess, train_op, sv.global_step,
                                                                            metrics_op=metrics_op)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # Check the validation data only at every third of an epoch
                print('---- Running Validation Step ----')
                for batch in range(int(num_val_batch_per_epoch)):
                    validation_accuracy, validation_mean_IOU = eval_step(sess, metrics_op_val)

            # We log the final training loss
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Final Validation Accuracy: %s', validation_accuracy)
            logging.info('Final Validation Mean IOU: %s', validation_mean_IOU)

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                # Plot the predictions - check validation images only
                logging.info('Saving the images now...')
                predictions_value, annotations_value = sess.run([predictions_val, annotations_val])

                for i in range(eval_batch_size):
                    predicted_annotation = predictions_value[i]
                    annotation = annotations_value[i]

                    plt.subplot(1, 2, 1)
                    plt.imshow(predicted_annotation)
                    plt.subplot(1, 2, 2)
                    plt.imshow(annotation)
                    plt.savefig(photo_dir + "/image_" + str(i))


if __name__ == '__main__':
    run()
