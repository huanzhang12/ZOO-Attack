## Copyright (C) IBM Corp, 2017-2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange

import keras
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.datasets import cifar10
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper

from setup_cifar import CIFARModel

FLAGS = flags.FLAGS

def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def setup_tutorial():
    """
    Helper function to check correct configuration of tf and keras for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    return True


def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for CIFAR
    :param y: the ouput placeholder for CIFAR
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = CIFARModel(use_log = True).model
    # model = CIFARModel(use_log = True).model
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an CIFAR model
    if FLAGS.load_pretrain:
        # use the restored CIFAR model
        tf_model_load(sess)
    else:
        train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train, verbose=True, save=True,
                    args=train_params)
  
    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy



def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = CIFARModel(use_log = True).model
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    init_all=False, verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def cifar_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=50, holdout=150, data_aug=6,
                   nb_epochs_s=50, lmbda=0.1):
    """
    CIFAR tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """
    keras.layers.core.K.set_learning_phase(0)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session and set as Keras backend session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)

    # Get CIFAR data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
                              nb_epochs, batch_size, learning_rate)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params)
    accuracies['sub'] = acc
    print('substitution model accuracy:', acc)

    # Find the correctly predicted labels
    original_predict = batch_eval(sess, [x], [bbox_preds], [X_test],
                          args=eval_params)[0]
    original_class = np.argmax(original_predict, axis = 1)
    true_class = np.argmax(Y_test, axis = 1)
    mask = true_class == original_class
    print(np.sum(mask), "out of", mask.size, "are correct labeled,", len(X_test[mask]))  
    
    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.4, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    wrap = KerasModelWrapper(model_sub)
    fgsm = FastGradientMethod(wrap, sess=sess)

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy
    
    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, bbox_preds, X_test[mask], Y_test[mask],
                          args=eval_params)
    print('Test accuracy of excluding originally incorrect labels: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex_exc_ori'] = accuracy
    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test[mask], Y_test[mask],
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute (excluding originally incorrect labels): ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex_exc'] = accuracy

    return accuracies


def main(argv=None):
    print(cifar_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda))


if __name__ == '__main__':
    # General flags
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 50, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    # Flags related to saving/loading
    flags.DEFINE_bool('load_pretrain', False, 'load pretrained model from sub_saved/cifar-model')
    flags.DEFINE_string('train_dir', 'sub_saved', 'model saving path')
    flags.DEFINE_string('filename', 'cifar-model', 'cifar model name')
    app.run()
