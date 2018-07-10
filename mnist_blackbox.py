## Copyright (C) IBM Corp, 2017-2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import numpy as np
from six.moves import xrange

import keras
from keras import backend
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_keras import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper

from setup_mnist import MNISTModel

FLAGS = flags.FLAGS


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
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
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
    model = MNISTModel(use_log = True).model
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    if FLAGS.load_pretrain:
        tf_model_load(sess)
    else:
        train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train, verbose=False, save=True,
                    args=train_params)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy


def substitute_model(img_rows=28, img_cols=28, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: keras model
    """
    model = Sequential()

    # Find out the input shape ordering
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(input_shape=input_shape),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


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
    # model_sub = substitute_model()
    model_sub = MNISTModel(use_log = True).model
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
            if FLAGS.cached_aug:
                augs = np.load('sub_saved/mnist-aug-{}.npz'.format(rho))
                X_sub = augs['X_sub']
                Y_sub = augs['Y_sub']
            else:
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
                # cache the augmentation
                if not FLAGS.cached_aug:
                    np.savez('sub_saved/mnist-aug-{}.npz'.format(rho), X_sub = X_sub, Y_sub = Y_sub)

    return model_sub, preds_sub


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1, attack="fgsm", targeted=False):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    X_test = X_test[:FLAGS.n_attack]
    Y_test = Y_test[:FLAGS.n_attack]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
                              nb_epochs, batch_size, learning_rate)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    time_start = time.time()
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda)
    model_sub, preds_sub = train_sub_out
    time_end = time.time()
    print("Substitue model training time:", time_end - time_start)

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
    wrap = KerasModelWrapper(model_sub)


    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}

    if attack == "fgsm":
        attacker_params = {'eps': 0.4, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
        fgsm = FastGradientMethod(wrap, sess=sess)
        x_adv_sub = fgsm.generate(x, **attacker_params)
        attacker = fgsm
        adv_inputs = X_test
        ori_labels = Y_test
        print("Running FGSM attack...")
    else:
        print("Running Carlini and Wagner\'s L2 attack...")
        yname = "y"
        adv_ys = None
        # wrap = KerasModelWrapper(model)
        cwl2 = CarliniWagnerL2(wrap, back='tf', sess=sess)
        attacker_params = {'binary_search_steps': 9,
                     'max_iterations': 2000,
                     'abort_early': True,
                     'learning_rate': 0.01,
                     'batch_size': 1,
                     'initial_const': 0.01,
                     'confidence': 20}
        # generate targeted labels, 9 for each test example
        if targeted:
            adv_ys = []
            targeted_class = []
            for i in range(0, X_test.shape[0]):
                for j in range(0,10):
                    # skip the original image label
                    if j == np.argmax(Y_test[i]):
                        continue
                    adv_ys.append(np.eye(10)[j])
                    targeted_class.append(j)
            attacker_params['y_target'] = np.array(adv_ys, dtype=np.float32)
            # duplicate the inputs by 9 times
            adv_inputs = np.array([[instance] * 9 for instance in X_test],
                                  dtype=np.float32)
            adv_inputs = adv_inputs.reshape((X_test.shape[0] * 9, 28, 28, 1))
            # also update the mask
            mask = np.repeat(mask, 9)
            ori_labels = np.repeat(Y_test, 9, axis=0)
        else:
            adv_inputs = X_test
            ori_labels = Y_test
        attacker = cwl2

    if attack == "fgsm":
        # Evaluate the accuracy of the "black-box" model on adversarial examples
        accuracy = model_eval(sess, x, y, model(x_adv_sub), adv_inputs, ori_labels,
                              args=eval_params)
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute: ' + str(accuracy))
        accuracies['bbox_on_sub_adv_ex'] = accuracy

    time_start = time.time()
    # Evaluate the accuracy of the "black-box" model on adversarial examples
    x_adv_sub_np = attacker.generate_np(adv_inputs, **attacker_params)
    accuracy = model_eval(sess, x, y, bbox_preds, x_adv_sub_np, ori_labels,
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute (NP): ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy
    time_end = time.time()
    print('Attack time:', time_end - time_start)

    # Evaluate the targeted attack
    bbox_adv_predict = batch_eval(sess, [x], [bbox_preds], [x_adv_sub_np],
                          args=eval_params)[0]
    bbox_adv_class = np.argmax(bbox_adv_predict, axis = 1)
    true_class = np.argmax(ori_labels, axis = 1)
    untargeted_success = np.mean(bbox_adv_class != true_class)
    print('Untargeted attack success rate:', untargeted_success)
    accuracies['untargeted_success'] = untargeted_success
    if targeted:
        targeted_success = np.mean(bbox_adv_class == targeted_class)
        print('Targeted attack success rate:', targeted_success)
        accuracies['targeted_success'] = targeted_success

    if attack == "cwl2":
        # Compute the L2 pertubations of generated adversarial examples
        percent_perturbed = np.sum((x_adv_sub_np - adv_inputs)**2, axis=(1, 2, 3))**.5
        # print(percent_perturbed)
        # print('Avg. L_2 norm of perturbations {0:.4f}'.format(np.mean(percent_perturbed)))
        # when computing the mean, removing the failure attacks first
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(np.mean(percent_perturbed[percent_perturbed > 1e-8])))

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, bbox_preds, adv_inputs[mask], ori_labels[mask],
                          args=eval_params)
    print('Test accuracy of excluding originally incorrect labels (should be 1.0): ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex_exc_ori'] = accuracy

    if attack == "fgsm":
        # Evaluate the accuracy of the "black-box" model on adversarial examples (excluding correct)
        accuracy = model_eval(sess, x, y, model(x_adv_sub), adv_inputs[mask], ori_labels[mask],
                              args=eval_params)
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute (excluding originally incorrect labels): ' + str(accuracy))
        accuracies['bbox_on_sub_adv_ex_exc'] = accuracy

    # Evaluate the accuracy of the "black-box" model on adversarial examples (excluding correct)
    x_adv_sub_mask_np = x_adv_sub_np[mask]
    accuracy = model_eval(sess, x, y, bbox_preds, x_adv_sub_mask_np, ori_labels[mask],
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute (excluding originally incorrect labels, NP): ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex_exc'] = accuracy

    return accuracies


def main(argv=None):
    mnist_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda, attack=FLAGS.attack, targeted=FLAGS.targeted)


if __name__ == '__main__':
    # General flags
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('n_attack', -1, 'No. of images used for attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 30, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    # Flags related to attack
    flags.DEFINE_string('attack', 'cwl2', 'cwl2 = Carlini & Wagner\'s L2 attack, fgsm = Fast Gradient Sign Method')
    flags.DEFINE_bool('targeted', False, 'use targeted attack')

    # Flags related to saving/loading
    flags.DEFINE_bool('load_pretrain', False, 'load pretrained model from sub_saved/mnist-model')
    flags.DEFINE_bool('cached_aug', False, 'use cached augmentation in sub_saved')
    flags.DEFINE_string('train_dir', 'sub_saved', 'model saving path')
    flags.DEFINE_string('filename', 'mnist-model', 'cifar model name')

    os.system("mkdir -p sub_saved")

    app.run()
