## l2_attack_black.py -- attack a black-box network optimizing for l_2 distance
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import os
import tensorflow as tf
import numpy as np
import scipy.misc
from numba import jit
import math
import time

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 0.5      # the initial constant c to pick as a first guess

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # def sign(x):
    #     return np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print('grad', grad)
    # print('hess', hess)
    # hess[hess < 0] = 1.0
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # negative hessian cannot provide second order information, just do a gradient descent
    hess[hess < 0] = 1.0
    # hessian too small, could be numerical problems
    hess[hess < 0.1] = 0.1
    # print(hess)
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print('delta', old_val - m[indice])
    m[indice] = old_val
    # print(m[indice])

@jit(nopython=True)
def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print(grad)
    # print(hess)
    # positive hessian, using newton's method
    hess_indice = (hess >= 0)
    # print(hess_indice)
    # negative hessian, using ADAM
    adam_indice = (hess < 0)
    # print(adam_indice)
    # print(sum(hess_indice), sum(adam_indice))
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # print(adam_indice)
    # Newton's Method
    m = real_modifier.reshape(-1)
    old_val = m[indice[hess_indice]] 
    old_val -= lr * grad[hess_indice] / hess[hess_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[hess_indice]]), down[indice[hess_indice]])
    m[indice[hess_indice]] = old_val
    # ADMM
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch[adam_indice]))) / (1 - np.power(beta1, epoch[adam_indice]))
    old_val = m[indice[adam_indice]] 
    old_val -= lr * corr * mt[adam_indice] / (np.sqrt(vt[adam_indice]) + 1e-8)
    # old_val -= lr * grad[adam_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[adam_indice]]), down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1
    # print(m[indice])

class BlackBoxL2:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS, print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 use_log = False, use_tanh = True, use_resize = False, adam_beta1 = 0.9, adam_beta2 = 0.999, reset_adam_after_found = False,
                 solver = "adam", save_ckpts = "", load_checkpoint = "", start_iter = 0,
                 init_size = 32, use_importance = True):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        self.use_importance = use_importance
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size

        self.use_tanh = use_tanh
        self.use_resize = use_resize
        self.save_ckpts = save_ckpts
        if save_ckpts:
            os.system("mkdir -p {}".format(save_ckpts))

        self.repeat = binary_search_steps >= 10

        # each batch has a different modifier value (see below) to evaluate
        # small_shape = (None,self.small_x,self.small_y,num_channels)
        shape = (None,image_size,image_size,num_channels)
        single_shape = (image_size, image_size, num_channels)
        small_single_shape = (self.small_x, self.small_y, num_channels)
        
        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size
        if self.use_resize:
            self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
            # scaled up image
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_size, image_size])
            # operator used for resizing image
            self.resize_size_x = tf.placeholder(tf.int32)
            self.resize_size_y = tf.placeholder(tf.int32)
            self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
            self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y])
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            # no resize
            self.scaled_modifier = self.modifier
        # the real variable, initialized to 0
        self.load_checkpoint = load_checkpoint
        if load_checkpoint:
            # if checkpoint is incorrect reshape will fail
            print("Using checkpint", load_checkpoint)
            self.real_modifier = np.load(load_checkpoint).reshape((1,) + small_single_shape)
        else:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        # self.real_modifier = np.random.randn(image_size * image_size * num_channels).astype(np.float32).reshape((1,) + single_shape)
        # self.real_modifier /= np.linalg.norm(self.real_modifier) 
        # these are variables to be more efficient in sending data to tf
        # we only work on 1 image at once; the batch is for evaluation loss at different modifiers
        self.timg = tf.Variable(np.zeros(single_shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, single_shape)
        self.assign_tlab = tf.placeholder(tf.float32, num_labels)
        self.assign_const = tf.placeholder(tf.float32)
        
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # broadcast self.timg to every dimension of modifier
        if use_tanh:
            self.newimg = tf.tanh(self.scaled_modifier + self.timg)/2
        else:
            self.newimg = self.scaled_modifier + self.timg
        
        # prediction BEFORE-SOFTMAX of the model
        # now we have output at #batch_size different modifiers
        # the output should have shape (batch_size, num_labels)
        self.output = model.predict(self.newimg)
        
        # distance to the input data
        if use_tanh:
            self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2), [1,2,3])
        else:
            self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.timg), [1,2,3])
        
        # compute the probability of the label class versus the maximum other
        # self.tlab * self.output selects the Z value of real class
        # because self.tlab is an one-hot vector
        # the reduce_sum removes extra zeros, now get a vector of size #batch_size
        self.real = tf.reduce_sum((self.tlab)*self.output,1)
        # (1-self.tlab)*self.output gets all Z values for other classes
        # Because soft Z values are negative, it is possible that all Z values are less than 0
        # and we mistakenly select the real class as the max. So we minus 10000 for real class
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            if use_log:
                # loss1 = - tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
            else:
                # if targetted, optimize for making the other class (real) most likely
                loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
        else:
            if use_log:
                # loss1 = tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(self.other + 1e-30))
            else:
                # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)

        # sum up the losses (output is a vector of #batch_size)
        self.loss2 = self.l2dist
        self.loss1 = self.const*loss1
        self.loss = self.loss1+self.loss2
        
        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.used_var_list = np.zeros(var_size, dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype = np.float32)
        self.modifier_down = np.zeros(var_size, dtype = np.float32)

        # random permutation for coordinate update
        self.perm = np.random.permutation(var_size)
        self.perm_index = 0

        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        # self.beta1 = 0.8
        # self.beta2 = 0.99
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        self.stage = 0
        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype = np.float32)
        self.hess = np.zeros(batch_size, dtype = np.float32)
        # for testing
        self.grad_op = tf.gradients(self.loss, self.modifier)
        # compile numba function
        # self.coordinate_ADAM_numba = jit(coordinate_ADAM, nopython = True)
        # self.coordinate_ADAM_numba.recompile()
        # print(self.coordinate_ADAM_numba.inspect_llvm())
        # np.set_printoptions(threshold=np.nan)
        # set solver
        solver = solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            print("unknown solver", solver)
            self.solver = coordinate_ADAM
        print("Using", solver, "solver")

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i+size, j:j+size] = np.max(image[i:i+size, j:j+size])
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double = False):
        prev_modifier = np.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0]*2, old_shape[1]*2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype = np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:,:,i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                prob[:,:,i] = scipy.misc.imresize(image_pool, 2.0, 'nearest', mode = 'F')
            else:
                prob[:,:,i] = image_pool
        prob /= np.sum(prob)
        return prob


    def resize_img(self, small_x, small_y, reset_only = False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(self.real_modifier)
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x, self.resize_size_y: self.small_y, self.resize_input: self.real_modifier})
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

    def fake_blackbox_optimizer(self):
        true_grads, losses, l2s, loss1, loss2, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
        # ADAM update
        grad = true_grads[0].reshape(-1)
        # print(true_grads[0])
        epoch = self.adam_epoch[0]
        mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grad)
        corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)
        # print(grad.shape, mt.shape, vt.shape, self.real_modifier.shape)
        # m is a *view* of self.real_modifier
        m = self.real_modifier.reshape(-1)
        # this is in-place
        m -= self.LEARNING_RATE * corr * (mt / (np.sqrt(vt) + 1e-8))
        self.mt = mt
        self.vt = vt
        # m -= self.LEARNING_RATE * grad
        if not self.use_tanh:
            m_proj = np.maximum(np.minimum(m, self.modifier_up), self.modifier_down)
            np.copyto(m, m_proj)
        self.adam_epoch[0] = epoch + 1
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]


    def blackbox_optimizer(self, iteration):
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.batch_size * 2 + 1, axis=0)
        var_size = self.real_modifier.size
        # print(s, "variables remaining")
        # var_indice = np.random.randint(0, self.var_list.size, size=self.batch_size)
        if self.use_importance:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False, p = self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False)
        indice = self.var_list[var_indice]
        # indice = self.var_list
        # regenerate the permutations if we run out
        # if self.perm_index + self.batch_size >= var_size:
        #     self.perm = np.random.permutation(var_size)
        #     self.perm_index = 0
        # indice = self.perm[self.perm_index:self.perm_index + self.batch_size]
        # b[0] has the original modifier, b[1] has one index added 0.0001
        for i in range(self.batch_size):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
        losses, l2s, loss1, loss2, scores, nimgs = self.sess.run([self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.newimg], feed_dict={self.modifier: var})
        # losses = self.sess.run(self.loss, feed_dict={self.modifier: var})
        # t_grad = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        # self.grad = t_grad[0].reshape(-1)
        # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        # self.coordinate_ADAM_numba(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh, true_grads)
        # coordinate_Newton(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_Newton_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        self.solver(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # adjust sample probability, sample around the points with large gradient
        if self.save_ckpts:
            np.save('{}/iter{}'.format(self.save_ckpts, iteration), self.real_modifier)

        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            # self.sample_prob = self.get_new_prob(tmp_mt.reshape(self.real_modifier.shape))
            self.sample_prob = self.sample_prob.reshape(var_size)

        # if the gradient is too small, do not optimize on this variable
        # self.var_list = np.delete(self.var_list, indice[np.abs(self.grad) < 5e-3])
        # reset the list every 10000 iterations
        # if iteration%200 == 0:
        #    print("{} variables remained at last stage".format(self.var_list.size))
        #    var_size = self.real_modifier.size
        #    self.var_list = np.array(range(0, var_size))
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]
        # return losses[0]

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        # we can only run 1 image at a time, minibatches are used for gradient evaluation
        for i in range(0,len(imgs)):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i], targets[i]))
        return np.array(r)

    # only accepts 1 image at a time. Batch is used for gradient evaluations at different points
    def attack_batch(self, img, lab):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # remove the extra batch dimension
        if len(img.shape) == 4:
            img = img[0]
        if len(lab.shape) == 2:
            lab = lab[0]
        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10

        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)

        # set the upper and lower bounds for the modifier
        if not self.use_tanh:
            self.modifier_up = 0.5 - img.reshape(-1)
            self.modifier_down = -0.5 - img.reshape(-1)

        # clear the modifier
        if not self.load_checkpoint:
            if self.use_resize:
                self.resize_img(self.resize_init_size, self.resize_init_size, True)
            else:
                self.real_modifier.fill(0.0)

        # the best l2, score, and image attack
        o_best_const = CONST
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = img
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
    
            bestl2 = 1e10
            bestscore = -1

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: img,
                                       self.assign_tlab: lab,
                                       self.assign_const: CONST})

            # use the current best model
            # np.copyto(self.real_modifier, o_bestattack - img)
            # use the model left by last constant change
            
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
            if not self.load_checkpoint:
                if self.use_resize:
                    self.resize_img(self.resize_init_size, self.resize_init_size, True)
                else:
                    self.real_modifier.fill(0.0)
            # reset ADAM status
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            if self.solver_name != "fake_zero":
                multiplier = 24
            for iteration in range(self.start_iter, self.MAX_ITERATIONS):
                if self.use_resize:
                    if iteration == 2000:
                    # if iteration == 2000 // 24:
                        self.resize_img(64,64)
                    if iteration == 10000:
                    # if iteration == 2000 // 24 + (10000 - 2000) // 96:
                        self.resize_img(128,128)
                    # if iteration == 200*30:
                    # if iteration == 250 * multiplier:
                    #     self.resize_img(256,256)
                # print out the losses every 10%
                if iteration%(self.print_every) == 0:
                    # print(iteration,self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier}))
                    loss, real, other, loss1, loss2 = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier})
                    print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, loss[0], real[0], other[0], loss1[0], loss2[0]))
                    sys.stdout.flush()
                    # np.save('black_iter_{}'.format(iteration), self.real_modifier)

                attack_begin_time = time.time()
                # perform the attack 
                if self.solver_name == "fake_zero":
                    l, l2, loss1, loss2, score, nimg = self.fake_blackbox_optimizer()
                else:
                    l, l2, loss1, loss2, score, nimg = self.blackbox_optimizer(iteration)
                # l = self.blackbox_optimizer(iteration)

                if self.solver_name == "fake_zero":
                    eval_costs += np.prod(self.real_modifier.shape)
                else:
                    eval_costs += self.batch_size

                # reset ADAM states when a valid example has been found
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    # we have reached the fine tunning point
                    # reset ADAM to avoid overshoot
                    if self.reset_adam_after_found:
                        self.mt.fill(0.0)
                        self.vt.fill(0.0)
                        self.adam_epoch.fill(1)
                    self.stage = 1
                last_loss1 = loss1

                # check if we should abort search if we're getting nowhere.
                # if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    if l > prev*.9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = l

                # adjust the best result found so far
                # the best attack should have the target class with the largest value,
                # and has smallest l2 distance
                if l2 < bestl2 and compare(score, np.argmax(lab)):
                    bestl2 = l2
                    bestscore = np.argmax(score)
                if l2 < o_bestl2 and compare(score, np.argmax(lab)):
                    # print a message if it is the first attack found
                    if o_bestl2 == 1e10:
                        print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, l2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2, l2))
                        sys.stdout.flush()
                    o_bestl2 = l2
                    o_bestscore = np.argmax(score)
                    o_bestattack = nimg
                    o_best_const = CONST

                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed
            if compare(bestscore, np.argmax(lab)) and bestscore != -1:
                # success, divide const by two
                print('old constant: ', CONST)
                upper_bound = min(upper_bound,CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound)/2
                print('new constant: ', CONST)
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                print('old constant: ', CONST)
                lower_bound = max(lower_bound,CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound)/2
                else:
                    CONST *= 10
                print('new constant: ', CONST)

        # return the best solution found
        return o_bestattack, o_best_const

