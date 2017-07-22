## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
from numba import jit
import math

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 0.55     # the initial constant c to pick as a first guess

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # for i in range(batch_size):
    #    grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.002 
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
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

class BlackBoxL2:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 use_log = False, use_tanh = False):
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
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.use_tanh = use_tanh

        self.repeat = binary_search_steps >= 10

        # each batch has a different modifier value (see below) to evaluate
        shape = (None,image_size,image_size,num_channels)
        single_shape = (image_size, image_size, num_channels)
        
        # the variable we're going to optimize over
        # support multiple batches
        self.modifier = tf.placeholder(tf.float32, shape=shape)
        # the real variable, initialized to 0
        # self.real_modifier = np.load('best.model.npy').reshape((1,) + single_shape)
        self.real_modifier = np.zeros((1,) + single_shape, dtype=np.float32)

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
            self.newimg = tf.tanh(self.modifier + self.timg)/2
        else:
            self.newimg = self.modifier + self.timg
        
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
                loss1 = tf.maximum(- tf.log(self.other), - tf.log(self.real))
            else:
                # if targetted, optimize for making the other class (real) most likely
                loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
        else:
            if use_log:
                loss1 = tf.log(self.real)
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

        # set random seed
        np.random.seed(1216)

        # prepare the list of all valid variables
        var_size = image_size * image_size * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.used_var_list = np.zeros(var_size, dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size)
        self.modifier_down = np.zeros(var_size)

        # random permutation for coordinate update
        self.perm = np.random.permutation(var_size)
        self.perm_index = 0

        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype = np.float32)
        # for testing
        self.grad_op = tf.gradients(self.loss, self.modifier)

    def fake_blackbox_optimizer(self):
        true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
        # ADAM update
        grad = true_grads[0].reshape(-1)
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
        return losses[0], l2s[0], scores[0], nimgs[0]


    def blackbox_optimizer(self, iteration):
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.batch_size * 2 + 1, axis=0)
        var_size = self.real_modifier.size
        # print(s, "variables remaining")
        # var_indice = np.random.randint(0, self.var_list.size, size=self.batch_size)
        # var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False)
        # var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False, p = self.sample_prob)
        # indice = self.var_list[var_indice]
        indice = self.var_list
        # regenerate the permutations if we run out
        # if self.perm_index + self.batch_size >= var_size:
        #     self.perm = np.random.permutation(var_size)
        #     self.perm_index = 0
        # indice = self.perm[self.perm_index:self.perm_index + self.batch_size]
        # b[0] has the original modifier, b[1] has one index added 0.0001
        for i in range(self.batch_size):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.001
        losses, l2s, scores, nimgs = self.sess.run([self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: var})
        t_grad = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        self.grad = t_grad[0].reshape(-1)
        coordinate_ADAM(losses, indice, self.grad, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # adjust sample probability, sample around the points with large gradient
        med = np.sort(np.absolute(self.grad))[-10]
        ns = self.model.image_size
        nc = self.model.num_channels
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        base_prob = 1.0 / var_size * 1.0
        def get_coo(c, y, x):
            return c * (ns * ns) + y * ns + x
        for i in range(self.batch_size):
            if self.grad[i] > med:
                # this is an important pixel, increase the sample probability of nearby pixels
                ind = indice[i]
                # convert to 2D 
                ind_c = ind // (ns * ns)
                ind_xy = ind % (ns * ns)
                ind_y = ind_xy // ns
                ind_x = ind_xy % ns
                # print(i, ind_c, ind_y, ind_x, self.grad[i])
                cnt = 0
                if ind_c != 0:
                    self.sample_prob[get_coo(ind_c - 1, ind_y, ind_x)] = base_prob
                    cnt += 1
                    if ind_x != 0:
                        self.sample_prob[get_coo(ind_c - 1, ind_y, ind_x - 1)] = base_prob
                        cnt += 1
                    if ind_x != ns - 1:
                        self.sample_prob[get_coo(ind_c - 1, ind_y, ind_x + 1)] = base_prob
                        cnt += 1
                    if ind_y != 0:
                        self.sample_prob[get_coo(ind_c - 1, ind_y - 1, ind_x)] = base_prob
                        cnt += 1
                    if ind_y != ns - 1:
                        self.sample_prob[get_coo(ind_c - 1, ind_y + 1, ind_x)] = base_prob
                        cnt += 1
                if ind_c != nc - 1:
                    self.sample_prob[get_coo(ind_c + 1, ind_y, ind_x)] = base_prob
                    cnt += 1
                    if ind_x != 0:
                        self.sample_prob[get_coo(ind_c + 1, ind_y, ind_x - 1)] = base_prob
                        cnt += 1
                    if ind_x != ns - 1:
                        self.sample_prob[get_coo(ind_c + 1, ind_y, ind_x + 1)] = base_prob
                        cnt += 1
                    if ind_y != 0:
                        self.sample_prob[get_coo(ind_c + 1, ind_y - 1, ind_x)] = base_prob
                        cnt += 1
                    if ind_y != ns - 1:
                        self.sample_prob[get_coo(ind_c + 1, ind_y + 1, ind_x)] = base_prob
                        cnt += 1
                if ind_x != 0:
                    self.sample_prob[get_coo(ind_c, ind_y, ind_x - 1)] = base_prob
                    cnt += 1
                if ind_x != ns - 1:
                    self.sample_prob[get_coo(ind_c, ind_y, ind_x + 1)] = base_prob
                    cnt += 1
                if ind_y != 0:
                    self.sample_prob[get_coo(ind_c, ind_y - 1, ind_x)] = base_prob
                    cnt += 1
                if ind_y != ns - 1:
                    self.sample_prob[get_coo(ind_c, ind_y + 1, ind_x)] = base_prob
                    cnt += 1
                # self.sample_prob[indice] /= (cnt / 2)
        # renormalize
        self.sample_prob /= np.sum(self.sample_prob)

        # if the gradient is too small, do not optimize on this variable
        # self.var_list = np.delete(self.var_list, indice[np.abs(self.grad) < 5e-3])
        # reset the list every 10000 iterations
        # if iteration%200 == 0:
        #    print("{} variables remained at last stage".format(self.var_list.size))
        #    var_size = self.real_modifier.size
        #    self.var_list = np.array(range(0, var_size))
        return losses[0], l2s[0], scores[0], nimgs[0]

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
            r.extend(self.attack_one(imgs[i], targets[i]))
        return np.array(r)

    # only accepts 1 image at a time. Batch is used for gradient evaluations at different points
    def attack_one(self, img, lab):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10

        # set the upper and lower bounds for the modifier
        self.modifier_up = 0.5 - img.reshape(-1)
        self.modifier_down = -0.5 - img.reshape(-1)

        # the best l2, score, and image attack
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

            # clear the modifier
            # self.real_modifier.fill(0.0)
            # use the current best model
            # np.copyto(self.real_modifier, o_bestattack - img)
            # use the model left by last constant change
            
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier}))
                    # np.save('black_iter_{}'.format(iteration), self.real_modifier)

                # perform the attack 
                # l, l2, score, nimg = self.fake_blackbox_optimizer()
                l, l2, score, nimg = self.blackbox_optimizer(iteration)

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
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
                    o_bestl2 = l2
                    o_bestscore = np.argmax(score)
                    o_bestattack = nimg

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
        return o_bestattack

