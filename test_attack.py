## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import random
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


from PIL import Image


def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = (img + 0.5)*255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        use_log = False
        print('Loading model...')
        # data, model =  MNIST(), MNISTModel("models/mnist", sess, use_log)
        # data, model =  MNIST(), MNISTModel("models/mnist-distilled-100", sess, use_log)
        # data, model = CIFAR(), CIFARModel("models/cifar", sess, use_log)
        data, model = ImageNet(), InceptionModel(sess, use_log)
        print('Done...')
        batch_size = 1
        if isinstance(model, InceptionModel):
            batch_size = 10
        attack = CarliniL2(sess, model, batch_size=batch_size, initial_const = 1.0, max_iterations=1000, confidence=0, use_log=use_log)

        print('Generate data')
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=6, inception=isinstance(model, InceptionModel))
        print('Done...')
        print(inputs.shape)
        inputs = inputs[0:batch_size]
        targets = targets[0:batch_size]
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i], "original_{}.png".format(i))
            print("Classification:", np.argsort(model.model.predict(inputs[i:i+1]))[-1:-6:-1])
            print("Target:", np.argmax(targets[i]))
            print("Adversarial:")
            show(adv[i], "adversarial_{}.png".format(i))
            show(adv[i] - inputs[i], "attack_diff.png")
            
            print("Classification:", np.argsort(model.model.predict(adv[i:i+1]))[-1:-6:-1])

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

        # t = np.random.randn(28*28).reshape(1,28,28,1)
        # print(model.model.predict(t))

