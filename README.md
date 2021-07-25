**As requested by IBM, this repository is moved to https://github.com/IBM/ZOO-Attack, but we aim to keep both repositories synced up.** The code is released under Apache License v2.

ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks 
=====================================

ZOO is a **z**eroth **o**rder **o**ptimization based attack to attack deep
neural networks (DNNs).  We propose an effective black-box attack that only
requires access to the input (images) and the output (confidence scores) of a
targeted DNN. We formularize the attack as an optimization problem (similar as
Carlini and Wagner's attack), and propose a new loss function suitable for the
black-box setting.  We use zeroth order stochastic coordinate descent to
optimize on the target DNN directly, along with dimension reduction,
hierarchical attack and importance sampling techniques to make the attack
efficient. No transferability or substitute model is required.

There are two variants of ZOO, ZOO-ADAM and ZOO-Newton, corresponding to
different solvers (ADAM and Newton) to find the best coordinate update.
In practice ZOO-ADAM usually works better with fine-tuned parameters,
but ZOO-Newton is more stable when close to the optimal solution.

The experiment code is based on Carlini and Wagner's L2 attack, with
zeroth order optimizer added in `l2_attack_black.py`. The inception model
is updated to a new version (`inception_v3_2016_08_28.tar.gz`), and 
an unified interface `test_all.py` is added.

For more details, please see our paper:

[ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models](https://arxiv.org/abs/1708.03999)
by Pin-Yu Chen\*, Huan Zhang\*, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh

\* Equal contribution


Setup and train models
-------------------------------------

The code is tested with python3 and TensorFlow v1.2 and v1.3. The following
packages are required:

```
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py numba
```

Prepare the MNIST and CIFAR-10 data and models for attack:

```
python3 train_models.py
```

To download the inception model:

```
python3 setup_inception.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

http://download.huan-zhang.com/datasets/adv/img.tar.gz


and put the `imgs` folder in `../imagenetdata`. This path can be changed
in `setup_inception.py`.

Run attacks
--------------------------------------

An unified attack interface, `test_all.py` is provided. Run `python3 test_all.py -h`
to get a list of arguments and help.

The following are some examples of attacks:

Run ZOO black-box targeted attack, on the mnist dataset with 200 images, with
ZOO-ADAM solver, search for best regularization constant for 9 iterations, and
save attack images to folder `black_results`. To run on the CIFAR-10 dataset,
replace 'mnist' with 'cifar10'.

```
python3 test_all.py -a black -d mnist -n 200 --solver adam -b 9 -s "black_results"
```

Run Carlini and Wagner's white-box targeted attack, on the mnist dataset with
200 images, using the Z (logits) value in objective (only available in
white-box setting), search for best regularization constant for 9 iterations,
and save attack images to folder `white_results`.

```
python3 test_all.py -a white -d mnist -n 200 --use_zvalue -b 9 -s "white_results"
```

Run ZOO black-box *untargeted* attack, on the imagenet dataset with 150 images, with ZOO-ADAM
solver, do not binary search the regularization parameter (i.e., search only 1
time), and set the initial regularization parameter to a fixed value (10.0). Use
attack-space dimension reduction with image resizing, and reset ADAM states
when the first attack is found.  Run a maximum of 1500 iterations, and print
out loss every 10 iterations. Save attack images to folder `imagenet_untargeted`.

```
python3 test_all.py --untargeted -a black -d imagenet -n 150 --solver adam -b 1 -c 10.0 --use_resize --reset_adam -m 1500 -p 10 -s "imagenet_untargeted"
```

Run ZOO black-box targeted attack, on the imagenet dataset, with the 69th image
only.  Set the regularization parameter to 10.0 and do not binary search. Use
attack-space dimension reduction and hierarchical attack with image resizing,
and reset ADAM states when the first attack is found.  Run a maximum of 20000
iterations, and print out loss every 10 iterations. Save attack images to
folder `imagenet_all_tricks_img69`.


```
python3 test_all.py -a black --solver adam -d imagenet -f 69 -n 1 -c 10.0 --use_resize --reset_adam -m 20000 -p 10 -s "imagenet_all_tricks_img69"
```

Importance sampling is on by default for ImageNet data, and can be turned off by
`--uniform` option. To change the hierarchical attack dimension scheduling,
change `l2_attack_black.py`, near line 580.

