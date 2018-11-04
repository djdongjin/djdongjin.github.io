---
layout: post
title: "Note 9 of Deep Learning: Convolutional Neural Network"
categories: [Deep Learning, Machine Learning, CNN]
---

## Convolution Operation

Suppose we want to locate a spaceship at time t, which can be described by $x(t)$, but sometimes our monitor may have some noise. To obtain a less noisy estiamte of the spaceship's position, we may want to average together several measurements, in which more recent measurements are more relevant and have larger weights. So we need another *weight function* $w(a)$ to assign weights to each position/moment. By now we can obtain a new function that provides smoothed estimate of the position of the spaceship:

$$\begin{aligned}
s(t) = \int x(a)w(t-a)da.
\end{aligned}$$ 

which is called **convolution**. The first argument $x$ is called *input* and the second argument $w$ is called *kernel*.

If the input $x$ is discrete, we can use *convolution* in a discrete setting, and convolution is commutative, so we can write it as:

$$\begin{aligned}
S(i,j)& = (I*K)(i,j) = \sum_m\sum_n I(m,n)K(i-m,j-n)\\
&= (K*I)(i,j) = \sum_m\sum_n K(m,n)I(i-m,j-n).
\end{aligned}$$

## Motivation

Three ideas motivates us to use convolutions: *sparse interactions/weights, parameter sharing, equivariant representations*.

### Sparse Interactions

> by making the kernel smaller that the input.

In a full-connected neural network, every output unit interacts with every input unit, and is generated using matrix multiplication by a matrix of parameters with a separate parameter describing the interaction between each input and output unit.

CNNs, however, have sparse interactions. For example, the input image has thousands of pixels, but we can detect small, meaningful features like edges with kernels that occupy only tens or hundreds of pixels, through which we can store fewer parameters and use fewer computing operations.

Suppose we have $m$ input units and $n$ output units in one layer, a matrix multiplication of a full-connected neural network typically need $m \times n$ parameters and has $O(m \times n)$ runtime. On the contrary, in a CNN setting, if we limit the kernel size to $k, k \lt m$, we only need $k \times m$ parameters, or $k$ because of parameter sharing as below, and have only $O(k \times n)$ runtime. 

![Sparse connection v.s. full connection](/assets/2018-11-03-sparse-connection.jpg)

### Parameter Sharing

> means one parameter will be used in more than one function.

In a feedforward neural network, each weight is used exactly one time when computing the output of the layer, $w_{i,j} \times x_i$, whereas a weight in a CNN will be used for generating every elements of the output.

Recall that through sparse interactions, we have $k \times n$ paramters, now we  further reduce the number of parameters to $k$ by parameter sharing, though the runtime is still $O(m \times n)$.

![Parameter sharing v.s. independent parameters](/assets/2018-11-03-parameter-sharing.jpg)

### Equivariant Representations

A function $f(x)$ is equivariant to a function $g(x)$ if $f(g(x)) = g(f(x))$.  

* [ ] To do. 

## Pooling

A CNN layer typically is a block consisting of three different layers. The first one is a convolution layer, the second one is a activation function layer that performs some non-linear transformation, the third one is a pooling layer that modifies the output of the layer by replacing the output at a certain location with a summary statistic of the nearby outputs, such as max, average, min, etc.

Pooling layers help to make representation become approximately *invariant* to small translations of inputs.Invariance to local translation is very useful, especially if we care more about if some features exist instead of its exact locations.

Since pooling can be seen as a statistic of a pixel as well as its neighbours, we no longer need all these pixels in the output. This also means that we can see pooling as ways of downsampling.

Some straightforward pooling methods, such as max pooling, average pooling, don't need any parameters. But pooling can also complicate some kinds of neural networks that use top-down information, such as Boltzmann machines and antoencoders, which will introduces new parameters.

## Variants of the Basic Convolution Function

* [ ] placeholder. 

## Random or Unsupervised Features

The most expensive part of CNN training is *learning features*. The output layer is not expensive due to the small number of feature inputs after passing through several pooling layers. One way to reduce the cost of convolutional network training is *using features that are not trained in a supervised fashion*.

Three basic strategies can be used for obtaining convolution kernels without supervised learning. One is to *initialize them randomly*. Another is to *design them by hand*, for example by setting kernels to detect edges or corners. The last one is to *learn kernels with unsupervised learning*, for example, applying *k-means* clustering to small image patchs and then use each learned centroid as a convolution kernel.

