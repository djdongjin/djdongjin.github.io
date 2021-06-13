---
layout: post
title: "Note 8 of Deep Learning: Optimization"
categories: [Machine Learning, Deep Learning]
---

> Optimization techniques for neural network training.

## Challenges in Neural Network Optimization

### Local Minima

A convex optimization problem can be seen as finding a local minimum. Any local minimum is guaranteed to be a global minimum because it is either a real global minimum or a local minimum in a flat region.

With non-convex models such as neural networks, it's extremely harmful if we encounter a local minimum that is significantly different with the global minimum.

**Saddle points**: a kind of point with 0 gradient. Some points around a saddle point have greater cost than the saddle point whereas others have a lower cost.

### Cliffs and Exploding Gradients

Neural networks with many layers often have steep regions resembling **cliffs**, which may be caused by the multiplication of several large weights together, and may also cause that the gradient update step moves the parameters extremely far.

The cliff can be solved by heuristic *gradient clipping* which will reduce the step size if the gradient is too large.

## Basic Algorithms

### Stochastic Gradient Descent (SGD)

SGD works as follows

![SGD](/assets/2018-11-02-SGD.jpg = 250x250)

An important point is that we should decrease the learning rate $\epsilon_k$ over time in SGD, because we introduces some noise by randomly sampling $m$ training examples from the whole dataset. In such a situation, the gradient may not vanish even when we arrive at a minimum, where as the gradient of batch gradient descent will become small or 0. So we can use fixed learning rate $\epsilon$ in batch gradient descent, but in SGD we typically decay the learning rate linearly until iteration $\tau$

$$
\epsilon_k = (1-\alpha)\epsilon_0 + \alpha\epsilon_\tau
$$

where $\alpha=\frac{k}{\tau}$. $\epsilon$ will be a constant after iteration $\tau$.

By now we have 3 parameters, $\tau, \epsilon_0, \epsilon_\tau$. $\tau$ is usually set to the number of iterations required to make a few hundred passes through the whole training set. $\epsilon_\tau$ is set to roughly 1% of the value of $\epsilon_0$. For $\epsilon_0$, it is higher than the learning rate that yields the best performance after the ﬁrst 100 iterations. So it is best to monitor the first several iterations and use a learning rate that is higher than the best-performing learning rate but not so high that it causes severe instability.

### Momentum

Momentum is designed to accelerate learning especially when the method encountered *high curvature, small but consistent gradients, noisy gradients*.

Momentum introduces a velocity variable $v$ that is the direction and speed at which parameters move through parameter space, and set to an exponentially decaying average of the negative gradient.

The update rule of momentum is given by

$$\begin{aligned}
v &\gets \alpha v - \epsilon\nabla_\theta \frac{1}{m}\sum_{i=1}^m L(f(x^{i)};\theta),y^{(\theta)}),\\
\theta &\gets \theta + v,
\end{aligned}$$

where the velocity $v$ accumulates the gradient elements $\nabla_\theta \frac{1}{m}\sum_{i=1}^m L(f(x^{i)};\theta),y^{(\theta)})$. $\alpha \in [0,1)$ determines how quickly the contributions of previous gradients exponentially decay. The larger $\alpha$ is relative to $\epsilon$, the more previous gradients affect the current direction.

The SGD algorithm with momentum works as follows

![SGD with Momentum](/assets/2018-11-02-momentum.jpg)

## Algorithms with Adaptive Learning Rates

### AdaGrad

AdaGrad, shown below, *individually* adapts the learning rates of all parameters by scaling them inversely proportional to the square root of the sum of all of their historical squared values, so that parameters with larger partial derivative of the loss have a rapid decrease in their learning rate and vice versus.

AdaGrad can enjoys some desirable theoretical properties in convex optimization setting, whereas the accumulation of squared gradients *from the beginning of training* can result in a premature and excessive decrease in the effective learning rate when training deep neural networks.

![AdaGrad](/assets/2018-11-02-AdaGrad.jpg)

### RMSProp

Based on AdaGrad, RMSProp performs better in the non-convex setting by changing the gradient accumulation into an **exponentially weighted moving average**.

By using *exponentially weighted moving average*, RMSProp can discard history from the very past so that it can converge rapidly after finding a convex bowl. It also introduces a new hyperparameter $\rho$ that controls the length scale of the moving average.

Empirically, RMSProp is an effective and practical optimization algorithm for neural networks. It is also one of the go-to optimization algorithms being employed routinely by deep learning practitioners.

![RMSProp](/assets/2018-11-02-RMSProp.jpg)

### Adam
> The name derives from "adaptive moments"

Adam can be seen as a combination of RMSProp and momentum.

![Adam](/assets/2018-11-02-Adam.jpg)

## Optimization Strategies and Meta-Algorithms

### Batch Normalization
> A method of adaptive reparametrization that reduces the problem of coordinating updates across many layers.

Batch normalization can be applied to any input or hidden layer in a network. Suppose $H$ bis a mini-batch of activations of the layer to normalize, with the activations for each example appearing in a row of the matrix. We replace $H$ with following as normalization:

$$
H' = \frac{H-\mu}{\sigma}.
$$

where $\mu$ is a vector containing the mean of each unit, and $\sigma$ is a vector containing the standard deviation of each unit. Then the rest of the network operates on $H'$ in the same way the the original network operated on $H$.

At training time, we have:

$$\begin{aligned}
\mu &= \frac{1}{m}\sum_i H_i,\\
\sigma &= \sqrt{\delta+\frac{1}{m}\sum_i(H-\mu)_i^2}.
\end{aligned}$$

* [ ] Batch Normalization.