---
layout: post
title: "Note 7 of Deep Learning: Regularization
categories: [Deep Learning, Machine Learning]
---

## Parameter norm penalties

A loss function with parameter norm penalty typically has a form of

$$
\tilde J(\theta; X, y) = J(\theta; X, y) + \alpha \Omega(\theta)
$$

where hyperparameter $\alpha$ control the relative importance of the norm penalty term $\Omega(\theta)$.

An important thing is that in neural network, only the **weights** are penalized in the norm penalty and the **biases** are unregularized. Each weight specifies the interaction between two variables whereas biases only control one variable, which means that leaving biases unregularized will not induce too much variance. Another reason is that regularizing biases will introduce a significant amount of underfitting.

### L2 regularization
> also called *weight decay*.

L2 regularization is also know as *ridge regression*. A loss function with L2 regularization has a form of 

$$\begin{aligned}
\tilde J(w;X,y) &= J(w;X,y) +\frac12 \mid\mid w \mid\mid _2^2\\
 &= J(w;X,y) + \frac12 w^Tw.
 \end{aligned}$$
 
### L1 regularization

The L1 regularization is given by

$$\begin{aligned}
\tilde J(w;X,y) = J(w;X,y) + \alpha \mid\mid w \mid\mid_1
\end{aligned}$$

with corresponding gradient

$$\nabla_w \tilde J(w;X,y) = \alpha sign(w) + \nabla_w J(X,y;w).$$

## Data augmentation
> Particularly effective technique for *object recognition* and *speech recognition*.

For images, operations like transforming the training images a few pixels in each direction, rotating the image or scaling the image often effectively improve generalization. But be sure that transformations that change the correct class are not applied to the dataset. For instance, we cannot rotate images $\pi/2$ in dataset containing 6 and 9 in classification tasks, which will lead to the wrong labels.

For speech recognition, we can apply *injecting noise in the input* as a form of data augmentation. It is useful because the speech inputs of neural networks usually include some noise. Adding some random noise into samples in the training dataset will improve the generalization ability of the model.

One aspect we need to pay attention to is that when comparing different machine learning algorithms, we must make sure that we perform the same data augmentation for datasets used for training each models.

## Noise robustness

The addition of noise with infinitesimal variance at the input of the model can be seen as imposing a penalty on the norm of the weights.

Another way of using noise is by adding it to the weights, which is shown to be an effective regularization strategy in neural networks, especially RNN.

### Injecting noise at the output targets

**Why**: most dataset have some amount of mistakes in the *y* labels, which is harmful to maximize $P(p \mid x)$. One way to prevent this is to explicitly model this type of noise on the labels. For example, **label smoothing** regularizes a k-classfication model with softmax by replacing the hard 0 and 1 classification targets with $\frac{\epsilon}{k}$ and $1-\frac{k-1}{k}\epsilon$ respectively.

## Semi-supervised learning

To estimate $P(y \mid x)$, we use both unlabeled examples from $P(x)$ and labeled examples from $P(x,y)$.

## Parameter tying and parameter sharing
> Intuition is that there should be some dependencies between the model parameters.

Sometimes we may have some priors that one parameter of a model should be similar to another parameter of a model. One way to simulate this prior is to add a parameter norm penalty with form of

$$
\Omega(w^A, w^B) = \mid\mid w^A - w^B \mid\mid^2_2
$$

Another viable and more popular way is to use constraints: *to force sets of parameters to be equal*, which is referred to as **parameter sharing**. One advantage of parameter sharing is that only the unique (shared) set of parameters need to be stored in memory.

## Bagging and other ensemble methods

**Bagging**: (bootstrap aggregating) train several different models separately, then each model votes on the output for test examples. Also called *model averaging*.

Different ensemble strategies include training different model with different algorithms and loss function, training the same type of model on different dataset.

Bagging involves constructing different datasets which all have same size with the original dataset by sampling with replacement from the original dataset, which means some examples may be replicated which some are missed. Different datasets lead to the differences among models.

However, model averaging is not used in benchmarking algorithms because any machine learning algorithms can benefit from it in some way. On the other hand, machine learning contests are usually won by ensemble models.

Typically, bagging is not used in neural network models, since differences in random initialization, random mini-batch selection, hyperparameters, etc, are enough to cause different members of the ensemble to make partially independent errors.

Another ensemble method is **boosting** in which an ensemble with higher capacity than the individual methods is constructed.

## Dropout

Dropout can be seen as a practical method of applying *bagging* in large neural networks.

Dropout lets each unit active with a probability of $\mu$ and died with a probability of $1-\mu$ in training step, so that a single neural network can generate any sub-network when optimizing parameters.

> The probability of each unit being 1/active is a hyperparameter, usually 0.5 for hidden layers and 0.8 for an input layer.

![For each unit, we add an extra mask $mu$ generated independently by a probability distribution to control whether it is used in this epoch of training](/assets/2018-10-31-dropout.jpg)

## Adversarial training

*Adversarial examples* are examples that are very similar to original examples, maybe just added a Gaussian noise. The differences are not visible for human observer, but significantly affect model behaviours and generate wrong prediction.

The primary cause of these adversarial examples is the linear combination. Neural networks are built on linear building blocks combined with some activation functions. If we change each input by $\epsilon$, the linear function with weights $w$ will change as much as $\epsilon\mid\mid w \mid\mid_1$, which can be very large amount and lead to wrong output if $w$ is high-dimensional.

Adversarial training solve this problem by encouraging the network to be **locally constant** in the neighbourhood of the training data, which can be seen as a way of explicitly introducing a *local constancy prior* into supervised neural networks.