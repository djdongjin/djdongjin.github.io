---
layout: post
title: "Underfit & Overfit"
categories: [Machine Learning]
---
## Underfit & Overfit
Before discussing underfit and overfit, we need to define two errors：**training error** and **generalization error**. Usually we use training set to train our model and then evaluate the performance of our model on a different dataset, named test set. So the error our model makes on training set is **training error**, and error on a random test set is **generalization error**.

Also, there is a prerequisite in statistical learning theory that training set and test set are generalized independently from the same probability distribution, which means that given a machine learning model and its parameters, we should get the same training error and generalization error. But we know that we get parameters by the training process, so there is a little difference between these two errors, and generally speaking, training error is smaller, since we use training set to train the model, rather than test set.

We should that the goal of training a model is to reduce both two errors, rather than only the training error. So except the right result in which we get two small errors, there are two other results we should avoid:

1. **Underfit**: our model has a bad performance on training set(means a big training error), let along on test set.
2. **Overfit**: out model has a better performance on training set(a smaller training error), but a bad performance on test set(a bigger generalization error)

And the key factors causing underfit and ovefit are *the complexity of model* and *the size of training set*.

## the complexity of model
Given a training set, it's obvious that the more complex a model is, the better its fitting ability is, like, a linear model can only learn linear relation, but a polynomial model can learn many relations, both linear and non-linear. So, a more complex model is more likely to get a lower training error, and to cause overfit(means it learns too many things!), and a relatively simple model is more likely to get a higher training error, which means underfit(it learns too few things because it is too simple).
![](/assets/2017-07-20-Overfit-Underfit-complexity.jpg)
## the size of training set
Given a model, it's also obvious that the size of training set will effect the performance of the model. If we have just several training examples, like, less than the number of parameters, overfit is more likely to happen because the model we get is too close to the distribution of the little training set, rather than the whole distribution. In this situation we can add more training examples to overcome overfit. On the other hand, a too big training set has no effect on generalization error, because no matter how big the training set, training examples have the same distribution.
![](/assets/2017-07-20-Overfit-Underfit-size.jpg)


## Summary
Now, we can see that underfit is caused by a too simple model, and overfit is caused both by a too complex model or by too small training set. So when you suffer underfit, you can try to use a more complex model, and when you suffer overfit, you can try to add more training examples or use a relatively simple model.


