---
layout: post
title: "Note 3 of PRML: Linear Models for Regression"
categories: [PRML, Machine Learning]
---

> Linear models: linear functions of the adjustable **parameters** (instead of *input variables*, which is just the simplest form of linear models).

## From linear regression to linear models for regression

Generally, linear regression models have a form of 

$$
y(x, w) = \sum_{i=0}^{M-1}w_ix_i
$$

where $x_0=1$, corresponding *bias* $w_0$.

We can extend this class of models by adding some nonlinearity to input $X$ and only keeping the linearity in terms of weight $W$, through which we get general **Linear models**, of the form

$$
y(x, w) = \sum_{j=0}^{M-1}w_j \phi_j(x) = w^T\phi_j(x)
$$

where $\phi_j(x)$ are called *basis functions* (as before, we can define $\phi_0(x)=1$). One particular example of linear models is *polynomial regression*, in which we have $\phi_j(x)=x^j$ as the basis function.

There are still other basis functions including

$$
\phi_j(x) = \exp\{-\frac{(x-\mu_j)^2}{2s^2}\}
$$

where $\mu_j$ controls the locations of the basis functions in input space and $s$ controls the spatial space. They are referred to as 'Gaussian basis functions' because of the similarity with Gaussian function except that they have no normalization coefficients, which are not important because each of them has an adjustable parameter $w_j$.

Another example is the *sigmoid basis function* of the form

$$
\phi_j(x) = \sigma(\frac{x-\mu_j}{s})
$$

where sigmoid function is defined by

$$
\sigma(a) = \frac{1}{1+\exp(a)}.
$$


---
A regularization technique is adding a regularization term to a cost function in order to control overfitting. A particular choice is known as **Weight decay** of the form (suppose the original cost function is least squares)

$$
\frac{1}{2}\sum_{n-1}^N \{t_n-w^T\phi(x_n)\}^2 + \frac{\lambda}{2}w^Tw.
$$

It is so named because it encourages weights to decay towards zero unless supported by data.