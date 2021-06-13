---
layout: post
title: "Note 1 of PRML: Intro"
categories: [Deep Learning, Machine Learning]
---

- Supervised learning: classification, regression.
- Unsupervised learning: clustering, density estimation, visualization.
- Reinforcement learning: find suitable actions to take by a process of trial and error in a given situation in order to maximize a reward; a trade-off between *exploration*, in which the system ties out new actions to see their effectiveness, and *exploitation*, in which the system makes use of actions known to yield a high reward.

## 1.1. Polynomial Curve Fitting

*Linear models*: functions that are linear in the unknown parameters.

The more flexible polynomials with larger values of power order, corresponding more complex models, are becoming increasingly tuned to the random noise on the target values, which reduce its generalization ability and cause **overfitting**.

For a given complex model, **overfitting** will become less severe with the increase size of dataset, since the larger the dataset, the more complex/flexible the model that we can afford to fit the data. We will even suffer **underfitting** when the size of dataset is large enough because the model is no longer complex for current dataset.
> section 3.4 introduces how to avoid overfitting by adopting *Bayesian model*.

**Regularization** is used to control overfitting by adding a penalty term to the loss function in order to discourage the coefficients/weights from reaching large values. The importance of regularization is controlled by hyper-parameter $\lambda$. The larger $\lambda$ the more penalty we add to the model, the more overfitting we reduced, which may even cause underfitting though.

## 1.2. Probability Theory

*Three types of probabilities*: joint probability, marginal probability, conditional probability.

**Two rules of probabilities**:

1. Sum rule: $P(X) = \sum_Y P(X, Y)$.
2. Product rule: $P(X, Y) = P(Y\mid X)P(X)$.

**Bayes' theorem**: $P(Y\mid X) = \frac{P(X\mid Y)P(Y)}{P(X)}$, where $P(X)$ can also be written as $\sum_Y P(X\mid Y)P(Y)$.
1. Prior probability: $P(Y)$, which is given before the observation happens.
2. Posterior probability: $P(Y\mid X)$, obtained after X is observed.

**Expectation** of $f(x)$ is the average value of $f(x)$ under a probability distribution $p(x)$ as defined below:

$$\begin{align}
E[f] &= \sum_x p(x)f(x) \\
E[f] &= \ p(x)f(x)dx    \\
E[f] &\approx \frac1N \sum_{n=1}^N f(x_n) \\
E_x[f\mid y] &= \sum_x p(x\mid y)f(x).
\end{align}$$

**Variance** of $f(x)$ (or x itself) is the variability of $f(x)$ or x around its mean value $E[f]$:

$$\begin{align}
var[f] &= E[(f(x)-E[f(x)])^2] \\
​       &= E[f(x)^2] - E[f(x)]^2. \\
var[x] &= E[x^2] - E[x]^2.
\end{align}$$

**Covariance** of x and y measures the extent of x and y vary together:

$$\begin{align}
cov[x, y] &= E_{x,y}[(x-E[x])(y-E[y])] \\
​          &= E_{x,y}[xy] - E[x]E[y]
\end{align}$$

The convariance of two vector $\textit{x}$ and $\textit{y}$ is a matrix computed by:

$$\begin{align}
cov[x, y] &= E_{x,y}[(x-E[x])(y^T-E[y^T])] \\
​          &= E_{x,y}[xy^T]-E[x]E[y^T].
\end{align}​$$

**Gaussian distribution** is defined as follows by two parameters, mean $\mu$ and variance $\sigma^2$:
$$
G(x\mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma^2}\exp^{\frac{(x-\mu)^2}{2\sigma^2}}.
$$
Why parameters called mean and variance?

$$\begin{align}
E[x] &= \int{G(x\mid \mu, \sigma^2) x}dx = \mu. \\
E[x^2] &= \int{G(x\mid \mu, \sigma^2) x^2}dx = \mu^2 + \sigma^2. \\
var[x] &= E[x^2] - E[x]^2 = \sigma^2.
\end{align}$$

Gaussian distribution over D-dimensional vector $\textbf{x}$:
$$
G(\textbf{x}\mid \mu,\Sigma) = \frac{1}{(2\pi)^{D/2}\mid \Sigma\mid ^{1/2}}\exp^{-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)}.
$$
where mean $\mu$ is a D-dim vector, covariance $\Sigma$ is a D*D matrix.

*Independent and identically distribution* refers to data that are independently drawn from the same distribution.

**Maximum likelihood (MLE)**: estimate parameter $w$ to the value that maximizes the likelihood function $p(D\mid w)$, (the prob that the observed data appears).

*Drawback of MLE*: underestimate the variance of data. Suppose given n values of variable *x* sampled from a Gaussian distribution, $x=(x_1,...,x_n)$, our goal is find $\mu, \sigma^2$ satisfying:

$$\begin{align}
(\mu,\sigma^2) &= argmin_{\mu,\sigma^2}\sum_n \ln p(x_i\mid \mu,\sigma^2),\\
\mu_{ML} &= \frac1N\sum_{n=1}{N}x_n, \\
\sigma^2_{ML} &= \frac1N\sum_{n=1}{N}(x_n-\mu_{ML})^2,\\
\end{align}$$

Whereas

$$\begin{align}
E[\mu_{ML}] &= \mu, \\
E[\sigma^2_{ML}] &= (\frac{N-1}{N}) \sigma^2.
\end{align}$$

This phenomenon is called *bias* of MLE in which the true variance of observed data is **underestimated**.

**Maximum posterior (MAP)**: determine parameter $w$ by finding the most probable value of $w$ given data, in other words, by maximizing the *posterior distribution*.

## 1.3. Model Selection

**S-fold Cross-validation**: partition data into *S* groups, each time  *S-1* groups were used for training and the remaining one was used for evaluation. Repeat *S* times and then average performance scores. When *S* equals N, size of data, called *leave-one-out*.

## 1.5. Decision Theory

Getting an output given an input x and a trained model can be seen as two steps: **inference step**, in which we determine the joint distribution $p(x, t)$ by using training data to learn a model for $p(C_k\mid x)$; **decision step**, in which we make optimal decisions given the posterior probabilities.

### Three types of decision making approaches**
> In a decreasing order of complexity

#### Generative models

In inference step, determine the class-conditional densities $p(x\mid C_k)$ for each class $C_k$, then infer the prior class probabilities $p(C_k)$. Lastly use Bayes's theorem to find posterior class probabilities $p(C_k\mid x)$:

$$\begin{align}
p(C_k\mid x) &= \frac{p(x\mid C_k)p(C_k)}{P(x)}, \\
p(x) &= \sum_k p(x\mid C_k)p(C_k).
\end{align}$$

In decision step, we use decision theory to determine class membership. For instance, choose the class with highest probability or set a error matrix.

It's called so if approaches model the distribution of inputs and outputs, because by sampling from them it's possible to generate synthetic data in input space.

#### Discriminative models

In inference step, determine the posterior class probabilities $p(C_k\mid x)$ **directly** (e.g. MLE), and then assign new x to a class using decision theory.

#### No-probability models

Find a discriminant function *f(x)* which maps input x directly to a class label, in which probabilities play no rule.

## 1.6. Information Theory

We can see the information we received, given a value of a variable x, as the '**degree of surprise**' because, we will receive more information if we were told that an improbable event happened than if we were told that a very likely event happened (since we have known it will happen!).

So the amount of information, $h(x)$, is related to $p(x)$. For two independent event x and y, we should have:

$$\begin{align}
h(x,y) &= h(x)+h(y),\\
p(x,y) &= p(x)p(y).
\end{align}$$

So obviously $h(x)$ must be given by the **logarithm** of $p(x)$, so:
$$
h(x) = -\log_2p(x).
$$
And more importantly, the amount of information of x should be given by the expectation of $h(x)$ and $p(x)$:
$$
H[x] = -\sum_x p(x)\log_2p(x).
$$
Where $H[x]​$ is called **entropy** of x.