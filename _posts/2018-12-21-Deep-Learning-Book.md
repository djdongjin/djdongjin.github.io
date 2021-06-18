---
title: "Deep Learning -- Ian GoodFellow"
layout: post
date: 2018-12-21 20:10:47 +07:00
modified: 2021-06-18 20:10:47 +07:00
tags: [Machine Learning, Deep Learning, Book]
comments: true
---

* TOC
{:toc}

## 11. Practical Methodology

Practical design process:

1. Determine goals: what error metric to use and corresponding target value, both of which should be driven by the problem that the application is intended to solve.
2. Establish a working end-to-end pipeline and the estimation of the appropriate performance metrics as soon as possible.
3. Instrument the system well to determine bottlenecks, diagnose which parts are performing worse than expected and whether it is due to overfitting, underfitting or a defect in the data or implementation.
4. Based on specific findings from instruments, repeatedly make incremental changes, including gathering new data, tuning hyperparameters, changing algorithms.

### Performance Metrics

> Need to decide both an expected target value, according to previous benchmark results or error rate that is necessary for a system to be safe, and which metrics to use such as accuracy, precision, recall, etc.

**Precision** is the fraction of detections reported by the model that were correct; **Recall** is the fraction of true events that were detected. When using them, it’s common to plot a *PR curve* with precision on the y-axis and recall on the x-axis.

We can also convert precision $p$ and recall $r$ into an $F-score$ given by

$$ F = \frac{2pr}{p+r} $$

Another metric is **Coverage** which is the fraction of examples for which the machine learning system is able to produce a response. It is useful when the system is allowed to refuse to make a decision and deliver to human to make a decision.

Many metrics are available but what is more important is to *determine which performance metric to improve ahead of time and then concentrate on that.*

### Default Baseline Models

> The goal is to establish a reasonable end-to-end system as soon as possible, which can be used as a baseline.

A reasonable choice of optimization method is **SGD** with momentum with a decaying learning rate. Popular decay schemes include 1) decaying linearly until reaching a fixed minimum learning rate; 2) decaying exponentially; 3) decreasing learning rate by a factor of 2-10 each time validation error plateaus. Another reasonable alternative is **Adam**. 

**Batch normalization** may have a dramatic effect on optimization performance, especially for convolutional networks and networks with sigmoidal nonlinearities such as *sigmod* or *tanh*. Batch normalization should be introduced if optimization is likely problematic.

Some *mild forms of regularization* should be included from the start unless the training set contains tens of millions of examples. **Early stopping** should be used universally. **Dropout** is a regularizer compatible with many models.

### Determine whether to Gather More Data

> After implementing the baseline, it’s often better to gather more data than to improve learning algorithm or try out different algorithms.

If current performance on the training set is poor, there is not reason to gather more data since the learning algorithm even doesn’t utilize the training set available so far (, which means *underfitting*). So try improve the size of the model by adding more layers or hidden units in each layer, or by tuning hyperparameters such as learning rate.

If large or fine-tuned models still do not work well, the problem may appear to be the *quality of the training data*, such as too noisy or no  useful features. So turn to collect cleaner data or richer set of features.

If performance on training set is acceptable, then measure the performance on test set. If performance on test set is still acceptable, there is nothing left to be done (or you can try improve the learning algorithm). If performance on test set is poor (, which means *overfitting*), then gathering more data is one of the most effective solutions. An alternative of gathering more data, or say, reducing overfitting, is to reduce the size of the model or improve regularization by tuning hyperparameters such as weight decay coefficients or by adding regularization strategies such as dropout, L2 regularization.

When deciding to gather more data, it’s also necessary to decide *how much data to gather*. It’s useful to plot curves showing the relationship between *training set size* (may on a logarithmic scale) and *generalization error*.

### Selecting Hyperparameters

Learning rate may be the most important hyperparameter, so if you have no enough time, tune learning rate first.

### Debugging Strategies

1. *Visualize the model in action*: when training a object detection model, view some images with predicted result superimposed on it; when training a generative model of speech, listen to some of produced speech samples, etc. Directly observing the machine learning model performing its tasks can help you estimate if the *quantitative* performance result it achieves seem reasonable.
2. *Visualize the worst mistakes*: by viewing the training set examples that are the hardest to model correctly, one can often discover problems with the way the data has been preprocessed or labeled.
3. *Reasoning about software using train and test error*: if training error is low but test error is high, the software implementation should work properly, and the model overfits (or there is an error when saving the model and then reloading for evaluation).
4. Compare back-propagated derivatives to numerical derivatives.
5. Monitor histograms of activations and gradient.

## 13. Linear Factor Models

A linear factor model is defined by using a stochastic, linear decoder function that generates $x$ by adding noise to a linear transformation of $h$ which is a latent variable representing the data.

The data generation process is as follows. First, the explanatory factors $h$ are sampled from

$$
h \sim p(h)
$$

where $p(h) = \prod\_i p(h\_i)$ is a factorial distribution. Then the real-valued observable variables is generated given the factors by:

$$
x = Wh + b + noise
$$

where the noise is Gaussian and diagonal.

### Probabilistic PCA and Factor Analysis

> Probabilistic PCA, factor analysis and other linear factor models only differ in the choices made for the model’s *prior* over latent variables $h$ before observing $x$ and noise distributions.

In *factor analysis*, the latent variable prior is a *unit variance Gaussian*

$$
h \sim \mathcal{N}(h;0,I)
$$

The noise is drawn from a diagonal covariance Gaussian distribution, with covariance matrix $\psi = diag(\sigma^2)$, with $\sigma^2 = [\sigma\_1^2,...,\sigma\_n^2]^T$.

The latent variables thus can *capture the dependencies* between the different observed variables $x$. And $x$ is just a multivariate normal random variable with

$$
x \sim \mathcal{N}(x;b,WW^T+\psi).
$$

To get probabilistic PCA model, we modify the factor analysis model, making the conditional variances $\sigma\_i^2$ equal to each other. So the covariance of $x$ is $WW^T+\sigma^2I$ where $\sigma^2$ is now  a scalar. This yields

$$\begin{aligned}
x &\sim \mathcal(N)(x;b,WW^T+\sigma^2I) \\\\
  &= Wh + b + \sigma z
\end{aligned}$$

where $z \sim \mathcal{N}(z;0,I)$ is a Gaussian noise.

### Independent Component Analysis (ICA)

> To modeling linear factors that seeks to separate an observed signal into many underlying signals that are scaled and added together to form the observed data.

### Slow Feature Analysis (SFA)

> A linear factor model that uses information from time signals to learn invariant features.
> Slowness principle: the important characteristics of scene change vary slowly compared to the individual measurements that make up the scene.

## 14. Autoencoders

Autoencoder is a neural network trained to copy its input to its output. It can be seen as two parts: an encoder $h = f(x)$ that generates a latent representation, a decoder $r = g(h)$ that generates the reconstruction of the input given its latent representation.

Ideally, autoencoders are designed not to recover all data, but to copy only approximately and to copy only input that *resembles the training data*. Since the model tries to prioritize which aspects of the input should be copied, it usually learns properties of the data. Autoencoders can be used for dimensionality reduction and feature learning.

### Undercomplete Autoencoders

> Autoencoders whose code dimension $h$ is less than the input dimension.

Training a neural network to copy the input itself and constraining the latent representation $h$ to have smaller dimension than $x$ can force the autoencoder to capture the most salient features of the training data.

When the decoder is linear and the loss function is mean squared error, an under complete autoencoder learns to span the same subspace as *PCA*.

Autoencoders with nonlinear encoder function $f$ and $g$ can learn a more powerful nonlinear generalization of PCA. But if it is allowed too much capacity, the autoencoder may not extract useful information about the distribution of the data.

A similar problem also occurs if the hidden code has dimension equal to the input and in the *overcomplete* case.

### Regularized Autoencoders

> Use loss functions that encourage the model to have other properties besides the ability of performing copy task, such as sparsity of the representation, smallness of the derivative of the representation, and robustness to noise or to missing inputs.

A regularized autoencoder can be nonlinear and overcomplete but can still learn useful hidden code about the data distribution.

Nearly any generative model with latent variables and an inference procedure (for computing latent representations given input) may be viewed as a particular form of autoencoder.

#### Sparse Autoencoders

An autoencoder whose training criterion involves a sparsity penalty $\Omega (h)$ on the coder layer $h$ as a regularization term. So its loss function is:

$$
L(x, g(f(x))) + \Omega (h)
$$

#### Denoising Autoencoders

> Instead of adding a penalty $\Omega$ to the loss function, DAE directly changes the reconstruction error of the loss function.

A DAE minimizes

$$
L(x, g(f(\tilde{x})))
$$

where $\tilde{x}$ is a copy of x that is corrupted by some form of noise. So a DAE need to undo this corruption instead of copying the input.

> Receive a corrupted data as input and trained to predict the original, uncorrupted data as output.

The DAE training procedure introduces a corruption process $C(\tilde{x} \mid x)$ which represents a conditional distribution over corrupted samples $\tilde{x}$ given original data $x$.

DAE learns a *reconstruction distribution* $p_{reconstruct}(x \mid \tilde{x})$ as follows:

1. Sample a training example $x$ from the training data;
2. Sample a corrupted version $\tilde{x}$ from $C(\tilde{x} \mid x)$;
3. Use $(x, \tilde{x})$ as a training example for estimating the DAE reconstruction distribution $p_{reconstruct}(x \mid \tilde{x})=p_{decoder}(x \mid h)$ with h the output of encoder $f(\tilde{x})$.

### Learning Manifolds with Autoencoders

An important characterization of a manifold is *the set of its tangent planes*. At a point x on a d-dimensional manifold, the tangent plane is given by d basis vectors that span the local directions of variation allowed on the manifold.

### Contractive Autoencoders (CAE)

CAE introduces an explicit regularizer on the hidden code $h=f(x)$ to encourage the derivatives of $f$ to be as small as possible:

$$
\Omega(h) = \lambda \mid\mid \frac{\alpha f(x)}{\alpha x} \mid\mid _F^2.
$$

CAE is trained to map a neighborhood of input points to a smaller neighborhood of output points, by add a derivative penalty into the cost function.

## 16. Structured Probabilistic Models for Deep Learning

A *structured probabilistic model* is a way of describing a probability distribution, using a graph to describe which random variables in the probability distribution interact with each other directly. The **graph** here is in the graph theory sense -- a set of vertices connected to one another by a set of edges. Thus, structured probabilistic models also refers to *probabilistic graphical models (PGM)*.

> One of the major difficulties in graphical modelling is understanding which variables need to be able to interact directly, for example, which *graph structures* are most suitable for a given problem.

Structured probabilistic models only pay attention to *direct interactions* between random variables, which allows the models to have significantly fewer parameters, thus need less data and reduce computational cost on storing the model, performing inference, etc.

To illustrate this, suppose the finishing times of a three people team, a, b, c, in a relay race. A is the first person, so his finishing time doesn't depends on others. B is the second person, so his finishing time depends on A. C is the third person, so his finishing time depends on both A and B. But C's finishing time only *indirectly* depends on A via B's. If already knowing B's finishing time, we don't need to find out A's to estimate C's finishing time. All these mean that we only need to model the race using *two* interactions and omit the third, indirect interaction between A and C.

### Using Graphs to Describe Model Structure

> Using graphs to represent interactions between random variables. Only direct interactions need to be explicitly modelled.

PGMs can be roughly divided into two categories: models based on *directed acyclic graphs*, and based on *undirected graphs*.

#### Directed Models

> The *directed graphical model* is also known as the *belief network* or *Bayesian network*.

Drawing an arrow from $a$ to $b$ in directed models means that we define the probability distribution over $b$ via $a$ conditional distribution.

Formally, a directed graphical model defined on variables $x$ is defined by a directed acyclic graph $\mathcal{G}$, and a set of *local conditional probability distributions* $p(x\_i \mid P_{a_\mathcal{G}}(x\_i))$ where $P_{a_{\mathcal{G}}}(x\_i)$ is the parents of $x\_i$. The probability distribution over $x$ is given by

$$
p(x) = \prod\_i p(x\_i \mid p_{a_{\mathcal{G}}}(x\_i))
$$

In the relay race example, we have

$$
p(a, b, c) = p(a)p(b \mid a)p(c \mid b)
$$

Still, there are some kinds of information that cannot be encoded in the graph. Suppose C will finish his running in a fixed time $t\_c$ no matter when B finish his running. In such a situation, we can model $p(c \mid b)$ with $O(k)$ parameters instead of $O(k^2)$. But the assumption that C's running time is independent to all other factors cannot be encoded in a graph over $t\_0, t\_1, t\_2$ which correspond to their finishing time. Instead, we encode this information in the definition of the condition distribution itself, (here is $p(c \mid b)$, by limiting its available value).

In a nutshell, a directed graphical model syntax doesn't have any constraint on how the conditional distributions are defined. It only defines which variables are allowed to take in as arguments. (In a simple way, it only define whether an edge exists, but not define how the edge is constrained).

#### Undirected Models

> The *undirected graphical model* is also known as the *Markov random field (MRF)* or *Markov networks*.

Unlike directed models, the edges in an undirected model has no arrow, thus is not associated with a conditional probability distribution.

Formally, an undirected graphical model is a structured probabilistic model defined on an undirected graph $\mathcal{G}$. For each clique $\mathcal{C}$, a factor $\phi (\mathcal{C})$, also called *clique potential*, measures the affinity of the variables in the clique for being in each of their possible joint states. The factors are constrained to be non-negative. They define an *unnormalized probability distribution*

$$
\tilde{p}(x) = \prod\_{\mathcal{C} \in \mathcal{G}} \phi (\mathcal{C}).
$$

> A clique of a graph is a subset of nodes that are all connected to each other by an edge.

#### The Partition Function

In undirected models, we need to use the below normalized probability distribution to obtain a valid probability distribution:

$$
p(x) = \frac{1}{Z} \tilde{p}(x)
$$

where Z equals to $Z = \int \tilde{p}(x) dx$. Note that if the $\phi (\mathcal{C})$ functions have parameters, Z is also a function of these parameters. The normalizing constant **Z** is known as the *partition function*.

> We generally need to resort to approximate Z, since Z is an integral or sum over all possible joint assignments of the state x and thus often intractable to compute.

#### Energy-based Models (EBM)

One way to enforce $\any x, \mathcal{p}(x) \> 0$ is to use an *energy-based model (EBM)*:

$$
\tilde{p}(x) = \exp{-E(x))
$$

where $E(x)$ is known as the *energy function*.

If we learned clique potentials directly, we must use constrained optimization to arbitrarily impose some specific minimal probability value (at lease, non-negative). By learning energy function, we can now use unconstrained optimizations.

Any distribution in forms of energy functions is an example of a *Boltzmann distribution*. Thus, many energy-based models are called *Boltzmann machines*.

### Sampling from Graphical Models

**Ancestral sampling**: used for sampling in directed acyclic models.

**Gibbs sampling**: used for sampling in undirected models.

---RBM

## 17. Monte Carlo Methods

Las Vegas algorithms and Monte Carlo algorithms are two rough categories of randomized algorithms. Las Vegas algorithms always return a precisely correct answer (or report fail), by consuming a random amount of resources; Monte Carlo algorithms return answers (by approximation) with a random amount of error, which may be reduced by expending more resources.

### Monte Carlo Sampling

When a sum or integral cannot be computed exactly (e.g. it has an exponential numbers of terms), we can approximate it using *Monte Carlo sampling*, which views the sum or integral as an expectation under some distribution, and approximate the *expectation by a corresponding average*.

Let

$$\begin{aligned}
s &= \sum_x p(x)f(x) = E_p[f(x)] \\
(or) &= \int p(x)f(x) = E_p[f(x)]
\end{aligned}$$

*s* can be approximated by drawing *n* samples $x^{(1)},...,x^{(n)}$ from *p* and computing the empirical average

$$
\hat{s_n} = \frac{1}{n} \sum_{i=1}^n f(x^{(i)})
$$

We also have

$$\begin{aligned}
Var[\hat{s}_n] &= \frac{1}{n^2} \sum_{i=1}^n Var[f(x)] \\
&= \frac{Var[f(x)]}{n}.
\end{aligned}$$

which gives us a way of estimating $Var[\hat{s}_n]$.

When x cannot be sampled from *p*, an alternative is to use *importance sampling*, and more generally, to form a sequence of estimators that coverage towards the distribution of interest, which is the approach of **Monte Carlo Markov chains (MCMC)**.

### Importance Sampling

It's important to decide which part of the integrand should play the role of the probability $p(x)$ and which part should play the role of the quantity $f(x)$ whose expected value is to be estimated. But any decomposition can be rewritten as 

$$
p(x)f(x) = q(x)\frac{p(x)f(x)}{q(x)}
$$

where we now sample from $q$ and average $\frac{pf}{q}$.

In many cases, the problem to be solved will specify a given $p$ and $f$, which may not be the *optimal* choice in terms of the number of samples required to obtain a given level of accuracy. We can suppose $q^*$ is the optimal choice which can be derived easily. The optimal $q^*$* corresponds to **optimal importance sampling**.

### Markov Chain Monte Carlo Methods (MCMC)
