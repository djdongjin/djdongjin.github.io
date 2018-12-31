---
layout: post
title: Note 13 of Deep Learning: Linear Factor Models
categories: [Deep Learning, Machine Learning]
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>


A linear factor model is defined by using a stochastic, linear decoder function that generates $x$ by adding noise to a linear transformation of $h$ which is a latent variable representing the data.

The data generation process is as follows. First, the explanatory factors $h$ are sampled from

$$
h \sim p(h)
$$

where $p(h) = \prod_i p(h_i)$ is a factorial distribution. Then the real-valued observable variables is generated given the factors by:

$$
x = Wh + b + noise
$$

where the noise is Gaussian and diagonal.

## Probabilistic PCA and Factor Analysis
> Probabilistic PCA, factor analysis and other linear factor models only differ in the choices made for the model’s *prior* over latent variables $h$ before observing $x$ and noise distributions.

In *factor analysis*, the latent variable prior is a *unit variance Gaussian*

$$
h \sim \mathcal{N}(h;0,I)
$$

The noise is drawn from a diagonal covariance Gaussian distribution, with covariance matrix $\psi = diag(\sigma^2)$, with $\sigma^2 = [\sigma_1^2,...,\sigma_n^2]^T$.

The latent variables thus can *capture the dependencies* between the different observed variables $x$. And $x$ is just a multivariate normal random variable with

$$
x \sim \mathcal{N}(x;b,WW^T+\psi).
$$

To get probabilistic PCA model, we modify the factor analysis model, making the conditional variances $\sigma_i^2$ equal to each other. So the covariance of $x$ is $WW^T+\sigma^2I$ where $\sigma^2$ is now  a scalar. This yields

$$\begin{aligned}
x &\sim \mathcal(N)(x;b,WW^T+\sigma^2I) \\
  &= Wh + b + \sigma z
\end{aligned}$$

where $z \sim \mathcal{N}(z;0,I)$ is a Gaussian noise.

## Independent Component Analysis (ICA)
> To modeling linear factors that seeks to separate an observed signal into many underlying signals that are scaled and added together to form the observed data.

## Slow Feature Analysis (SFA)
> A linear factor model that uses information from time signals to learn invariant features.
> Slowness principle: the important characteristics of scene change vary slowly compared to the individual measurements that make up the scene.

