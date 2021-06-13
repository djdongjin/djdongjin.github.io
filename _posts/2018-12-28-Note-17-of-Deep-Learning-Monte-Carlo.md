---
layout: post
title: "Note 17 of Deep Learning: Monte Carlo Methods"
categories: [Deep Learning]
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

Las Vegas algorithms and Monte Carlo algorithms are two rough categories of randomized algorithms. Las Vegas algorithms always return a precisely correct answer (or report fail), by consuming a random amount of resources; Monte Carlo algorithms return answers (by approximation) with a random amount of error, which may be reduced by expending more resources.

## Monte Carlo Sampling
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

## Importance Sampling
It's important to decide which part of the integrand should play the role of the probability $p(x)$ and which part should play the role of the quantity $f(x)$ whose expected value is to be estimated. But any decomposition can be rewritten as 

$$
p(x)f(x) = q(x)\frac{p(x)f(x)}{q(x)}
$$

where we now sample from $q$ and average $\frac{pf}{q}$.

In many cases, the problem to be solved will specify a given $p$ and $f$, which may not be the *optimal* choice in terms of the number of samples required to obtain a given level of accuracy. We can suppose $q^*$ is the optimal choice which can be derived easily. The optimal $q^*$* corresponds to **optimal importance sampling**.

## Markov Chain Monte Carlo Methods (MCMC)

