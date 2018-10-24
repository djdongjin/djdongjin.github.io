---
layout: post
title: "Note 2 of PRML: Probability Distributions"
categories: [Deep Learning, Machine Learning, PRML, Probability]
---

- **Density estimation**: model the probability distribution $p(x)$ given a finite observation set $x_1,...,x_N$.
- **Parametric density estimation**: determine the suitable values of parameters given an observed dataset, such as $(\mu,\sigma^2)$ in Gaussian distribution.
- **Nonparametric density estimation**: the form of distribution depends on the *size* of dataset; parameters in such distributions only control the model complexity (e.g. histograms, nearest-neighbours).

## 2.1. Binary Variables

**Bernoulli distribution**: given $p(x=1\|\mu)=\mu,0 \leq \mu \leq \mu$, we have:

$$\begin{aligned}
Bern(x|\mu) &= \mu^x (1-\mu)^{1-x} \\
E[x] &= \mu \\
var[x] &= \mu (1-\mu)
\end{aligned}​$$

Given a dataset $D={x_1,...x_n}$, the likelihood function is:

$$\begin{aligned}
p(D|\mu) = \prod_{n=1}^{N}p(x_n|\mu)=\prod_{n=1}^{N}\mu^{x_n}(1-\mu)^{1-x_n}.
\end{aligned}$$

By using a Maximum (logarithm) Likelihood Estimation (MLE), from the perspective of **frequentist**, we have

$$\begin{aligned}
\mu_{ML} &= \arg\max_{\mu} \ln p(D|\mu) \\
&= \arg\max_{\mu} \sum_{n=1}^{N}(x_n\ln \mu + (1-x_n)\ln (1-\mu)).\\
\end{aligned}$$

By letting the partial derivative equals to 0, and assume the number of observations of $x=1$ within the dataset equals to $m$, we have:
$$\begin{aligned}
\mu_{ML} = \frac1N \sum_1^N x_n = \frac{m}{N}.
\end{aligned}$$

**Binomial distribution**: N times of repeated and independent Bernoulli:

$$\begin{aligned}
Bin(m|N,\mu) &= C_m^N \mu^m (1-\mu)^{N-m}, \\
E[m] &= \sum_{m=0}^N mBin(m|N, \mu) = N\mu, \\
var[m] &= \sum_{m=0}^N (m-E[m])^2Bin(m|N,\mu) = N\mu (1-\mu).
\end{aligned}$$

### 2.1.1. Beta distribution

**Problem of MLE for Bernoulli&Binomial**: if all observations of $x_i$ equal to 1, $\mu$ will equal to 1 as well by MLE, which obviously causes the overfitting.

**Solution**: introduce a *prior distribution* $p(\mu)$ to do density estimation from the perspective of **Bayesian** treatment.

**Conjugacy**: choose a *prior* to be proportional to the likelihood so that the **posterior distribution**, which is proportional to the product of prior of likelihood, will have the same functional form as the prior.

**Beta distribution**: hyperparameters a and b are used to control the distribution of $\mu$.

$$\begin{aligned}
Beta(\mu|a,b) &= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1}, \\
E[\mu] &= \frac{a}{a+b}, \\
var[\mu] &= \frac{ab}{(a+b)^2(a+b+1)}.
\end{aligned}$$

Combining beta distribution, as prior, with Binomial distribution, as a likelihood, we have the posterior distribution of $\mu$:

$$\begin{aligned}
p(\mu|m,l,a,b) &\propto \mu^{m+a-1}(1-\mu)^{l+b-1} \\
&= \frac{\Gamma(m+a+l+b)}{\Gamma(m+a)\Gamma(l+b)}\mu^{m+a-1}(1-\mu)^{l+b-1}.
\end{aligned}$$

Where $l=N-m$ corresponding the number of $x=0$.

The result of posterior shows that it is actually an updated Beta distribution given some new data, so that it can further act as the prior if there are some subsequently observe additional data, which gives us a **sequential approach** to learning from the Bayesian viewpoint.

**Prediction**: given the dataset by now, predict the next trial using the posterior,

$$\begin{aligned}
p(x=1|D) &= \int_0^1 p(x=1|\mu)p(\mu|D)d\mu = \int_0^1 \mu p(\mu|D) d\mu = E[\mu|D] \\
&= \frac{m+a}{m+a+l+b}.
\end{aligned}$$
> Remember the expectation of beta distribution.

### 2.2. Multinomial Variables

X can take one of K possible exclusive states instead of only two. Usually we represent x using one-hot encoder, a K-dimensional vector in which $x_k=1$ and all remaining equals to 0. Suppose we use $\mu_k$ to represent the probability of $x_k$, we have

$$\begin{aligned}
p(x|\mu) = \prod_{k=1}^{K} \mu_k^{x_k}.
\end{aligned}$$

Where $\sum_k \mu_k = 1$. This distribution can be regarded as a generalization of Bernoulli distribution to more than 2 outcomes.

Now given a dataset $D={x_1,...x_N}$, the likelihood function is:

$$\begin{aligned}
p(D|\mu) &= \prod_n \prod_k \mu_k^{x_{nk}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k}, \\
m_k &= \sum_n x_{nk}.
\end{aligned}$$

Considering the constraint that $\sum_k \mu_k=1$, we cannot directly let partial derivative equal to 0. Instead, we use a Lagrange multiplier $\lambda$:

$$\begin{aligned}
\mu_k &= \arg\max_\mu \sum_k m_k \ln \mu_k + \lambda (\sum_k \mu_k -1) \\
&= -m_k / \lambda.
\end{aligned}$$

Then we feed the result to the constraint $\sum_k \mu_k=1$ and get $\lambda = -N$. Thus the MLE result is:

$$\begin{aligned}
\mu_k = m_k / N.
\end{aligned}$$

**Multinomial distribution**: The joint distribution of the quantities $m_1,...,m_K$ given $N$ observations:

$$\begin{aligned}
Mult(m_1,..,m_K|\mu,N) &= \frac{N!}{m_1!...m_K!} \prod_k \mu_k^{m_k},\\
\sum_k \mu_k &= N.
\end{aligned}$$

### 2.1.1. Dirichlet distribution

> Prior distributions for Multinomial distribution.

The conjugate prior can be given by

$$\begin{aligned}
p(\mu|\alpha) &\propto \prod_k \mu_k^{\alpha_{k-1}}, \\
Dir(\mu|\alpha) &= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)...\Gamma(\alpha_K)} \prod_k \mu_k^{\alpha_{k-1}},\\
\alpha_0 &= \sum_k \alpha_k.
\end{aligned}$$

Again, we will get a sequential learning with multinomial distribution, which can be seen from the form of posterior below:

$$\begin{aligned}
p(\mu|D,\alpha) &\propto p(\mu|\alpha)p(D|\mu) \propto \prod_k \mu_k^{\alpha_k+m_k-1} \\
&= Dir(\mu|\alpha+m) \\
&= \frac{\Gamma(\alpha_0+m)}{\Gamma(\alpha_1+m_1)...\Gamma(\alpha_K+m_K)} \prod_k \mu_k^{\alpha_k+m_k-1}.
\end{aligned}$$