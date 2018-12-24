<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

---
layout: post
title: Notes of COMP551 Applied Machine Learning
categories: [Machine Learning, Courses]
---

# Lecture 2&3: Linear Regression
**i.i.d assumption**: the examples, $x_i$, in the training set are *independently* and *identically distributed*:
- Independently: each $x_i$ is freshly sampled according to some probability distribution *D* over the data domain *X*, which means $x_i$ is not dependent to any $x_j$ in the same training set.
- Identically: the distribution *D* is the same probability distribution for all examples.

**Solving linear regression analytically**
Suppose we have $y_i = \sum_{i=0}^{m} w_ix_i$ where $x_i=1$ for each example corresponding bias parameter $w_0$. A most common choice is minimizing *mean-square error (MSE)*:

$$\begin{aligned}
L(w) &= argmin_w \sum_{i=1}^{n} (y_i - w^Tx_i)^2 \\
	 &= argmin_w ||Y - Xw||^2 \\
	 &= argmin_w (Y-Xw)^T(Y-Xw) \\
\end{aligned}$$

We can minimize the loss function by taking the derivatives with w and setting to zero:

$$\begin{aligned}
X^T(Y-Xw) &= 0 \\
X^TY &= X^TXw \\
w &= (X^TX)^{-1}X^TY
\end{aligned}$$

**Problem of analytical solution: computation extensive**
Overall we have 3 matrix multiplication and 1 matrix inversion. Suppose we have m examples, each with n features. $X^TX$ is the most expensive multiplication among the 3 and takes $nm^2$ operations. $(X^TX)^{-1}$ will take $m^3$ operations. So the overall time complexity is $\Omega{m^3}$ (or just say polynomial), which is problematic for large scale datasets, say, $m=10^6$.

**Gradient descent solution**
After getting the derivative, instead of computing $w$ directly, we update $w$ a bit according to the derivative&learning rate, and iterate until it falls into a local minima (convergence).
 
# Lecture 4: Linear classification

**Evaluation**: use cross-validation for model selection:
- Training set: select a hypothesis $f$ from the hypotheses space $F$ (parameter tuning, e.g. regression for a given degree).
- Validation set: compare best $f$ from each hypothesis class across different classes (e.g. different degree regression).
- Test set: get a true estimate of the generalization error.

**Ridge regression**: add L2 regularization as a penalty for model complexity in loss function to reduce overfitting.

$$
w = argmin_{w}(\sum_{i=1}^n (y_i - w^Tx_i)^2 + \lambda\sum_{j=0}^m w_j^2)
$$


**Lasso regression**: add L1 regularization as a penalty.

$$
w = argmin_{w}(\sum_{i=1}^n (y_i - w^Tx_i)^2 + \lambda\sum_{j=0}^m \|w_j\|)
$$

# Lecture 5: Generative models for linear classification
Two approaches for linear classification:
- Discriminative learning: directly estimate $P(y \mid x)$.
- Generative learning: separately model $P(x \mid y)$ and $P(y)$; use Bayes rule to estimate $P(y \mid x)$.

## Linear discriminant analysis (LDA)
According to Bayes rule:

$$P(y \mid x) = \frac{p(x \mid y)p(y)}{p(x)}$$

LDA makes an explicit assumption about $p(x \mid y)$

$$
p(x \mid y) = \frac{e^{-\frac {1}{2}\left( x-\mu\right) ^{T}\Sigma ^{-1}\left( x-\mu\right) }}{\left( 2\pi \right) ^{1/2}\left| \Sigma \right| ^{1/2}}
$$

which is a multivariate Gaussian with mean $\mu$ and covariance matrix $\Omega$. $x$ here is an $m*1$ vector.
> A key assumption of LDA is that both resulting classes have the **same** covariance matrix $\Omega$.
> Parameters to learn include $p(y), \mu, \Sigma$.

## Scale up generative learning: Naive Bayes
> model $P(x \mid y)$ and $P(y)$ and then compute $P(y \mid x)$.

**Naive bayes assumption**: assume $x_j$s are conditionally independent given $y$, that is, $P(x_j \mid y) = P(x_j \mid y, x_k)$ for all $j,k$.

With naive bayes assumption, we have:

$$\begin{aligned}
P(x \mid y) &= P(x_1,...,x_m \mid y) \\
			&= P(x_1 \mid y)P(x_2 \mid y, x_1)P(x_m \mid y, x_1,...,x_{m-1}) \\
			&= P(x_1 \mid y),...,P(x_m \mid y)
\end{aligned}$$

## Gaussian Naive Bayes
> Extending Naive Bayes to continuous inputs.

$P(y)$ is still a binomial distribution, but $P(x \mid y)$ is a multivariate (normal) Gaussian distribution with mean $\mu \in R^n$ and covariance matrix $\Sigma \in R^n * R^n$:
- If all classes have the same $\Sigma$, it is a *LDA*.
- If $\Sigma$ is distinct between classes, it is a *QDA*.
- If $\Sigma$ is diagonal (e.g. features are independent), it is *Gaussian Naive Bayes*.



