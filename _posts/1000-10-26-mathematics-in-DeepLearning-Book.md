---
layout: post
title: "Mathematics in DeepLearning Book"
categories: [Deep Learning, Machine Learning, Math]
---

## Linear Algebra

### Eigen decomposition

**Eigenvector**: a vector satisfying

$$
Av = \lambda v.
$$

where $A$ is a square matrix, $v$ is called the *eigenvector* of $A$, $\lambda$ is called the the *eigenvalue* corresponding eigenvector $v$.

**Eigen decomposition**: suppose matrix $A$ has eigenvectors $\{v^{(1)},...,v^{(n)}\}$ with corresponding eigenvalues $\{\lambda_1,...,\lambda_n\}$, let $V=[v^{(1)},...,v^{(n)}]$, $\lambda=[\lambda_1,...,\lambda_n]^T$, the eigen decomposition of $A$ is:

$$
A = Vdiag(\lambda)V^{-1} (=Q\Lambda Q^T)
$$

### SVD: Singular Value Decomposition
> More general, for example, non-square matrix has a SVD but not have a eigen decomposition

$$
A = UDV^T.
$$

where $U$ is a $m \times m$ orthogonal matrix, $D$ is a $m \times n$ diagonal matrix, $V$ is a $n \times n$ orthogonal matrix, given $A$ is a $m \times n$ matrix.

## Probability and Information Theory

For discrete variable $x$, we use **probability mass function, PMF** to describe the probability that $x$ equals to each value. For continuous variable $x$, we use **probability density function, PDF** to describe the probability density of $x$ over its domain of definition.

## Numerical computation

Some term definitions:

1. partial derivative: $\frac{\alpha f(x)}{\alpha x_i}$
2. gradient: derivative vector that contains each partial derivatives: $\nabla_x f(x) = [\frac{\alpha f(x)}{\alpha x_1},...,\frac{\alpha f(x)}{\alpha x_n}]$.
3. Jacobian matrix: derivatives in which both input and output are vector, $f: R^m \to R^n$. $J \in R^{n \times m}, J_{i,j}=\frac{\alpha f(x_i)}{\alpha x_j}$
4. Hessian matrix: second derivatives, gradient version of Jacobian matrix: $H(f)(x)_{i,j} = \frac{\alpha^2 f(x)}{\alpha x_i \alpha x_j}$

## Machine learning basics

**When use L1 instead of L2**

Sometimes it's import to distinguish between zero values and values near zero (but not zero). The gradient of L2 near zero is too flat to distinguish those values, where as L1 keep the same gradient anywhere.

**What is Frobenius norm**

$$
\mid\mid A \mid\mid_F = \sqrt {\sum_{i,j}A_{i,j}^2}.
$$

We can control underfit or overfit of a model by adjusting the **capacity** of the model, which is the ability to fit various types of function.

One way to adjust the *capacity* of a model is choosing an appropriate **hypothesis space**, the set of functions that can be a solution of the model, for example, linear space, polynomial space, etc.

It's still hard, however, to find the best solution in such a space, which we can also call **representational space**. Instead, we usually find a function which could *significantly* reduce the error on training error. Thus, we can see that, caused by some extra limit factors such as un-perfect learning method, the **effective capacity** of the learning method may less than the *representational space* of the model set.

**VC dimension**: measure the capacity of a binary classifier.

---
* [ ] MLE vs. MAP
 