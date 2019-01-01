---
layout: post
title: " Note 14 of Deep Learning: Autoencoders "
categories: [Deep Learning]
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

Autoencoder is a neural network trained to copy its input to its output. It can be seen as two parts: an encoder $h = f(x)$ that generates a latent representation, a decoder $r = g(h)$ that generates the reconstruction of the input given its latent representation.

Ideally, autoencoders are designed not to recover all data, but to copy only approximately and to copy only input that *resembles the training data*. Since the model tries to prioritize which aspects of the input should be copied, it usually learns properties of the data. Autoencoders can be used for dimensionality reduction and feature learning.

## Undercomplete Autoencoders
> Autoencoders whose code dimension $h$ is less than the input dimension.

Training a neural network to copy the input itself and constraining the latent representation $h$ to have smaller dimension than $x$ can force the autoencoder to capture the most salient features of the training data.

When the decoder is linear and the loss function is mean squared error, an under complete autoencoder learns to span the same subspace as *PCA*.

Autoencoders with nonlinear encoder function $f$ and $g$ can learn a more powerful nonlinear generalization of PCA. But if it is allowed too much capacity, the autoencoder may not extract useful information about the distribution of the data.

A similar problem also occurs if the hidden code has dimension equal to the input and in the *overcomplete* case.

## Regularized Autoencoders
> Use loss functions that encourage the model to have other properties besides the ability of performing copy task, such as sparsity of the representation, smallness of the derivative of the representation, and robustness to noise or to missing inputs.

A regularized autoencoder can be nonlinear and overcomplete but can still learn useful hidden code about the data distribution.

Nearly any generative model with latent variables and an inference procedure (for computing latent representations given input) may be viewed as a particular form of autoencoder.

### Sparse Autoencoders
An autoencoder whose training criterion involves a sparsity penalty $\Omega (h)$ on the coder layer $h$ as a regularization term. So its loss function is:

$$
L(x, g(f(x))) + \Omega (h)
$$

### Denoising Autoencoders
> Instead of adding a penalty $\Omega$ to the loss function, DAE directly changes the reconstruction error of the loss function.

A DAE minimizes

$$
L(x, g(f(\tilde{x})))
$$

where $\tilde{x}$ is a copy of x that is corrupted by some form of noise. So a DAE need to undo this corruption instead of copying the input.

## Denoising Autoencoders
> Receive a corrupted data as input and trained to predict the original, uncorrupted data as output.

<img src="/assets/DAE.jpg" width="100%">

The DAE training procedure introduces a corruption process $C(\tilde{x} \mid x)$ which represents a conditional distribution over corrupted samples $\tilde{x}$ given original data $x$.

DAE learns a *reconstruction distribution* $p_{reconstruct}(x \mid \tilde{x})$ as follows:
1. Sample a training example $x$ from the training data;
2. Sample a corrupted version $\tilde{x}$ from $C(\tilde{x} \mid x)$;
3. Use $(x, \tilde{x})$ as a training example for estimating the DAE reconstruction distribution $p_{reconstruct}(x \mid \tilde{x})=p_{decoder}(x \mid h)$ with h the output of encoder $f(\tilde{x})$.

## Learning Manifolds with Autoencoders
An important characterization of a manifold is *the set of its tangent planes*. At a point x on a d-dimensional manifold, the tangent plane is given by d basis vectors that span the local directions of variation allowed on the manifold.

## Contractive Autoencoders (CAE)
CAE introduces an explicit regularizer on the hidden code $h=f(x)$ to encourage the derivatives of $f$ to be as small as possible:

$$
\Omega(h) = \lambda \mid\mid \frac{\alpha f(x)}{\alpha x} \mid\mid _F^2.
$$

CAE is trained to map a neighborhood of input points to a smaller neighborhood of output points, by add a derivative penalty into the cost function.
