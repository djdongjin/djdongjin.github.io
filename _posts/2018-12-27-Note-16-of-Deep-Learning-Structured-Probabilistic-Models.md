---
layout: post
title: "Note 16 of Deep Learning: Structured Probabilistic Models for Deep Learning"
categories: [Deep Learning]
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

A *structured probabilistic model* is a way of describing a probability distribution, using a graph to describe which random variables in the probability distribution interact with each other directly. The **graph** here is in the graph theory sense -- a set of vertices connected to one another by a set of edges. Thus, structured probabilistic models also refers to *probabilistic graphical models (PGM)*.

> One of the major difficulties in graphical modelling is understanding which variables need to be able to interact directly, for example, which *graph structures* are most suitable for a given problem.

Structured probabilistic models only pay attention to *direct interactions* between random variables, which allows the models to have significantly fewer parameters, thus need less data and reduce computational cost on storing the model, performing inference, etc.

To illustrate this, suppose the finishing times of a three people team, a, b, c, in a relay race. A is the first person, so his finishing time doesn't depends on others. B is the second person, so his finishing time depends on A. C is the third person, so his finishing time depends on both A and B. But C's finishing time only *indirectly* depends on A via B's. If already knowing B's finishing time, we don't need to find out A's to estimate C's finishing time. All these mean that we only need to model the race using *two* interactions and omit the third, indirect interaction between A and C.

## Using Graphs to Describe Model Structure
> Using graphs to represent interactions between random variables. Only direct interactions need to be explicitly modelled.

PGMs can be roughly divided into two categories: models based on *directed acyclic graphs*, and based on *undirected graphs*.

### Directed Models
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

### Undirected Models
> The *undirected graphical model* is also known as the *Markov random field (MRF)* or *Markov networks*.

Unlike directed models, the edges in an undirected model has no arrow, thus is not associated with a conditional probability distribution.

Formally, an undirected graphical model is a structured probabilistic model defined on an undirected graph $\mathcal{G}$. For each clique $\mathcal{C}$, a factor $\phi (\mathcal{C})$, also called *clique potential*, measures the affinity of the variables in the clique for being in each of their possible joint states. The factors are constrained to be non-negative. They define an *unnormalized probability distribution*

$$
\tilde{p}(x) = \prod\_{\mathcal{C} \in \mathcal{G}} \phi (\mathcal{C}).
$$ 

> A clique of a graph is a subset of nodes that are all connected to each other by an edge.

### The Partition Function
In undirected models, we need to use the below normalized probability distribution to obtain a valid probability distribution:

$$
p(x) = \frac{1}{Z} \tilde{p}(x)
$$

where Z equals to $Z = \int \tilde{p}(x) dx$. Note that if the $\phi (\mathcal{C})$ functions have parameters, Z is also a function of these parameters. The normalizing constant **Z** is known as the *partition function*.

> We generally need to resort to approximate Z, since Z is an integral or sum over all possible joint assignments of the state x and thus often intractable to compute.

### Energy-based Models (EBM)
One way to enforce $\any x, \mathcal{p}(x) \> 0$ is to use an *energy-based model (EBM)*:

$$
\tilde{p}(x) = \exp{-E(x))
$$

where $E(x)$ is known as the *energy function*.

If we learned clique potentials directly, we must use constrained optimization to arbitrarily impose some specific minimal probability value (at lease, non-negative). By learning energy function, we can now use unconstrained optimizations.

Any distribution in forms of energy functions is an example of a *Boltzmann distribution*. Thus, many energy-based models are called *Boltzmann machines*.

## Sampling from Graphical Models

**Ancestral sampling**: used for sampling in directed acyclic models.

**Gibbs sampling**: used for sampling in undirected models.


---RBM
