---
layout: post
title: “Relational Inductive Biases, Deep Learning, and Graph Networks”
categories: [Deep Learning, Graph Neural Network, GNN]
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

> This is a note for the paper: *Relational inductive biases, deep learning, and graph networks*.

## Introduction

*Combinatorial generalization*: construct new inferences, predictions, and behaviours from known building blocks.

Modern deep learning methods often follows an “end-to-end” design philosophy which emphasizes *minimal a prior representational and computational assumptions*, and seeks to avoid explicit structure and “hand-engineering”. Such an emphasis works well and has been affirmed by due to the current abundance of cheap data and computing resources, but makes trading off *sample efficiency for more flexible learning a relational choice*.

There are still key challenges such a philosophy faces in *complex language and science understanding, reasoning about structured data, transferring learning beyond the training conditions, and learning from small amounts of experience*. Most of these challenges demand combinatorial generalization, and an approach which doesn’t eschews compositionally and explicit structures.

A recent class of models has arisen at the intersection of deep learning and structured approaches, focusing on approaches for *reasoning about explicitly structured data, and graphs in particular*. The similarity among these approaches is a capacity for *performing computation over discrete entities and the relations between them*. The difference between these approaches and classical approaches is *how the representations and structure of the entities and relations (and corresponding computations) can be learned*. Concretely, these approaches carry strong **relational inductive biases** by specifying architectural assumptions, which lead to learning about entities and relations.

## Relational Inductive Biases

### Relational Reasoning

**Structure** is defined as the product of composing a set of known building blocks. **Structured representations** capture this composition, for example, the arrangement of the elements. **Structured computations** operate over the elements and their composition as a whole.

**Relational reasoning** involves manipulating structured representations of *entities*/nodes and *relations*/edges, using *rules*/functions for how they can be composed.

For example, graphical models can represent complex joint distributions by making explicit random conditional independences among random variables. These models are successful because they capture the sparse structure which underlies many real-world generative processes, and support efficient algorithms for learning and reasoning.

### Inductive Biases

In a learning process, the algorithm may find multiple solutions which are equally good. An **inductive bias** allows a learning algorithm to *prioritize* one solution over another, independent of the observed data. For example, in a Bayesian model, its inductive biases are expressed through the *choice and parameterization of the prior distribution*. An inductive bias may be just a regularization term to avoid overfitting, or may be encoded in the architecture of the algorithm itself.
—
Many machine learning methods having a capacity for relational reasoning use a *relational inductive bias*, or inductive biases for simplicity, which impose constraints on relationships and interactions among entities in a learning process. some relational inductive biases are as follows:
  
![][image-1]

## Graph Neural Networks
> Neural networks that operate on graphs, and structure their computations accordingly.

The author of the paper also introduced an open-source *graph networks (GN)* framework which defines a class of functions for relational reasoning over graph-structured representations. The lack of “neural” in GN is to indicate that they can be implemented with functions other than neural networks.

The main computation unit in the GN framework is the *GN block*, a “graph-to-graph” module which takes a *graph as input*, performs *computations* over the structure, and returns a *graph as output*.

The key design principles of the GN block include: **flexible representations, configurable within-block structure, composable multi-block architectures**.

In this GN framework, a graph is defined by a 3-tuple $G=(u, V, E)$ which includes elements such as node $v\_i$, edge $e\_k$, and global attributes $u$. An edge $k$ has a sender node $s\_k$ and receiver node $r\_k$. The $V=\lbrace v\_i \rbrace _{i=1:N^e}$ is the set of nodes where each $v_i$ is a node’s attribute. The $E= \lbrace (e_k, r_k, s_k) \rbrace _{k=1:N^e}$ is the set of edges where each $e\_k$ is the edge’s attribute.

A GN block contains three **update** functions, $\phi$, and three **aggregation** functions, $\rho$:

$$\begin{aligned}e'_k &= \phi^e(e_k, v_{rk}, v_{sk}, u), \prod \bar{e}'_i = \rho^{e \to v}(E'\_i) \\
v'_i &= \phi^v(\bar{e}'_i, v\_i, u), \prod \prod \bar{e}'=\rho^{e \to u}(E') \\\\
u' &= \phi^u(\bar{e}',\bar{v}', u), \prod \prod \bar{v}'=\rho^{v \to u}(V'). 
\end{aligned}$$

where $E'_i = \lbrace (e'_k, r\_k, s\_k) \rbrace _{r_k=i, k=1:N^e}, V' = \lbrace v'_i \rbrace _{i=1:N^v}, E' = \cup_i E'_i= \lbrace (e'_k, r_k, s_k) \rbrace _{k=1:N^e}$.

$\phi^e$ is used to map across all edges to compute per-edge updates; $\phi^v$ is used to map across all nodes to compute per-node updates; and $\phi\_u$ is used once to compute the global update. $\rho$ functions take a set as input, and reduce it to a single element which represents the aggregated information. The $\rho$ functions must be *invariant* to permutations of their inputs and should take variable numbers of arguments.

The computational steps in a GN block is as follows. Intuitively, it mainly including three stages: a) edge update; b) node update; c) global update.

![][image-2]
![][image-3]

### Relational inductive biases in graph networks

The GN framework has several strong relational biases in a learning process. 
First, graphs express *arbitrary relationships* among entities which means the GN’s input determines how representations interact and are isolated, rather than those choices determined by the fixed architecture. For example, the presence of the relationship between two nodes are expressed by an edge; the absence of an edge corresponds the assumption that the two nodes have no relationship and shouldn’t effect each other *directly*.
Second, graphs represent entities and relations as sets which are invariant to permutations and also means GNs are invariant to the order of these elements.
Third, a GN’s *per-node* and *per-edge* functions reused across all nodes and edges, which means GNs automatically support a form of combinatorial generalization, because, graphs are composed of edges, nodes, and global features, a single GN can operate on graphs of different sizes and shapes.

## Design Principles for Graph Network Architectures

### Flexible representations

The flexible representations of graph networks are expressed in terms of *the representation of the attributes*, and in terms of *the structure of the graph itself*.
The global, node, and edge **attributes** of a GN block can use arbitrary representational formats such as real-valued vectors and tensors, but other structures such as sequences, sets or graphs can also be used. The outputs of edges and nodes often correspond to lists of vectors or tensors, one per edge or node, and the global outputs correspond to a single vector/tensor, so that a GN’s output can be passed to other deep learning blocks like MLPs, RNNs.

For graph structure, there are typically two ways of defining how input data can be represented as a graph: input that explicitly specifies the relational structure such as knowledge graphs, social networks, parse trees, optimization problems; and input that the structure must be inferred or assumed, such as visual scenes, text corpora, programming language source code.

### Configurable within-block structure

placeholder until 4.5


[image-1]:	/assets/2018-11-04-relational-bias.jpg "Relational Inductive Biases"
[image-2]:	/assets/2018-11-04-aggregation.jpg
[image-3]:	/assets/2018-11-04-update.jpg