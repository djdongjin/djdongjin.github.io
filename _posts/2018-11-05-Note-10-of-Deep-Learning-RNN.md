---
layout: post
title: " Note 10 of Deep Learning: Recurrent and Recursive Neural Network "
categories: [Deep Learning, RNN]
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

A recurrent neural network generates the output of time step $t$ not only according to $x^{(t)}$, but also according to the previous history, represented by the hidden state, $h^{(t)}$. Given an input, a RNN will first update its hidden state, then generate the output according to the new hidden state. Main computing functions include:

$$\begin{aligned}
a^{(t)} &= f(U \times x^{(t)} + W \times h^{(t-1)}) + b \\\\\\
h^{(t)} &= \tanh(a^{(t)}) \\\\\\
o^{(t)} &= V \times h^{(t)} + b\\\\\\
\hat{y^{(t)}} &= softmax(o^{(t)}) \\\\\\
L^{(t)} &= loss(y^{(t)}, o^{(t)}).
\end{aligned}$$

as shown in the following picture:

![][image-1]

## Computing the Gradient in a RNN

Gradient computing in a RNN is almost the same as in a feedforward NN, except that backpropogating from $t^{(i+1)}$ to $t^{(i)}$ in the unfolded RNN, instead of from layer $i+1$ to layer $i$ as in a feedforward NN. Such a method is called *back-propagation through time* (BPTT).

## Bidirectional RNNs

Bi-RNNs combine an RNN that moves forward through time beginning from the start of the sequence with *another* RNN that move backward through time beginning from the end of the sequence, so that, when computing the output $o^{(t)}$, there will be two hidden states, $h^{(t)}$ corresponding the state of the forward RNN, and $g^{(t)}$ corresponding the state of the backward RNN, which will be concatenated to generate $o^{(t)}$ by utilizing both the past and the future.

![][image-2]

## Encoder-Decoder, Sequence-to-Sequence Architectures
> Able to map an input sequence to an output sequence with different length.
Sequence-to-sequence (seq2seq) architectures contains two RNNs, one of which is called *encoder* and used to generate the context representation of the input sequence, $C$, and the other is called *decoder* and used to generate the output sequence given $C$. Typically the last hidden state of encoder $h\_{n\_x}$ is used as the context representation $C$. The innovation of such an architecture is that the length $n\_x$ and $n\_y$ can vary from each other.

The two RNNs are trained jointly to maximize the average of $log P(y^{(1)},...,y^{(n\_y) \mid x^{(1)},...,x^{(n\_x)}}$ over all $(x,y)$ pairs in the training set.

One limitation of seq2seq is when the context *C* has a dimension that is too small to properly summarize a long input sequence. This limitation can be resolved by an *attention mechanism* that learns to associate elements of the sequence $C$ to elements of the output sequence.

![the last hidden state of encoder is used as the context representation C.][image-3]

## Recursive Neural Networks

Recursive NN differentiate itself from Recurrent NN with a different computational graph which is structured as a deep tree, rather than a chain-like structure of RNNs, as illustrated below:

![][image-4]

One Advantage of recursive nets is that it can shorten the depth of input sequence from $O(\rho)$ (in recurrent nets) to $O(\log \rho)$.

The question here is how to structure a recursive net tree. One way is constructing a tree that is not related to the data such as balanced binary tree. Another way is constructing using external methods, such as the parse tree of the sentence when applied to NLP tasks.

## Long-short Term Memory (LSTM) and Other Gated RNN
> Used for solving gradient vanish and explode in Long-Term Dependencies

- LSTM
- GRU

### Clipping Gradients
Since gradient explode is caused by too large gradient, we can limit the gradient and clip it when exceeding some threshold.

### Regularizing to encourage information flow





[image-1]:	/assets/2018-11-05-RNN.jpg "RNN: circular v.s. unfolded"
[image-2]:	/assets/2018-11-05-BiRNN "Bidirectional-RNN"
[image-3]:	/assets/2018-11-05-seq2seq-no-attention.jpg "Sequence-to-sequence without attention"
[image-4]:	/assets/2018-11-05-recursive.jpg "Recursive NN"