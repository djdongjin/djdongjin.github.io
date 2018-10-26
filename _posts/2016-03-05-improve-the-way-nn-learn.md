---
layout: post
title: "Improving the Way Neural Networks Learn"
categories: [Deep Learning, Machine Learning]
---

In this chapter the author explains a suite of techniques which can be used to improve the backpropagation and the way networks learn, including better choice of cost function, named **cross-entropy**; four **regularization methods** (L1 and L2 regularization, dropout, artificial expansion of the training data); a better method for initializing the weights in the network; a set of heuristics that help choose good hyper-parameters for the network.

## The cross-entropy cost function

We replace the quadratic cost with cross-entropy cost function, because neurons learn too slow caused by the  vanish gradient when the value of the quadratic cost is close to 0 and 1.

We define the cross-entropy cost function for one neuron by:

$$
C = -\frac{1}{n} \sum_x [y\ln a + (1-y) \ln (1-a)]
$$

where $n$ is the total number of examples of training data.

Two properties make it reasonable to interpret the cross-entropy as a cost function. First, it's non-negative, because, (a) all individual terms in the sum operation are negative, since both logarithms are of numbers between 0 and 1; and (b) there is a minus sign out the front of the sum.

Second, when the neuron's actual output, $a$, is close to the desired output, y, for all training inputs, the cross-entropy will be close to zero. We can easily prove this by assuming that either y = 0 and $a \approx 0$, or y =1 and $a \approx 1$. So the contribution of such an example to the cost will be low provided the actual output is close to the desired output.

In addition, the cross-entropy also has a benefit that it avoids the problem of learning slowing down, which can be proved by computing the partial derivative of the cross-entropy cost with respect to the weights and biases.

$$\begin{aligned}
\frac{\partial C}{\partial w_j} &= -\frac{1}{n} \sum_x \left(\frac{y}{\sigma(z)} -\frac{(1-y)}{1-\sigma(z)} \right)
\frac{\partial \sigma}{\partial w_j} \\
&=  -\frac{1}{n} \sum_x \left( 
\frac{y}{\sigma(z)} 
-\frac{(1-y)}{1-\sigma(z)} \right)\sigma'(z) x_j \\
&= \frac{1}{n}
\sum_x \frac{\sigma'(z) x_j}{\sigma(z) (1-\sigma(z))}
(\sigma(z)-y) \\
&= \frac{1}{n} \sum_x x_j(\sigma(z)-y)\\
\frac{\partial C}{\partial b} &= \frac1n \sum_x (\sigma(z) - y)
\end{aligned}$$

It's also easy to generalize the cross-entropy to many-neuron multi-layer networks.

$$
C = -\frac1n \sum_x \sum_i [y_i \ln a^L_i + (1-y_i) \ln (1 - a^L_i)]
$$

## Softmax

We can see softmax as a new type of output layer which is the same with a sigmoid layer, except that, instead of applying the sigmoid function to get the output, we apply the so-called *softmax* function to the $z^L_i$ to get the output, $a^L_i$:

$$
a^L_i = \frac{e^{z^L_i}}{\sum_k e^{z^L_k}}
$$

To see how a softmax layer could address the learning slowdown problem, we first define the *log-likelihood* cost associated to a single training input, (x, y):

$$
C = -\ln a^L_y
$$

And then we consider the partial derivative, $\partial C / \partial w^L_{jk}$ and $\partial C / \partial b^L_j$:

$$\begin{aligned}
  \frac{\partial C}{\partial b^L_j} &= a^L_j-y_j\\
  \frac{\partial C}{\partial w^L_{jk}} &= a^{L-1}_k (a^L_j-y_j)
\end{aligned}$$

It's useful to think of a softmax output layer with log-likelihood cost as being quite similar to a sigmoid output layer with cross-entropy cost (compare their partial derivative).
  
## Overfitting and regularization

Overfitting is a phenomenon that, although the model's accuracy on the training data is pretty high, it's accuracy on test set is too low. Overfitting means that the model we use now is so complicated that it can recognized some special patterns of the small training set rather than the overall patterns of the whole type of data, which also means it cannot perfectly generalized to other data it never saw before.

We use a new and independent dataset, validation set, to prevent overfitting by evaluating different trial choices of hyper-parameters, so that we can find the most appropriate architecture of the network.

In addition, one of the best ways of reducing overfitting is to increase the size of the training data. With enough training data, the model is more likely to recognize patterns on a bigger dataset, which also means it is more likely to recognize common patterns on the whole dataset (the ability of generalization).

### Regularization

Obviously, another way of reducing overfitting is to reduce the size of the network (which means less parameters). However, large networks have the potential to be more powerful than small networks. Through *regularization* techniques, we can keep the benefit of large networks and meanwhile reduce overfitting.

#### Weight decay / L2 regularization

L2 regularization is to add an extra term, called *regularization term*, to the cost function. Here is the regularized cross-entropy cost:

$$
C = -\frac 1n \sum_x \sum_j [y_j \ln a^L_j + (1-y_j)\ln (1-a^L_j)] + \frac{\lambda}{2n} \sum_w w^2.
$$

where $\lambda>0$ is known as *regularization parameter*, and biases are not included in the regularization term.

We can see regularization as a way of compromising between finding small weights and minimizing the original cost function, and the relative importance of the two elements depends on the regularization parameter $\lambda$. With regularization, the network prefers to learn small weights. Large weights will be learned only if they improve the original cost (the first term) considerably.

#### L1 regularization

$$\begin{aligned}
C &= C_0 +  \frac{\lambda}{n} \sum_w |w| \\
\frac{\partial C}{\partial w} &= \frac{\partial C_0}{\partial w} +\frac{\lambda}{n} \rm sgn(w).
\end{aligned}$$

#### Dropout

Unlike L1 and L2 regularization, dropout doesn't modify the cost function. Instead, dropout modify the network itself. The idea behind dropout is, when training the network, it randomly deletes some neurons to reduce the scale of the network. Here, "delete" means that some neurons become transparent only in this mini-batch,  rather than being deleted forever. Next mini-batch, these neurons will be restored and other randomly chosen neurons will become transparent.
![](/assets/2016-03-05-dropout.jpg)

## How to choose hyper-parameters?

Some heuristics that can be used for hyper-parameter optimization are introduced in this section. (However, there is more unknown heuristics you can try.)
 
 **Broad strategy** 

