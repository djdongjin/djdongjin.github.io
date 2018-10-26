---
layout: post
title: "Advice for Applying Machine Learning"
categories: [Deep Learning, Machine Learning]
---

This section is mainly about how to debug and optimize a learning algorithm, or, how to architecture or parameters of a learning model.
When we build a model making large errors in its predictions, here are some things we can do next:

1. Get more training examples
2. Try smaller sets of features
3. Try getting additional features
4. Try adding polynomial features
5. Try decreasing $\lambda$
6. Try increasing $\lambda$

## Evaluating a hypothesis

We can divide all data into two parts, one named training set, for training our model, and another named test set, for testing how well our model performs on examples it never sees before. Now we can evaluate a model as below:

1. Learn parameter $\theta$ from training data by minimizing training error $J\left(\theta\right)$.
2. Compute test error.

## Model selection & training/validation/test sets

What can we use for evaluating different models? The answer is another set, named validation data. Now we have three datasets: training set, for training models, validation set, for selecting the best model, test set, for evaluating our model.
A better scale for these three sets is: 70% for training set, 15% and 15% for validation set and test set.

## Diagnosing bias vs. variance

First let's define the meaning of bias and variance. We can say that  an underfit model has high bias, and an overfit model has high variance.

![](/assets/2017-07-23-bias-variance.jpg)

This picture shows how cross validation error and training error($J_{cv/training}\left(\theta\right)$) changes with the degree of polynomial *d*. 

From the picture we can see that if you have both high J_cv and J_training, and they are almost equal, you are suffering a bias problem(*underfit*), since the $d$ is relatively low, and if you have a low J_training but a very high J_cv, you are suffering a variance problem(*overfit*), because you model performs well on training set but poorly on validation set.

## Regularizaiton & bias/variance

![](/assets/2017-07-23-regularization1.jpg)
![](/assets/2017-07-23-regularization2.jpg)

This picture shows the relation between our model and parameter $\lambda$. When $\lambda$ is large, we will learn a model whose all parameters $\theta$ are very small, so we get an underfitting model, which means high bias problem. When $\lambda$ is intermediate, we will get a model with appropriate regularization, so we just get the right model. When $\lambda$ is small, which means that the regularization term is also small so that it can't work as we expect, we will get an overfit model, which means high variance problem.

In that case, how can we choose the right regularization parameter $\lambda$? That's the way, first, we choose several values, like 0, 0.01, 0.02, 0.04 ... 10; second, for each value of $\lambda$, we minimize $J\left(\theta\right)$ and get a trained $\theta$, and then, we calculate $J_{cv}(\theta)$. After doing this for each value of $\lambda$, we pick the $\lambda$ with the smallest $J_{cv}(\theta)$

## Learning curves

This section is mainly about *the relation between errors and training set size*. Here are three pictures to completely illustrate this relation:

![](/assets/2017-07-23-Learning-curves.jpg)

This one shows this relation on the right model. When training set size $m$ is pretty small, $J_{training}(\theta)$ is very tiny while $J_{cv}(\theta)$ is large(overfit). With the increase of $m$, $J_{training}(\theta)$ becomes bigger and $J_{cv}(\theta)$ becomes smaller, step by step, since we use more training examples to train our model. At last, these two J are almost equal and don't change again.

![](/assets/2017-07-23-high-bias.jpg)

This shows this relation on a model with high bias (underfit). Since such a model is underfitted, so $J_{training}(\theta)$ becomes larger very fast but $J_{cv}(\theta)$ becomes smaller slowly, and finally, both of them stop on a relatively large value, which means our model has a high error on both of these datasets. Nonetheless, getting more training data can't solve this problem because the main reason for this is that current model can't summarize features of so many examples, let alone more examples.

![](/assets/2017-07-23-high-variance.jpg)

Last one, high variance(overfit). When $m$ is relatively small, we have a large $J_{cv}(/theta)$ and a small $J_{training}(/theta)$, and the changing trends of these two variables is slower than those in the right model( the 1st picture), because our model is enough complicated to fit more training examples than the right model, and also because of this, its performance on validation set is relatively poor. And finally,  $J_{cv}(/theta)$ is a little larger than $J_{training}(/theta)$, but remember, getting more training examples can help improve this model because it is overfitted now.

## Error analysis

Here is a recommended approach for building a machine learning system:

1. Start with a simple algorithm that you implement quickly. Ant then test it on the cross-validation data.
2.  Plot learning curves to check if there is a high bias/variance problem in your algorithm, according that you can decide what to do, more data, more or less features, increasing or decreasing lambda, etc.
3.  Error analysis: Manually examine the examples in cross validation set that your algorithm made errors on. See if you spot some systematic trend in what type of examples it is making errors on. Like, if you have a spam classifier, 50% errors it makes on are examples including word "sale", now you can add a new feature to your classifier that see examples including "sale" as negative. Generally, we can categorize errors based on two rules: 1) what type of example the error is. 2) what features you think would have helped algorithm handle them correctly.

## Error metrics for skewed classes

When classes of our dataset is skewed, only using accuracy cannot evaluate a learning algorithm correctly. Say, if we train a algorithm using a dataset with 99 positive examples and 1 negative examples, we can get an algorithm with 99% accuracy just by letting it always returns 1 no matter what the input is. But is it a good algorithm. Absolutely not. So now we should use error metrics rather than accuracy to evaluate an algorithm. Here is it, with two terminologies:

|      type      | 1(Actual) | 0(actual)|
|:--------------:|:---------:|:---------:|
| **1(predict)** |     TP    |     FP    |
| **0(predict)** |     FN    |     TN   |

$Precision = \frac{True Positive}{True Positive+False Positive}=\frac{TP}{TP+FP}$
(of all examples where we predicted 1, what fraction actually is 1?)
$Recall=\frac{True Positive}{True Positve+False Negative}=\frac{TP}{TP+FN}$
(of all examples that actually are 1, what fraction did we correctly predict?)

![](/assets/2017-07-23-precision-recall.jpg)

Now, we have two new methods to evaluate a learning algorithm, but how to compare precision/recall numbers? The answer is F1 Score(F score)
$F = 2\frac{PR}{P+R}$
