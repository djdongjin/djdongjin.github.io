---
layout: post
title: "CS20si--TensorFlow for Deep Learning Research"
categories: [DeepLearning, MachineLearning, Mooc, TensorFlow]
---
We firstly define a computation graph to express the computing process, then we submit a tensor, which is the result we want, to a session to get the real result. TensorFlow need to be executed in this way because the core part of it is implemented by another language rather than Python. If we execute our program step-by-step, the time consuming is too heavy. So we need to define our program and then execute it as a whole for a better performance.


There are two ways to execute a graph, and if only one session is used in the whole context, we can create a default session by tf.InteractiveSession()

```
a, b = tf.constant(2), tf.constant(3)
c = a+b
# the 1st way to execute a graph
sess = tf.Session()
sess.run(c)
# the 2nd way to execute a graph
c.eval(session=sess)
# the 3rd, if we use just one session
sess = tf.InteractiveSession()
c.eval()
```

Generally speaking, there are 3 ways to express a tensor: constant, placeholder, Variable. tf.constant is used to express constant object; tf.placeholder(type) can express a type of object, and we give it a value only when we need to use it. tf.Variable is usually used to express weights or biases which we need to tune via training, and it need to be initialized before we use it, usually by tf.global_variables_initializer()

