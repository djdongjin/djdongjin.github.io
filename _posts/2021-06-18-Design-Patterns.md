---
title: Head First Design Patterns
layout: post
date: 2021-06-18 23:27:00 +7:00
modified: 2021-06-18 23:27:00 +7:00
tags: [software engineering, book]
description: summary of the design patterns in the book
comments: true
---

* TOC
{:toc}

We can think design patterns as a set of best practices in Object-Oriented Design (OOD). It gives us *a shared vocabulary*, and simplifies communications with others by *thinking at the pattern level* instead of the nitty-gritty object level.

This book contains most of the design patterns defined in the textbook "Design Patterns: Elements of Reusable Object-Oriented Software" (Gangs of Four). For each pattern, I will summarize the **design principles** involved, its definition, class relation graph, and some code snippet.

## Strategy Pattern

**Definition**: strategy pattern defines a family of algorithms/classes, encapsulates each one, and makes them inter-changeable. Strategy lets algorithms vary independently from clients that use it.

**Design Principle**:

1. Encapsulate what varies: identify the aspects of your application that vary and separate them from what stays the same.
2. Program to an interface, not an implementation.
3. Favor composition over inheritance.

> Composition means a HAS-A relationship, whereas inheritance means IS-A. Composition gives many benefits such as loosely-coupled class structures, dynamically changing behaviors at runtime, etc.

Next, we will use an example to demonstrate the disadvantages of not using the pattern and how the strategy pattern can resolve them by applying these design principles.

Suppose we are designing a *Duck* class hierarchy, in which each type of duck can fly and quack but their flying/quacking behaviors are not the same. First, let's see two designs without the strategy pattern:

1. Use inheritance: first define an abstract *Duck* class with two abstract methods (fly, quack). Then for each actual *Duck* subclass, we inherit from *Duck* and implement the two methods.
2. Use interface: first define an abstract *Duck* class and two interfaces (Flyable, Quackable). Then for each actual *Duck* subclass, we implement the two interfaces when necessary.

<figure>
<img src="/assets/img/design_pattern/strategy_before.png" alt="Before Strategy Pattern">
<figcaption>Class relationships without strategy pattern. Left is using inheritance; right is using interface.</figcaption>
</figure>

Both of the two designs have disadvantages, including:

1. Classes are tightly-coupled: we cannot separate *Ducks* from *behaviors*. If we want to change a behavior implementation, we have to change all the client code (`Duck`) that use this behavior (duck implementations). If $n$ `Duck` classes use the behavior, we have to change it in all $n$ classes.
2. Low-level code reusing: if we want to create a `DuckCall` class hierarchy, we cannot reuse the quack behaviors defined in `Duck`.
3. Behaviors are binded to a class implementation and cannot be changed at runtime: after initializing a duck object, we cannot change its behaviors.

Let's see how *strategy pattern* can mitigate these issues. We know that what changes are the behaviors not ducks. So we first separate the two behaviors from ducks by defining two interfaces, `FlyBehavior` and `QuackBehavior`, each of which has a set of class implementing the behavior (Principle 1). Then, we add two objects of the interfaces to `Duck` that hold the specific behavior implementations (Principle 2 and 3). Finally, for different types of `Duck`, we just need to pass the expected behavior implementations (Principle 3).

<figure>
<img src="/assets/img/design_pattern/strategy_after.png" alt="After Strategy Pattern">
<figcaption>Class relationships strategy pattern. We can see that client code (`Duck`) is separated from algorithms (here, behaviors) and use them via composition.</figcaption>
</figure>

Now, let's discuss how the design with `strategy pattern` resolves the disadvantages:

1. We have a loosely-coupled class structures where client code (`Duck`) is separated from algorithms (`FlyBehavior` and `QuackBehavior`). If we want to change a behavior implementation, we don't need to change client code.
2. Since we already encapsulate behavior implementaions into separate interface/class structure that is not binded to client code, we can use them in other client code as well via *composition*.
3. We no longer bind behaviors to client code implementations. Instead, we use composition that holds behavior objects, so we can change the objects at runtime.

> The key here is that, a duck now *delegate* its flying and quacking behaviors to coresponding behavior objects, instead of using quacking/flying methods defined in the Duck class (or subclass).
