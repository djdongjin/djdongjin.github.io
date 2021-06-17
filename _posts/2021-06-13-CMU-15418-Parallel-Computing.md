---
title: CMU 15-418 Parallel Computer Architecture and Programming 
date: 2021-06-13 20:10:47 +07:00
modified: 2021-06-13 20:10:47 +07:00
tags: [parallel computing, c++, mooc]
description: Study notes for the CMU 15418 course (2020 Spring)
comments: true
# image: "./shell_evolution.png"
---

* TOC
{:toc}

# Lecture 1. Why Parallelism

This is a class focusing on various of techniques of writing parallel programs. It consists of 4 labs that use different parallel strategies, including:

1. SIMD (Single Instruction Multiple Data) and multi-core parallelism;
2. CUDA (parallelism on GPU devices);
3. Parallelism via shared-address space model;
4. Parallelism via message-passing model.

The first lecture discusses the history of parallel programming and some performance advances in this field, such as:

1. Wider data paths/address bits: 4 bit -> ... -> 32 bit -> 64 bit;
2. Efficient pipeline: lower Cycles per Instruction (CPI);
3. Instruction-level Parallelism (**ILP**) that detects independent instructions and execute them in parallel;
4. Faster clock rates: 10 MHz -> ... -> 3GHz.

> However, it's more difficult to continue to advance ILP and clock rates due to the **Power Density Wall**.

On the hardware side, a transion has been happening from supercomputer to cloud computing, where the former focuses on *lower latency* and the latter more focuses on *high throughput*.

> You can think latency means finishing a single task quickly whereas throughput means finishing more tasks in a period of time.

The formal definition of **Parallel Computing**: a collecitons of processing elements that cooperate to solve problems quickly. That means:

1. We need multiple processors to achieve parallelism;
2. These processors need to work in parallel to complete a task;
3. We care about performance/efficiency when using parallelism.

> Parallelism v.s. concurrency: the two concepts are confusing. In my understanding, parallelism means, at a moment, multiple processors (e.g. cores) are working in parallel to complete a task. Concurrency only means there are multiple tasks/threads that are running. But they are not necessarily running in parallel. For example, OS may first start Task A and then put it in background and start Task B. After B is completed, OS continue to execute Task A. They also why we need multiple processors to achieve parallelism but can achieve concurrency using one processor.

This course uses **speedup** to measure parallelism when P processors are used, which is defined as *the ratio of executing time by using 1 processor and the exectuing time by using P processors*. Some potential issues when using parallelism and solutions inlcude:

1. Communication cost limits expected speedup (e.g., use P processors but only get P/2 speedup). Solution: minimize unnecessary communications.
2. Unbalanced work assignment limits expected speedup. Solution: balance workloads among processors to avoid any bottleneck.
3. Massive parallel execution that causes much higher communication cost compared to computation. (e.g., 100 cores to complete a simple task). Solution: partition the task to appropriate segmentations that are executable in parallel.

> In summary, we need a parallel thinking that includes *work decomposition, work assignment, and communication/syncronization management*.

# Lecture 2. Instruction-Level Parallelism (ILP) - WIP

