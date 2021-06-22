---
title: CMU 15-418 Parallel Computer Architecture and Programming 
layout: post
date: 2021-06-13 20:10:47 +07:00
modified: 2021-06-18 13:10:47 +07:00
tags: [parallel computing, c++, mooc]
description: Study notes for the CMU 15418 course (2020 Spring)
comments: true
# image: "./shell_evolution.png"
---

* TOC
{:toc}

## Lecture 1. Why Parallelism

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

> Parallelism v.s. concurrency: the two concepts are confusing. In my understanding, parallelism means, at a moment, multiple processors (e.g. cores) are working in parallel to complete a task(s). Concurrency only means there are multiple tasks/threads that are running. But they are not necessarily running in parallel but just interleaved. For example, OS may first start Task A and then put it in background and start Task B. After B is completed, OS continue to execute Task A.

This course uses **speedup** to measure parallelism when P processors are used, which is defined as *the ratio of executing time by using 1 processor and the exectuing time by using P processors*. Some potential issues when using parallelism and solutions inlcude:

1. Communication cost limits expected speedup (e.g., use P processors but only get P/2 speedup). Solution: minimize unnecessary communications.
2. Unbalanced work assignment limits expected speedup. Solution: balance workloads among processors to avoid any bottleneck.
3. Massive parallel execution that causes much higher communication cost compared to computation. (e.g., 100 cores to complete a simple task). Solution: partition the task to appropriate segmentations that are executable in parallel.

> In summary, we need a parallel thinking that includes *work decomposition, work assignment, and communication/syncronization management*.

## Lecture 2. Instruction-Level Parallelism (ILP)

Today's topic is *ILP*, which is mainly used in CPUs and done in the hardware level. But first, let's discuss the different parallelism strategies between two popular chips, CPU and GPU: in CPU, parallelism is incorporated into hardware-level, which dynamically schedules instructions, so it’s not (relatively) difficult to write parallel program on CPUs; however, parallelism on GPUs is done in software level, meaning programmers have to explicitly write code to handle/specify how to parallel the computation.

This also leads to different hardware design: CPU usually has only few powerful cores (tens) and most of the chip area is used for scheduling/communication (finding parallelism); in GPU, there is many simple cores (thousands). And softwares/programs are responsible for scheudling parallel execution (e.g., CUDA).

Let's back to *ILP* and see some ideas that CPUs use to speed up sequential code, including:

1. Pipelining & Superscalar: work on multiple instructions at once;
2. Out-of-Order (OoO) execution: dynamically schedule instructions (via dataflow graph);
3. Speculation: predict the next instruction to discover more independent work and roll back on incorrect predictions.

> All these techniques are implemented within the hardware, which is invisible to software. And in the software level, all instructions are still executed in-order, one-at-a-time.

First we define a simple CPU model which executes instruction in sequential, and assume an instruction execution is divided in to four stages:

1. Fetch: read the next instruction from memory;
2. Decode: figure out what to do and read inputs;
3. Execute: perform operations on ALU;
4. Commit: write results back to registers/memory.

Suppose each stage costs 1ns, such a model has a *latency* of 4ns/instruction and a *throughput* of 0.25instruction/ns.

> Latency: time spent to complete a job (request, instruction, etc).
> Throughput: number of jobs completed per time unit (1ns, 1day, etc).

### Pipelining

Pipelining executes multiple instructions that are **in different stages** in parallel. But it is still in-order execution and just keep fetching new instructions whenever possible.

It enables *ILP* by starting the next instruction immediately when the corresponding stage processor is available. Also since it's *ILP*, the latency of each instruction keeps unchanged (4ns/instruction). However, the throughput is improved 4 times, to 1instructions/ns(ideally). Because we have 4 stages, we can parallel 4 instructions at maximum.

One limitation of pipelining is that it requires *independent work*, meaning no *read-after-write* conflict. This introduces **data hazard** because mostly instructions are interrelated, especially those next to each other.

Solutions:

1. Stalling pipeline: when there is a *read-after-write* conflict, stall the pipeline to avoid to read the wrong value;
2. Forwarding data: after the write operation is done, it forwards the new value direclty to the next read instruction so that the read instruction doesn't need to wait until write is committed.

> Forward data is expensive, especially in deep & complex pipelines.

Another limitation of pipelining is **control hazard**: we pipeline instructions sequentially, but *branches* (goto, while, for) redirect execution to new location, making us fetch the wrong instructions.

Solutions:

1. Flushing pipeline: execute instructions sequentially. When there is an error, flush the wrong instructions and jump to the correct place;
2. Speculation: predict where to go next and execute instructions in the predicted place (can use ML to predict branches).

> Flush is also expensive in deep pipelines. For example, in nested loops, instruction errors happen more frequently.

### Out-of-Order (OoO) Execution

> Think of OoO execution as a topological sort on a dataflow graph.

Different from *pipelining* where we ensure instructions are executed in-order, *OoO execution* only ensures correct dataflow/true data dependency and thus avoid unnecessary data dependency (write-after-read, read-after-read).

> Dataflow increases parallelism by eliminating unnecessary dependences.

The first performance bottleneck in OoO is *Latency-bound*, in which lantency is bounded by a specific operation/instruction, making it unable to further improve latency. We can find the latency bound of a computation by its cretical path, which is *the longest path across iterations in a dataflow graph*.

<figure>
<img src="/assets/img/15418/2_OoO_arch.jpg" alt="Out-of-Order Microarchitecture">
<figcaption>An Out-of-Order microarchitecture. Only execute stage is out-of-order by utilizing an instruction buffer. The fetch/decode/commit stages are still in-order.</figcaption>
</figure>

With the above chip, a program is now *throughput-bounded*. That is because we can only execute 1 instruction at a time with a single execute unit (fetch and commit usually don't limit performance).

**Superscalar OoO** is the solution to unleash *throughput-bound*. It increase pipeline width by using multiple execution units, such that execution no longer limits performace (only dataflow does).

<figure>
<img src="/assets/img/15418/2_OoO_superscalar.jpg" alt="Out-of-Order Superscalar">
<figcaption>Out-of-Order superscalar. It has multiple execution units (and fetch/commit units) to increase pipeline bandwidth.</figcaption>
</figure>

<figure>
<img src="/assets/img/15418/2_OoO_superscalar_exec.jpg" alt="Out-of-Order Superscalar Execution">
<figcaption>Out-of-Order superscalar execution. Now there is only dataflow dependences.</figcaption>
</figure>

Another throughput limitation is *structural hazards*, which means data is ready but instructions cannot be executed because no hardward is available. A chip only contains limited numbers of different types of execution units such as floating-point units, interger units, etc.

> Register renaming: OoO processors eliminate false dependences (write-after-read, write-after-write) by transparently renaming registers.

### From ILP to Multicore

ILP hasn't gotten large boosts recently. One reason is that superscalar scheduling is complex. For example, to decide if it should issue two instructions, the CPU needs to compare all pairs of input/output registers to check if there is a read-after-write conflict.

Nowadays multicore is a more preferred choice where each core might become simpler. However, writing parallel software is still hard.

## Lecture 3. Modern Multi-Core Processor
