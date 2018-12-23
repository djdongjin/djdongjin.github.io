
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>
---
layout: post
title: Compositional Language Understanding with Text-based Relational Reasoning
categories: [Deep Learning, NLP, Reasoning]
---

# Study of Reasoning

## Inductive Logic Programming

## Relational Reasoning

## Propositional Satisfiability (SAT Solver)

# Proposal: CLUTRR
> In the dataset, the task is to learn the compositional relations directly from text.
> If the model learns compositional elements, it should be able to re-use it to solve larger problems.

**Dataset construction**:
1. Create a family of relation graphs where nodes are entities and edges are relations.
2. Choose any two nodes, sample a path between them.
3. Replace the edges with relations which are chosen randomly from a dictionary of templates.
4. Predict the relation between the start and the sink/end of the path.

2018-12-07-compositionalQA-dataset.jpg

**Add noise as distractors**
- For a path with k relations and k+1 nodes, add *m* distractor sentences per node.
- Distractor sentence is an explanation of attributes which are not relevant to reasoning such as “Sam likes to play soccer.”

**Generalization in CLUTRR**
- Explicitly control the number of relations (k) in a path.
- Train on smaller number (k=3) and test on larger numbers (k=4,5).
- Generalizations are expressed as picking up compositional elements of relations from k=3, and then generalize them on k=4,5..
- Training and testing have *different distributions*.

**Setup**
- Story/Input sentences: $S=(s\_1,s\_2,...,s\_n)$ where $s\_i=(w\_1,..,w\_m)$
- Subset of words $w\_i$ are entities = ${e\_1,..,e\_n}$ which are anonymized in cloze-style.
- Each sentence $s\_i$ describes a relation R.
- Predict relation between a pair of query entities $(e\_i,e\_j)$