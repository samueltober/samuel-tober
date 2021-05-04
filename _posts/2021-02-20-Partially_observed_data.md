---
layout: dark-post
title:  "Partially Observed Data"
date:   2021-02-20 21:49:05 +0200
categories: Projects
usemathjax: true
---

In this project I focused on the problem of learning with incomplete or missing data. I implemented the EM-algorithm for a small network, and investigated how missing data affects
the learning. But before we get to that, let’s briefly recap some building blocks of PGMs (probabilistic graphical models) and 
what we mean by learning in PGMs.

# Bayesian Networks
Simply put, a probabilistic graphical model consists of a structure, or graph,
that describes how variables influence each other, and weights that describes how
strong the influence is. Here, we limit our attention to Bayesian networks, which
can be represented as a directed graph and a table-CPD (conditional probability
distribution) for each node. Each node represents a variable, and a directed edge
from node A to node B means that the value of B depends directly on the value
of A. In this case, A would be called a parent of B. Each variable can assume a
finite set of discrete values (e.g. $${0, 1}$$, $${a_0, a_1, a_2}$$), and to fully specify the
Bayesian network we need to know the probability of each value given the values
of parent nodes. These are stated in a table, an example of which is given below,
where A is the only parent of B.

|               | B = 0           | B = 1         |
|:-------------:| :-------------: |:-------------:| 
|-----
|A = 0          | 0.3             | 0.7           | 
|-----
|A = 1          | 0.8             | 0.2           | 

From this table we can read that $$p(B = 0|A = 0) = 0.3$$, $$p(B = 1|A = 0) = 0.7$$
etc. It can be observed that each value should be between 0 and 1, and each row needs to sum to 1, since the row sum is $$\sum_{b\in val(B)} p(B = b|Pa(B))$$, where
val(B) denotes all possible values of B, and Pa(B) the parents of B.

