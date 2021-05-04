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

|     -         | B = 0           | B = 1         |
|:-------------:| :-------------: |:-------------:| 
|-----
|A = 0          | 0.3             | 0.7           | 
|-----
|A = 1          | 0.8             | 0.2           | 

From this table we can read that $$p(B = 0|A = 0) = 0.3$$, $$p(B = 1|A = 0) = 0.7$$
etc. It can be observed that each value should be between 0 and 1, and each row needs to sum to 1, since the row sum is $$\sum_{b\in val(B)} p(B = b|Pa(B))$$, where
val(B) denotes all possible values of B, and Pa(B) the parents of B.

# Learning
Learning generally means to infer some knowledge from given data. As described
above, a Bayesian network consists of a graph (structure) and a table-CPD for
each node (probabilities). Thus, we could imagine learning either the structure,
the probabilities or both. Structure learning can be a significantly more difficult task, especially as the number of variables grows, since there is an exponential
number of possible graphs that can represent the variables. Further, to decide
which structure is best, we need to compare all structures - possibly removing
some cases that deviate from prior knowledge too much - which in itself requires
us to estimate the parameters for that given structure. That is, parameter
estimation is a subtask of structure learning, and to keep this tutorial focused we
will only cover parameter estimation. For small toy examples the “true” structure
can often be known by reasoning about causality (we dodge the philosophical
discussion on whether causality actually exists or not). As an example, imagine
we roll two dice and record the individual values along with the sum. It is clear
that the two dice do not influence each other, but that both influence the sum.
And even though the sum can tell us something about the values of the dice (if
the sum is low, each dice rolled low), it does not affect the rolls. The true graph
is thus

<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
</style>

<img src="https://samueltober.github.io/samuel-tober/images/diagram-20210504.png" >

In this case, the sum is deterministic given the dice rolls, but the same reasoning
would go for a random variable. Imagine that we don’t know the probabilities
for the dice. We could then make a number of experiments, and use the collected
data to infer all the probabilities. Typically, we want to find the parameters $$\theta}$$
(probabilities) that maximise the (log) likelihood of the observed data. Formally,
if D is the dataset and $$\theta$$ are the values of the CPDs that we want to determine, the
maximum likelihood solution is $$\theta$$ = argmax_{\theta}p(D|\theta). Due to the independences
in Bayesian networks, the likelihood function can be decomposed so that we
can treat parts of the graph independently. In essence, what we end up with is
simply counting occurrences and using the frequencies as estimates.

# Missing Data
We say that we have missing, incomplete or partially observed data when we
have datapoints for which some variables are not recorded. This could be due to
human error (the variable was not recorded by mistake), the variable not being
applicable (a patient did not take a certain medical test) or the variable not being
directly measurable (these may or may not have a semantic interpretation, and
are usually called latent variables). One way of handling these cases would be to
simply ignore the missing cases, and do maximum likelihood on the observed
data. This could be a valid approach if only little data is missing, otherwise
we risk reducing the data a lot, which will lead to worse estimates. Another
approach is to fill in missing values with default values or random values from
a given distribution. If we have some prior knowledge on what these values or
distributions could be, this might be okay, especially if only few data are missing.
Otherwise, the estimates will be biased to whatever default values we use.

An even bigger issue is that even if we have some clever way of handling missing
data (such as expectation maximisation covered later), it could fail depending
on the process that determines if a value is missing or not. If the process is
independent of the data itself, then we are safe; otherwise we could be in trouble.

A sufficient condition for when missing data is okay is that the process that
determines whether data is missing or not is independent of the data - this is
called missing completely at random (MCAR). This was the case in the first
coin example. However, a weaker assumption is actually sufficient, namely that
the process is independent of the missing values given the observed values - this
is called missing at random (MAR). Consider an example where we have two
coins, A and B, and every time B shows “head” we don’t record the value of A.
Clearly, this is MAR but not MCAR. Sometimes, we can make the data MAR
by adding more variables.

