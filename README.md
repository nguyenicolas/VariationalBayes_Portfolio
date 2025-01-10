# Portfolio Construction with Bayesian Approximations
This repository contains code to apply Bayesian Approximations for the problem of portfolio construction. In particular, it contains the implementation of the methods in the paper [**Variational Bayes Portfolio Construction**](https://arxiv.org/abs/2411.06192).

## Overview
The goal is to maximize the exponential utility with risk paramter $\lambda$ under the posterior distribution (using available historaical data $H_n$), that is, finding the portfolio vector

$$ \delta\_\lambda^* = \rm{argmin}_{\delta\in\mathcal{D}} \int\_{Y\_{n+1}} e^{-\lambda \delta^\top Y\_{n+1} } \pi(\mathrm{d}Y\_{n+1}\mid H_n), $$

where $Y\_{n+1}$ is the future unknown obervation (a random variable) and $\pi(\mathrm{d}Y\_{n+1}\mid H_n)$ is the posterior predictive distribution, which is **intractable** for most model. 

We propose three methods to find an approximation of $\delta\_\lambda^*$.

## Portfolio construction with Markov-Chain Monte-Carlo (MCMC)
...
## Portfolio construction with Sequential Monte-Carlo (SMC)
...
## Portfolio construction with Variational Bayes (VB)
...
