# Portfolio Construction with Bayesian Approximations
This repository contains code to apply Bayesian Approximations for the problem of portfolio construction. In particular, it contains the implementation of the methods in the paper [**Variational Bayes Portfolio Construction**](https://arxiv.org/abs/2411.06192).

## Overview
The goal is to maximize the exponential utility with risk paramter $\lambda$ under the posterior distribution (using available historaical data $H_n$), that is, finding the portfolio vector

$$ \delta\_\lambda^* = \rm{argmin}_{\delta\in\mathcal{D}} \int\_{Y\_{n+1}} e^{-\lambda \delta^\top Y\_{n+1} } \pi(\mathrm{d}Y\_{n+1}\mid H_n), $$

where $Y\_{n+1}$ is the future unknown obervation (a random variable) and $\pi(\mathrm{d}Y\_{n+1}\mid H_n)$ is the posterior predictive distribution, which is **intractable** for most model. 

We propose three methods to find an approximation of $\delta\_\lambda^*$.

we denote the objective function 

$$\mathcal{R}(\delta) = \int\_{Y\_{n+1}} e^{-\lambda \delta^\top Y\_{n+1} } \pi(\mathrm{d}Y\_{n+1}\mid H_n).$$

## Portfolio construction with Markov-Chain Monte-Carlo (MCMC)
This consists in applying gradient descent on the objective function $\delta\mapsto\mathcal{R}(\delta)$, and remarking that the gradient $\nabla\mathcal{R}$ can be written as follows:

$$\nabla_\delta\mathcal{R} = -\lambda \mathbb{E}\_{\tilde{\pi}}[Y\_{n+1}],$$ 

where the form of $\tilde{\pi}$ is discussed in section 5.2. We can approximate this gradient as

$$\nabla_\delta\mathcal{R} \approx -\lambda \frac{1}{M}\sum\_{m=1}^M z\^{(k)},$$

where we sample a chain $(z\^{(k)})_k$ via MCMC.

## Portfolio construction with Sequential Monte-Carlo (SMC)
...
## Portfolio construction with Variational Bayes (VB)
...
