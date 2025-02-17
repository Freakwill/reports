
From the local mean to Transformer

**Abstract**

This report introduces a general machine learning framework, termed the localization method, which is rooted in the concept of the local mean and serves as the cornerstone for the self-attention mechanism in the Transformer architecture. The framework is rigorously defined through the establishment of the local model and localization trick, providing a strict and formal expression of its underlying principles. Furthermore, the report delves into the connections between the localization method and an array of other models, such as kernel methods, lazy learning, MeanShift, relaxation labeling, linear neighborhood propagation, and fuzzy inference. By examining these relationships, the report aims to illuminate the broader implications and potential applications of the localization method within the field of machine learning.

**Keywords** Local mean, Local model/Localization trick, Kernel method, Meanshift, Self-attention mechanism/Transformer


**Notations**
- sample space: $\mathcal{X}$
- sample squence sp.: $\mathcal{X}^*$
- random variable(rv): $x$
- sample: $X$ or $\{x_i\}$
- distribution: $x\sim p$
- machine learning model: $y\sim f(x,\theta)$
- approximation: $X\approx Y$


## Content

- Definition of Local models/Localization trick
- Kernel
- Local mean/self-local mean
- Lazy learning
- Local regression and other local models
- Self-attention mechanism/Transformer
- Extention


## Local models

Assume $l(x,\theta)$ is a loss function and the decision model is an opt. model as follows,
$$
\min_\theta \sum_i l(x_i,\theta)   ~~~ (A)
$$
where $\{x_i\}$ is the sample.


### Local decision models

**Definition（Local decision models）**
Given a target point $x_*$, the local model of (A) is defined as
$$
\min_\theta \sum_i K(x_*,x_i)l(x_i,\theta)   ~~~ (B)
$$
where $l(x,\theta)$ is a loss at the single point $x$, $\{x_i\}$ is the sample. The solution of (B) is related to $x_*$, thus denoted as $\hat{\theta}(x_*)$. We call (B) is the localization of (A).

The total loss on the sample is
$$
J(K)=\sum_i l(x_i,\hat{\theta}(x_i;K))
$$
where $J(K), \hat{\theta}(x_i;K)$ stress that the loss is also related to the kernel $K$, or the kernel matrix $K(X,X)=\{K(x_i,x_j)\}$.

### Localization kernels

A (localization) kernel is a bivariate function on sample space $K:\mathcal{X}\times\mathcal{X}\to \mathbb{R}$ with certain conditions.

The condition for constructing a kernel $K(x,y)$ is very weak: 
1. Non-negativity, i.e., $K(x,y) \geq 0$; 
2. Decrease with distance, i.e., $K(x,y)$ is a decreasing function of $d(x,y)$, where $d$ is the distance on $\mathcal{X}$;
3. Symmetry (not necessarily), i.e., can be represented as $K(x-y)$. Such a kernel (or univariate function $K(\cdot)$) is also known as a convolution kernel.

All of above are optional. Only the first one is assumed by default in most cases.

*Remark.* $K(x,y)$ could be seen as the joint distribution (un-normalized) of $x$ and $y$.

*Remark.* Please note that the localization method is different from the kernel method. There are many connections between the two, but there are also obvious differences. However, the two can definitely be unified.

#### Normalization

$\tilde{K}(x,y):=K(x,y)/\int K(x,y)\mathrm{d}y$, the denominator must be positive.

*Remark.* $\tilde{K}(x,y)$ could be seen as the conditional/transition distribution from $x$ to $y$.

A normalized kenerl: $\int K(x,y)\mathrm{d}y=1$
smoothing kenerl: $\int K(x,y)\mathrm{d}y>0$ (at least $\neq 0$)
de-smoothing kernel: $\int K(x,y)\mathrm{d}y=0$

#### Feature mapping

$\phi, \psi:\mathcal{X}\to H$ where $H$ is inner product space, called the feature space.

Construct kernel by two feature mapping.
$K(x,y)=\phi(x)\cdot \psi(y)$ or $=e^{\phi(x)\cdot \psi(y)}$ (for non-negativity)
More generally, $K(x,y)=F(\phi(x), \psi(y))$. Here $\phi$ and $\psi$ are called the query-mapping and key-mapping resp.

*Example.*
$$K(x,y)=\int p(z|x)p(z|y)dz$$

#### (Emperical) Kernel matrix

$K(X,Y)=\{K(x_i,y_j)\}$ where $X=\{x_i\},Y=\{y_j\}$ are samples.

*Remark.* $K(X,X)$ is shorten as $K(X)$.

Similarly feature matrix: $\phi(X):=\{\phi(x_i)\},\psi(Y)=\{\psi(y_i)\}$

Hence, $K(X,Y)=\phi(X)\psi(Y)^T$ or $e^{\phi(X)\psi(Y)^T}$

Normalization: 
$$\mathcal{K}(X,Y):=\{\frac{K(x_i,y_j)}{\sum_j K(x_i,y_j)}\}\\
=K(X,Y)\oslash K(X,Y)\mathbf{1}\\
=D^{-1}K(X,Y)
$$
where $D=\mathrm{dial}(\{\sum_j K(x_i,y_j)\})$

#### Laplacian

Laplacian (kernel) of $K$: $L(x,y)=\delta_{xy} -K(x,y)$
normalized Laplacian (kernel) of $K$: $\tilde{L}(x,y)=\delta_{xy} -\tilde{K}(x,y)$

Laplacian (matrix): $L(X,Y)=D -K(X,Y)$ 
Laplacian (matrix): $\tilde{L}(X,Y)=I -\tilde{K}(X,Y)$

*Remark.* The Laplacian is stemmed from the graph theory. Any graph has its (normalized) laplacian.

### Monte Carlo local models

Monte Carlo(re-sampling) form:
$$\sum_i K(x_0,x_i)l(x_i,\theta) \sim \sum_i l(x_i,\theta), x_i\sim p(x_i|x_0)\sim K(x_0,x_i)$$

Specially, stochastic form:
$$\sum_i K(x_0,x_i)l(x_i,\theta) \sim l(x_i,\theta), x_i\sim K(x_0,x_i)$$


### Local model for machine learing

Following def. reflects the original idea of local.
**localization for machine learing**
Given a machine learing model $y\sim f(x,\theta)$, we define its localized model as
$$
\min_\theta \sum_i K(x_0,x_i)|y_i-f(x_i,\theta)|^2
$$
or for some purposes,
$$
\min_\theta \sum_i K(x_0,y_0,x_i,y_i)|y_i-f(x_i,\theta)|^2
$$

### Neighbourhood/Topology

As a typical type of local model:
$\min_\theta \frac{1}{N}\sum_i K(x_0,y_0,x_i,y_i)|y_i-f(x_i,\theta)|^2$


## local mean

The terminal goal of the regression is calculate the **conditional expection**:
$$
E(y|x)\approx \sum_{x_i\in U_x} y_i \quad\text{or}\quad \sum_j y_i p(x_i|x)
$$
where $U_x$ is a certain neighorhood of $x$.

**Def**
A local mean of the sample $\{(x_i,y_i)\}$ on target var $x_0$, is defined as,
$$
\hat y(x_0):=\sum_i K(x_0,x_i)y_i/\sum_i K(x_0,x_i)
$$

### self-local model

*Fact*
Any local regression is reduced to the local mean, approximately.

**Def. self-local mean**
The **local mean mapping** (or **mean shifting**) is defined as,
$$
m(x_0):=\sum_i K(x_0,x_i)x_i/\sum_i K(x_0,x_i):\mathcal{X}\to\mathcal{X}
$$

The self-local mean is indeed the local mean of the sample $\{(x_i,x_i)\}$. Reversely, The local mean is a special type of self-local mean on $\mathcal{X}\times\mathcal{Y}$.

*Remark* The famous **MeanShift algorithm** is the iteration of the mapping $m$.


### Local model for autoregression(AR)

Considier the AR or dynamics as follows,
$$
x_{t+1}\sim f(x_t,t), x_t\in\mathcal{X},t=1,2,\cdots
$$
where $\mathcal{X}$ is a linear space.

According to the def x.x, the local model of (x.x) could be expressed as,
$$
\hat x_{t+1}:= \sum_s K(x_t,t,x_{s},s)x_{s+1}/\sum_s K(x_t,t,x_s,s)
$$
regarding the tuple $(x_{t}, t)$ as the input and $x_{t+1}$ as the output.


We prefer the following form.
**Def Local AR/Dynamical system**
The local model of (x.x) is called the local AR or local DS, expressed as,
$$
m(x_{t}):= \sum_s K(x_t,t,x_s,s)x_{s}/\sum_s K(x_t,t,x_s,s)
$$
where $t$ could take any type of values in principle.

### Self-attention 1

**Def self-attention(Relative Positional Embedding)**
One possible design of $K$ is $K_1(x_t,x_s)K_2(t,s)$, where $K_1$ represent the grahic-dependence of elements in $\mathcal{X}$ statically and $K_2$ represents the "position-encoding". 

Hence what the localization really dose is to transform the time-dependency to graphic-dependency.

When the kernel is unrelated to the positions, namely $K$ is designed as $K(x_t,x_s)$, it is reduced to local average.

*Example*
Let $K_2(t,s)=1_{|t-s|<\delta}$,
$$
m(x_{t}):= \sum_{|t-s|<\delta} K(x_t,x_s)x_{s}/\sum_{|t-s|<\delta} K(x_t,x_s)
$$


### Self-attention 2

**Def self-attention(Absolute Positional Embedding)**
self attention is a local dynamical system with kernel $K(x_t+p(t),x_s+p(s))$ where $p(t)$ is the position-encoding of $t$.

**Def Transformer**
The Transformer, the most popular large model structure, is identified with the local model with $m^{(6)}(x_{t})$, iterating $m$ six times.

Strictly, the local dynamical system should learn $K$ to implement the self-attention, that is to solve the followint opt problem:
$$
\min_K J(K):=\|X-KX\|^2_{F}
$$
where $K=\{K(x_t,t,x_s,s)\}_{st},X=\{x_{t}^{(j)}\}_{tj}$。

*Fact* Transformer is a seq. model of MeanShift.

## Embedding method

Assume $\mathcal{X}$ is discrete.

**Def self-local mode**
$$
\hat{x}_{t}:= \argmax_x\sum_{x_s=x} K(x_t,t,x_s,s)
$$

**Def**
Assume $c(x)$ represents the onehot encoding (responding proba.) of $x$.
$$
\hat{c}(x_t):= \sum_{s} K(x_t,t,x_s,s) c(x_s)/\sum_{s} K(x_t,t,x_s,s)
$$

## Other models

### Label slacking


### IFT

### Fuzzy inference


### self-adaptive kernel
1. param. kernel, $K(x,x';\alpha)$
2. multikernel, $\sum_lK_l(x,x')$
3. discrete kernel, 
   - for continuous rv, $K=K(x_i,x_j):N\times N$ or $K=\phi(x_i)\psi(x_j)^T,\phi,\psi:N\times d$
   - for discrete rv, $K=K(i,j):\mathcal{X}\times \mathcal{X}$ or $K=\phi(i)\psi(j)^T,\phi,\psi:\mathcal{X}\times d$


### Learning the kernels

- continous case:
$$
\max_{K}\|X-\tilde{K}X\|_F
$$

- disrete case:
$$
\max_{\phi,\psi}\sum_i(\phi(x_i)\psi(X)^TC(X))_i\oslash (\phi(x_i)\psi(X)^T1_{N\times p})
$$

## Extention

### Heterogenuous kernel

### super kernel

## Categorical-style method

$\mathrm{loc}(M)$ represents to apply the localization trick on a given model $M$.

I call the "functor" loc, constructor, mapping a model to another.

1. $\mathrm{loc}(M)\simeq \mathrm{loc}(\mathrm{loc}(M))$ (in categorical-sence, not strictly)
2. $\mathrm{loc}(M)$ must be complicative then $M$, if $M$ is linear.

<!-- ## Two toy examples

`local-demo.py`

![](local-average.png)

![](image-local-average.png)

 -->

*References*

- Yao-Hung Hubert Tsai, Shaojie Bai Makoto Yamada, Louis-Philippe Morency, Ruslan Salakhutdinov. Transformer Dissection: An Unified Understanding for Transformer’s Attention via the Lens of Kernel.

- Tianyang Lin, Yuxin Wang, Xiangyang Liu, Xipeng Qiu. A survey of transformers.
  
- Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V and Salakhutdinov, Ruslan. Transformer-xl: Attentive language models beyond a fixed-length context,, arXiv:1901.02860, 2019

