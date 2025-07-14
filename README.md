# Hamiltonian Neural Operator Experiments


### Introduction
The Hamiltonian $H$ is a function that represents the total energy of the system. Hamilton's equations state:

$$ \frac{\mathrm{d}\mathbf{q}}{\mathrm{d}t}=\frac{\partial H}{\partial \mathbf{p}}, \quad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}=-\frac{\partial H}{\partial \mathbf{q}}$$

Where $\mathbf{q}$ is the general position vector of the system, and $\mathbf{p}$ is the general momentum vector of the system.

### Model Architecture (Based on DeepONet)
Branch Network:
1. Inputs: system parameters $\mathbf{r}$ of the system
2. Outputs: latent embedding $\mathbf{b} \in \mathbb{R}^d$

Trunk Network:
1. Inputs: sampled phase-space vector $(\mathbf{q}, \mathbf{p})$
2. Outputs: latent embedding $\mathbf{t} \in \mathbb{R}^d$

Hamiltonian Output
 

  $$H(\mathbf{q}, \mathbf{p}; \mathbf{m}) = \langle b, t \rangle = \sum_{k=1}^{d} b_k t_k \quad \text{(inner product)}$$

  
### Loss Function
Sample $(q, p)$ across a range. At each point, enforce Hamilton's equations in the form

$$ \frac{\mathrm{d}\mathbf{q}}{\mathrm{d}t}-\frac{\partial H}{\partial \mathbf{p}}=0, \quad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}+\frac{\partial H}{\partial \mathbf{q}}=0$$

by using MSE loss.

In addition, we ensure that the Hamiltonian is constant by enforcing

$$\frac{\mathrm{d}H}{\mathrm{d}t}=0$$

In practice, the time derivative of the Hamiltonian is calculated as

$$\frac{\mathrm{d}H}{\mathrm{d}t} = \frac{\partial H}{\partial \mathbf{q}} \dot{\mathbf{q}} + \frac{\partial H}{\partial \mathbf{p}} \dot{\mathbf{p}} =  \frac{\partial H}{\partial \mathbf{q}}\left(  \frac{\partial H}{\partial \mathbf{q}}\right) = 0$$
  

### Integration
We are given from Hamilton's equations:

$$ \frac{\mathrm{d}\mathbf{q}}{\mathrm{d}t}=\frac{\partial H}{\partial \mathbf{p}}, \quad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}=-\frac{\partial H}{\partial \mathbf{q}}$$
  
We can integrate these derivatives numerically using Verlet integration to find trajectories over time. However, each calculation is done using our trained NN, and each derivative is far easier to compute (automatic differentiation of a neural network).
 
### Example: Mass-Spring System

The Hamiltonian of this system is written as 

$$H = \frac{1}{2}kq^2 + \frac{p^2}{2m}$$

Where $(p, q)$ represent the position and momentum of the mass on the spring. Now, from Hamilton's equations, we get

$$\frac{\mathrm{d}q}{\mathrm{d}t}=\frac{\partial H}{\partial p} = \frac{p}{m} $$ $$ \frac{\mathrm{d}p}{\mathrm{d}t}=-\frac{\partial H}{\partial q } = -kq$$

We can enforce these conditions, along with our zero derivative requirement, as

$$\frac{\partial H}{\partial p} - \frac{p}{m} = 0$$ $$ \frac{\partial H}{\partial q } - kq = 0$$ $$\frac{\mathrm{d}H}{\mathrm{d}t}=0$$


### Example: Pendulum

The Hamiltonian of this system is written as

$$
H(q, p) = \frac{p^2}{2ml^2} + mgl(1 - \cos q)
$$

where

* $q$ is the angular displacement,
* $p$ is the momentum,
* $m$ is the mass,
* $l$ is the length of the pendulum rod,
* $g$ is the gravitational acceleration.

From Hamilton’s equations we obtain

$$
\frac{\mathrm{d}q}{\mathrm{d}t} = \frac{\partial H}{\partial p}
=\frac{p}{ml^2} 
\quad
\frac{\mathrm{d}p}{\mathrm{d}t} =\frac{\partial H}{\partial q}
=-mgl\sin q.
$$

We enforce these conditions in the loss function, together with conservation of energy:

$$
\frac{\partial H}{\partial p} -\frac{p}{ml^2}
=0
$$

$$
\frac{\partial H}{\partial q} -mgl\sin q
=0
$$

$$
\frac{\mathrm{d}H}{\mathrm{d}t} = 0
$$





### Summary
This a formulation of a method to create a neural operator that can be used to calculate the Hamiltonian for any general system.
 
### Sources
1. [[1906.01563] Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563)
2. [[1910.03193] DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193)
3. [Deep Operator Networks (DeepONet) [Physics Informed Machine Learning] - YouTube](https://www.youtube.com/watch?v=CDCyOHXDRcI)
4. [Three-body problem - Wikipedia](https://en.wikipedia.org/wiki/Three-body_problem)