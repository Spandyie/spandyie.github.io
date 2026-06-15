---
layout: post
title: "Muon: Orthogonalized Updates for Efficient Optimization"
date: 2026-06-14
tags: [optimization, machine-learning, nlp, muon]
---

Stochastic gradient descent with momentum can be written as follows

$$m_t = \beta m_{t-1} + (1-\beta) \nabla L$$

$$\theta_t = \theta_{t-1} - \alpha m_t$$

Or in pseudocode:
```
Loop Until Convergence
	average_momentum = β × last_momentum + (1-β) × Gradient
	parameter = parameter - learning_rate × average_momentum
```
In practice the above update, i.e. the change in parameter with respect to input, is always low rank, i.e. a few parameters undergo a huge update while the rest update very little. The idea in MUON is that you still update the parameter along the similar direction, while not squashing the rare direction, so that everyone gets an equal chance. Give the rare direction a fair shot, which would otherwise have been lost owing to their low singular values.

The issue is that few directions with large singular values dominate the update, while the directions with smaller singular values get drowned out. An ideal optimization should optimize all the dimensions regardless of the eigenvalues along the most important dimension. This is exactly what Muon optimizer does; the idea is that instead of using the momentum in its full ranked form, wouldn't it make sense to extract the sign function, which can be written as:

$$\text{Sign}(M) = U V^T$$

For those of you who have done singular value decomposition, this formulation might seem very familiar, as it mirrors the SVD decomposition:

$$M = U \Sigma V^T$$ 

In the equation above, $U$ and $V$ are orthonormal matrices and $\Sigma$ is the diagonal matrix of singular values. The sign function squashes all singular values to 1, letting you use what matters from the matrix for parameter updates. 


## Muon Algorithm

Muon uses Nestrov momentum which has an additional look ahead step. Instead of orthogonalizing raw momentum, Muon first projects one step forward along the momentum direction and evaluates the gradient from there. This gives a more informed update direction. Here is what the algorithm looks like:

```
B0 = 0
for t in time:
    gradient = get_gradient(loss)
    B[t] = mu * B[t-1] +  gradient
    g̃ = mu * B[t] + gradient # the Nesterov "look-ahead"
    O[t] = NewtonSchulz(g̃)
    Parameter[t] = parameter[t-1] - learning_rate * O[t]

return parameter[t]
```

## Newton-Schulz

Newton-Schulz is basically an approach to orthogonalise a 2D vector in a cheap, memory-efficient way.  The real reason NS is cheaper is that SVD requires finding eigenvectors of `MᵀM`, which doesn't map well to GPU matmuls. NS is just repeated matmuls — which GPUs are built for. Newton-Schulz provides a simple iterative algorithm to get the sign function.

Here is what SVD does 
$$M = U \cdot \Sigma \cdot V^T$$

Now by some mathematical trick, if you can squash all the singular values to 1:
$$\text{sign}(M) = UV^T$$

Newton-Schulz is an iterative approach to calculate $UV^T$ without doing SVD. The algorithm starts with:

$$X_0 = \frac{M}{||M||}$$

And applies the equation below N times:
$$X_{t+1} = a \cdot X + b \cdot X \cdot (X^T X) + c \cdot X \cdot (X^T X)^2$$

where $a$, $b$, and $c$ are constants. This ensures every direction in the gradient gets equal weight in the update — rare directions with small singular values are no longer drowned out by dominant ones.

The Newton-Schulz algorithm is cheaper than SVD because it only requires matrix multiplication, which can be run efficiently on GPUs. Another important thing to note is that Muon only applies to 2D matrices, so if you have a higher-dimensional tensor, you need to reshape it to 2D first. For convolutions, which might be a 4D tensor `(64,32,3,3)`, you need to reshape this as `(64,288)`, orthogonalize it using NS, and reshape it back to the original form.
