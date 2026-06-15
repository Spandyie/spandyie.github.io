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
In practice this update — the change applied to a weight matrix — is almost always low rank, i.e. a few directions undergo a huge update while the rest barely move. The idea in Muon (**M**oment**U**m **O**rthogonalized by **N**ewton-schulz) is that you still update along the dominant direction, while not squashing the rare directions, so that everyone gets an equal chance. Give the rare direction a fair shot, which would otherwise have been lost owing to their low singular values.

The issue is that a few directions with large singular values dominate the update, while the directions with smaller singular values get drowned out. An ideal optimizer should move along every dimension regardless of its singular value. This is exactly what Muon does; the idea is that instead of using the momentum in its full ranked form, wouldn't it make sense to extract the sign function, which can be written as:

$$\text{Sign}(M) = U V^T$$

For those of you who have done singular value decomposition, this formulation might seem very familiar, as it mirrors the SVD decomposition:

$$M = U \Sigma V^T$$ 

In the equation above, $U$ and $V$ are orthonormal matrices and $\Sigma$ is the diagonal matrix of singular values. The sign function squashes all singular values to 1, letting you use what matters from the matrix for parameter updates. 


## Muon Algorithm

Muon uses Nesterov momentum, which adds a look-ahead step. Instead of orthogonalizing the raw momentum, Muon first projects one step forward along the momentum direction and evaluates the gradient from there. This gives a more informed update direction. Here is what the algorithm looks like:

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

## In Practice

Newton-Schulz wins over SVD because GPUs excel at matmuls, not eigendecomposition. Since Muon only works on 2D matrices, reshape higher-dimensional tensors: a 4D conv kernel `(64,32,3,3)` becomes `(64,288)`, gets orthogonalized, then reshapes back.

In practice Muon doesn't run on every parameter. It's applied to the 2D hidden weight matrices, while the embeddings, the output head, and the 1D parameters (biases, LayerNorm gains) are left to AdamW. Those layers don't have the low-rank structure Muon is designed to fix, so orthogonalization buys nothing there. The result is a hybrid: Muon where it helps, AdamW everywhere else.

One last detail: because the orthogonalized update has all singular values pushed to ~1, its magnitude no longer depends on the gradient scale — it only depends on the shape of the matrix. So the update is scaled by roughly $\sqrt{\max(d_{in}, d_{out})}$ to keep it comparable to an AdamW step. Without this, Muon's learning rate wouldn't transfer across layers of different sizes.

---

*Muon was introduced by Keller Jordan et al. (2024). See the [original writeup](https://kellerjordan.github.io/posts/muon/) for the tuned Newton-Schulz coefficients and benchmark results.*
