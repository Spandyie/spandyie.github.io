---
layout: post
title: "Flow Matching: Generative Models Without the Mess"
date: 2026-05-17
categories: [machine-learning, generative-models]
tags: [flow-matching, diffusion, pytorch, deep-learning]
---

# Flow Matching: Generative Models Without the Mess

I spent an afternoon trying to read the [Cambridge MLG blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) post on flow matching. The math was dense, the notation heavy, and I kept losing the thread somewhere around the Transport Equation. So I did what I usually do — I threw the equations away and tried to rebuild the intuition from scratch.

Here's what I ended up with.

---

## Start with the simplest question

Every generative model is solving the same basic problem: how do you turn random noise into something structured — an image, a molecule, a piece of audio?

GANs do it implicitly. A generator learns to reshape noise into data through an adversarial game, but you have no visibility into what's happening in the middle. It's a black box transformation.

What if instead you made the transformation explicit — step by step, over time?

Imagine watching noise slowly become an image. At t=0, pure noise. At t=1, a real sample. In between, something in between. That's the core idea behind both diffusion models and flow matching.

---

## The vector field

If you're a particle starting at some noise position x₀ and you need to reach a data sample x₁ by time t=1, what do you need at each moment along the way?

A direction. At every point in space and time, something that says: *go this way.*

That's the vector field u_t(x). The whole job of the neural network in flow matching is to learn this vector field. Given your current position and the current time, predict which direction to move.

The governing equation is just:

```
dx/dt = u_t(x_t)
```

Your velocity at time t equals the vector field at your current position. That's it.

---

## Why flow matching beats diffusion

Diffusion models were designed from physics — specifically, the mathematics of heat diffusion. The forward process adds noise via a stochastic differential equation borrowed from thermodynamics. The stochasticity wasn't chosen because it's geometrically elegant. It was chosen because the math worked out neatly in 2020.

The result is a winding, curved path through probability space. Getting from noise to data requires following that curve, which means many small steps — typically 20 to 50 at inference time.

Flow matching asks a simpler question: why not just draw a straight line?

If you linearly interpolate between x₀ and x₁:

```
x_t = x₀ + (x₁ - x₀) * t
```

...the target vector field at any point is just `x₁ - x₀`. Constant. Closed form. No noise schedule, no SDE, no score function.

Straighter path means fewer steps at inference. Models like Stable Diffusion 3 and Flux generate the same quality images in 4-8 steps that diffusion needed 50 for. Same architecture, cleaner objective.

---

## Training is just regression

The training objective follows directly. For each pair (x₀, x₁):

1. Sample a random time t
2. Compute x_t via interpolation
3. Predict the vector field at (x_t, t)
4. Minimize MSE against the target x₁ - x₀

No ODE simulation during training. No adversarial game. No ELBO. Just supervised regression against a closed-form target.

```python
def get_training_batch(n):
    x0 = sample_noise(n)
    x1 = sample_data(n)
    t = torch.rand(n, 1)
    xt = x0 + (x1 - x0) * t
    target = x1 - x0
    return t, xt, target
```

The neural network takes (x_t, t) concatenated and predicts the direction. At inference, you follow the predicted vector field with an Euler integrator.

---

## The crossing paths problem

When you pair noise and data samples randomly, paths cross. Two particles starting close together get assigned to distant targets and their straight-line paths intersect somewhere in the middle.

At the intersection point, the network sees the same position (x_t) being pulled in opposite directions. It gets confused — it can only output one direction, so it averages. That average is usually wrong for both particles.

The fix is Optimal Transport coupling. Instead of random pairings, match each noise point to the closest data point in the batch — minimizing total travel distance. In practice, you solve a small assignment problem per batch using `scipy.optimize.linear_sum_assignment` on the pairwise distance matrix.

```python
def ot_coupling(x0, x1):
    cost = torch.cdist(x0, x1)
    i, j = linear_sum_assignment(cost.detach().numpy())
    return x0[i], x1[j]
```

One extra line before interpolation. Nothing else changes.

The effect is measurable — lower training loss, cleaner samples, straighter paths that need even fewer inference steps.

---

## Building it in 30 lines

The full model — data sampling, interpolation, training loop, inference — fits in about 30 lines of PyTorch:

```python
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, xt, t):
        return self.net(torch.cat([xt, t], dim=1))

model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(2000):
    t, xt, target = get_training_batch(256)
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(xt, t), target)
    loss.backward()
    optimizer.step()

def sample(model, n=500, steps=100):
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n, 1), i / steps)
            x = x + model(x, t) * dt
    return x
```

Trained on a 2D moons dataset, this tiny 64-neuron network correctly learns to flow Gaussian noise into the two-moon distribution. The OT variant converges faster and generates sharper samples.

---

## What I actually find interesting here

The thing that stuck with me isn't the math — it's the design philosophy. Diffusion was designed from physics constraints that made the math tractable at the time. Flow matching was designed from geometry. Choose the simplest path that works, derive everything else from that choice.

The score function, which diffusion needs to estimate and which is notoriously hard to get right, disappears entirely. The vector field is its own thing — direct, interpretable, cheap to train against.

There's a broader lesson somewhere in there about what happens when you revisit old frameworks with cleaner priors.

---

*Code for this post, including the interactive visualization, is linked in the comments.*
