# CTCR Bilinear Motivation

CTCR measures the part of a hidden state that appears only when two conditions are jointly present:

```text
r_AB = h(A+B) - h(A) - h(B) + h(0)
```

If the model represented the two conditions additively, then `h(A+B)` would be approximately `h(A) + h(B) - h(0)`, and the residual would be near zero. A large CTCR residual therefore indicates a non-additive interaction: the model state contains information that is not explained by either condition alone.

The bilinear term is motivated by a simple low-dimensional interaction model. Suppose condition A activates a small subspace `U`, condition B activates a small subspace `V`, and the sleeper-style behavior depends on their interaction rather than their independent activation. The interaction can be approximated as a low-rank bilinear form:

```text
score(A, B) ~= z_A^T W z_B
```

where `z_A` and `z_B` are low-dimensional coordinates of the two condition states and `W` has low effective rank. Linear sparse features can detect `z_A` or `z_B` individually, but they do not naturally represent the conjunction unless the joint state has already been collapsed into a single linear direction. A bilinear sparse feature can directly represent pairwise interactions between two sparse latent factors.

This predicts three falsifiable outcomes:

1. CTCR should carry more sleeper-trigger signal than simpler residuals such as `h(A+B)-h(0)`.
2. Bilinear sparse features should gain most on examples where neither condition alone is strongly predictive.
3. The advantage should weaken on ordinary multi-turn jailbreaks if those attacks mainly produce smooth representational drift rather than discrete cross-turn conjunctions.
