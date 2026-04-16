---
title: "2D Scatter Density as a Visual Conditional Independence Test (v26)"
date: 2026-04-16 00:00:00 +0700
categories: [Research, Causal Discovery]
tags: [causal-inference, deep-learning, conv2d, thesis]
math: true
toc: true
---

> This is the second writeup in the ADIA Lab series. The [first post]({% post_url 2026-04-15-adia-causal-discovery-v11 %}) covered v11 — the 1D sorted curve approach with structural attention bias, achieving 81.14% balanced accuracy. This post covers v26, a fundamentally different representation: replacing sorted curves with 2D scatter density images processed by Conv2D. The best v26 result reaches **81.19% local / 80.96% LB**, comparable to v11 but capturing complementary signal.

---

## Disclaimer

This blog post is written by Claude with me as a proof reader and editor. The technical content is my work, but the writing style is a collaboration.

All the source code for this project is available in this [repository](https://github.com/aqu4holic/Graduation-Thesis-BsC-2026/).
All the experiments result is available in this [Google Sheet](https://docs.google.com/spreadsheets/d/13IvbOsEDasD2m-yo4f6JgL_T6YuEoQXT5RCv_9JcHBE/edit?usp=sharing).

---

## Motivation

The v11 solution represents each variable pair as a sorted observation curve — sort 1000 points by $u$'s values, read how $v$ changes across that ordering. A Conv1D encoder reads that length-1000 sequence. This is powerful for detecting the *functional shape* of the conditional expectation $E[v \mid u]$, but it is a 1D projection of what is fundamentally a 2D relationship.

The question is: what does the joint distribution of $(K, X)$ look like as a 2D object, and can a Conv2D model read causal structure directly from that image?

The theoretical motivation comes from d-separation. The 8 causal roles are defined by whether $K$ is marginally or conditionally (in)dependent of $X$ and $Y$. A 2D scatter density between $K$ and $X$ is a nonparametric representation of their joint distribution — exactly the object you would inspect visually to check whether two variables are independent. A uniform density means no relationship. A structured density means dependence. Conv2D on this image is learning to classify causal roles by recognizing these distributional signatures — which is a more direct encoding of what conditional independence testing actually measures.

---

## Representation: 8-Channel Density Image

For each non-$X$/$Y$ node $v$, the representation is an 8-channel $32 \times 32$ density image. Each channel is a 2D histogram of a variable pair, Gaussian-smoothed and normalized to sum to 1.

**Raw density channels (ch0–3):**

$$\text{ch}_0 = \text{density}(v, X), \quad \text{ch}_1 = \text{density}(v, Y)$$
$$\text{ch}_2 = \text{density}(X, v), \quad \text{ch}_3 = \text{density}(Y, v)$$

Channels 0–1 capture whether $v$ relates to $X$ and $Y$. Channels 2–3 are the transpose views. These are not redundant: Conv2D learns local spatial patches, and the local patch at position $(i, j)$ in $\text{density}(v, X)$ vs $\text{density}(X, v)$ encodes different local structure — the transpose breaks the symmetry in a way the model can exploit.

**ANM residual channels (ch4–7):**

$$\text{ch}_4 = \text{density}(v, \epsilon_X), \quad \text{ch}_5 = \text{density}(X, \epsilon_v)$$
$$\text{ch}_6 = \text{density}(v, \epsilon_Y), \quad \text{ch}_7 = \text{density}(Y, \epsilon_v)$$

where $\epsilon_X$ is the multivariate kernel regression residual of $X$ on all other variables (bandwidth 0.5). These encode causal direction asymmetry in 2D form. If $v \to X$ is the true causal direction, regressing $X$ on $v$ and others captures the effect, so $\epsilon_X$ is noise, and $\text{density}(v, \epsilon_X)$ is a uniform cloud. The reverse direction residual $\epsilon_v$ will show structure, making $\text{density}(X, \epsilon_v)$ non-uniform. The model learns to read this asymmetry between ch4 and ch5.

**Construction details:** Each channel is built with min-max normalization to $[0, 1]$, then a $32 \times 32$ histogram, then Gaussian smoothing with $\sigma = 4.0$ for raw channels and $\sigma = 2.0$ for ANM channels. The finer sigma for ANM channels is intentional — ANM residuals have sharper local structure that coarser smoothing would wash out. The density is normalized to sum to 1 (joint density, not conditional/row-normalized).

---

## Architecture: Dual-Pipeline with Edge Context Fusion

The most important architectural lesson from the v26 experiments was that **the 2D node representation alone is insufficient**. A pure Conv2D classifier on the 8-channel image, with no graph context, scored ~51–53%. This is not much better than chance on 8 classes.

The reason is that the node image only encodes the relationship of $v$ to $X$ and $Y$ — it has no information about the relationships among all other variables in the graph. Causal role classification is inherently a graph-level task: whether $v$ is a Confounder or a Mediator depends on the full graph structure, not just the $(v, X, Y)$ triplet.

The solution is to fuse the 2D node representation with the v11 edge pipeline as a context encoder.

**Edge pipeline (context encoder):** The full v11 edge pipeline runs on the graph — Conv1D over 8-channel sorted curves, structural self-attention over all $O(p^2)$ edge pairs. This produces a $d$-dimensional embedding per edge. The four embeddings corresponding to edges $(v \to X, v \to Y, X \to v, Y \to v)$ are aggregated into a node context vector. Crucially, the edge pipeline also has its own classification head (edge head) with an auxiliary loss — this is not used at inference, only at training to force the edge pipeline to learn meaningful representations rather than collapse.

**Node pipeline:** The 8-channel $32 \times 32$ image goes through a hierarchical Conv2D encoder with downsampling, producing a $d$-dimensional node embedding.

**Fusion:** The node embedding and edge context vector are concatenated and passed to the final node classification head.

**Loss:** $\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{node}}$, with $\lambda = 0.7$. Favoring the edge loss slightly gave better results than equal weighting, because the edge pipeline needs strong gradient signal to learn the context representations that the node pipeline depends on.

---

## Ablations and What Didn't Work

**Pure node 2D (no edge context): ~51–53%.** As discussed above, graph context is not optional. The edge pipeline is load-bearing as a context encoder, not just an auxiliary head.

**12-channel (adding kernel regression density): 80.42% vs. 8-channel 80.47%.** The natural extension of v11's multi-bandwidth kernel channels to the 2D setting — compute density$(v$, kernel coeff$(u, v))$ — did not help. The reason is an axis mismatch: in the 1D sorted curve, the kernel regression coefficient is a scalar computed per observation and can be sorted alongside the raw values. In the 2D density, the axes represent the raw variable values, and kernel coefficients live in a different space. Placing a kernel coefficient on the x-axis of a histogram while the y-axis is a raw variable value creates a mixed-axis image where Conv2D's spatial inductive bias is inappropriate — the model cannot extract meaningful local patches from a grid where the axes have incompatible units.

**Learnable smoothing (v26f): 79.37%.** Instead of hand-tuning the dual sigma ($4.0$ for raw, $2.0$ for ANM), the idea was to use a trainable depthwise Conv2D layer to learn per-channel smoothing kernels from near-raw histograms ($\sigma = 0.5$ base). This regressed by over 1%. The likely reason: at $\sigma = 0.5$ the histograms are very noisy (each of 1000 points contributes to only one bin), and the learnable smoothing layer has to both denoise and extract features simultaneously. The fixed hand-tuned sigmas act as a preprocessing step that separates noise reduction from feature learning. This is also evidence that the dual sigma values ($4.0$ vs $2.0$) are doing meaningful work — they are not arbitrary but reflect the different noise characteristics of raw scatter vs. ANM residuals.

---

## The Translation Invariance Caveat

Conv2D assumes **translation equivariance**: a feature detector that fires at position $(i, j)$ will also fire at $(i+1, j)$, weighted equally in the learned kernel. This is the right inductive bias for natural images, where a cat's ear looks the same regardless of where it appears in the frame.

For scatter density images, this assumption does not hold. Position in the density image carries absolute meaning: the top-right corner of $\text{density}(v, X)$ represents "high $v$, high $X$" — a fundamentally different configuration from the bottom-left corner "low $v$, low $X$." A confounder produces a specific blob pattern in a specific region of the joint density, and that region matters.

Despite this theoretical mismatch, Conv2D still learns useful features. The likely explanation is that the relevant causal signal is primarily encoded in the *shape and structure* of the density — whether it is uniform, elongated, curved, or asymmetric — rather than its absolute position. The Gaussian smoothing at $\sigma = 4.0$ also spreads density across a large portion of the grid, reducing the dependence on absolute position. And with enough data the model can learn to partially compensate by learning position-specific filters implicitly.

However, this is a genuine theoretical limitation. A position-aware architecture — such as a Vision Transformer with absolute position embeddings, or a model that explicitly includes the grid coordinates as additional input channels — would be more faithful to the representation. This is an open direction.

---

## Results

| Configuration | Local BA | LB |
|---|---|---|
| v25a (4ch raw density, no ANM) | 79.56% | — |
| v25b (4ch raw density) | 78.99% | — |
| v26b (8ch, early fusion) | 79.17% | — |
| v26c (pure node2D, no edge context) | 51.51% | — |
| v26d (favor edge loss, $\lambda$=0.7 inverted) | 80.37% | — |
| v26e 8ch base | **80.47%** | — |
| v26e 12ch (+ kernel density channels) | 80.42% | — |
| v26f (learnable smoothing) | 79.37% | — |
| v26e 8ch + XY aug | **81.19%** | **80.96%** |
| v26e 12ch + XY aug | 80.42% | — |

The best v26 result (81.19%) is comparable to v11+ (81.14%). More importantly, the two models are making different predictions — v11 reads 1D functional curves while v26 reads 2D joint distributions. This suggests that ensembling the two representations could yield further gains.

---

## v11 vs. v26: Different Models, Complementary Signal

The two representations are fundamentally different in what they encode.

v11 (1D sorted curves) is sensitive to the *functional shape* of $E[K \mid X]$ — the nonlinear trend, curvature, and heteroscedasticity of the conditional mean. It is optimal for detecting asymmetries in the functional relationship between variables.

v26 (2D density) is sensitive to the *joint distributional shape* of $(K, X)$ — the spread, correlation structure, and tail behavior of the joint distribution. It is more natural for detecting whether two variables are marginally independent (uniform density) vs. dependent (structured density).

Collider classification illustrates this difference. A Collider ($X \to K \leftarrow Y$) is marginally independent of both $X$ and $Y$, but conditioning on $K$ induces dependence. The 1D curve for a Collider looks like noise — no functional trend. The 2D density for a Collider also looks close to uniform marginally, but the residual channels (ch4–7) will show the collider bias pattern when $K$ is conditioned on. Each representation has a different angle on the same underlying structure.

---

## Summary

The v26 contribution is framing 2D scatter density as a visual conditional independence test: the joint density image of $(K, X)$ is a nonparametric representation of their relationship, and Conv2D learns to classify causal roles by recognizing density patterns that correspond to d-separation structures in the graph.

The key architectural lesson is that the node image alone is insufficient — it requires the edge pipeline as a graph context encoder, fused via a dual-pipeline architecture with an auxiliary edge classification loss.

The main caveats are: (1) Conv2D's translation equivariance assumption does not hold for density images where position carries absolute meaning; (2) kernel regression density channels do not transfer from 1D to 2D due to axis mismatch; (3) learnable smoothing fails because it conflates denoising with feature learning.

Both v11 and v26 reach comparable performance (~81%) but capture different aspects of causal structure, making them strong candidates for ensembling.
