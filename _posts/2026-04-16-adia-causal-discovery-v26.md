---
title: "2D Scatter Density as a Visual Conditional Independence Test (v26)"
date: 2026-04-16 09:00:00 +0700
categories: [Research, Causal Discovery]
tags: [causal-inference, deep-learning, conv2d, thesis]
math: true
toc: true
---

<!-- > This is the second writeup in the ADIA Lab series. The [first post]({% post_url 2026-04-15-adia-causal-discovery-v11 %}) covered v11 — the 1D sorted curve approach with structural attention bias, achieving 81.14% balanced accuracy. This post covers v26, a fundamentally different representation: replacing sorted curves with 2D scatter density images processed by Conv2D. The best v26 result reaches **81.19% local / 80.96% LB**, comparable to v11 but capturing complementary signal. -->
<!---->
<!-- --- -->
<!---->
<!-- ## Disclaimer -->
<!---->
<!-- This blog post is written by Claude with me as a proof reader and editor. The technical content is my work, but the writing style is a collaboration. -->
<!---->
<!-- All the source code for this project is available in this [repository](https://github.com/aqu4holic/Graduation-Thesis-BsC-2026/). -->
<!-- All the experiments result is available in this [Google Sheet](https://docs.google.com/spreadsheets/d/13IvbOsEDasD2m-yo4f6JgL_T6YuEoQXT5RCv_9JcHBE/edit?usp=sharing). -->
<!---->
<!-- --- -->
<!---->
<!-- ## Motivation -->
<!---->
<!-- The v11 solution represents each variable pair as a sorted observation curve — sort 1000 points by $u$'s values, read how $v$ changes across that ordering. A Conv1D encoder reads that length-1000 sequence. This is powerful for detecting the *functional shape* of the conditional expectation $E[v \mid u]$, but it is a 1D projection of what is fundamentally a 2D relationship. -->
<!---->
<!-- The question is: what does the joint distribution of $(K, X)$ look like as a 2D object, and can a Conv2D model read causal structure directly from that image? -->
<!---->
<!-- The theoretical motivation comes from d-separation. The 8 causal roles are defined by whether $K$ is marginally or conditionally (in)dependent of $X$ and $Y$. A 2D scatter density between $K$ and $X$ is a nonparametric representation of their joint distribution — exactly the object you would inspect visually to check whether two variables are independent. A uniform density means no relationship. A structured density means dependence. Conv2D on this image is learning to classify causal roles by recognizing these distributional signatures — which is a more direct encoding of what conditional independence testing actually measures. -->
<!---->
<!-- --- -->
<!---->
<!-- ## Representation: 8-Channel Density Image -->
<!---->
<!-- For each non-$X$/$Y$ node $v$, the representation is an 8-channel $32 \times 32$ density image. Each channel is a 2D histogram of a variable pair, Gaussian-smoothed and normalized to sum to 1. -->
<!---->
<!-- **Raw density channels (ch0–3):** -->
<!---->
<!-- $$\text{ch}_0 = \text{density}(v, X), \quad \text{ch}_1 = \text{density}(v, Y)$$ -->
<!-- $$\text{ch}_2 = \text{density}(X, v), \quad \text{ch}_3 = \text{density}(Y, v)$$ -->
<!---->
<!-- Channels 0–1 capture whether $v$ relates to $X$ and $Y$. Channels 2–3 are the transpose views. These are not redundant: Conv2D learns local spatial patches, and the local patch at position $(i, j)$ in $\text{density}(v, X)$ vs $\text{density}(X, v)$ encodes different local structure — the transpose breaks the symmetry in a way the model can exploit. -->
<!---->
<!-- **ANM residual channels (ch4–7):** -->
<!---->
<!-- $$\text{ch}_4 = \text{density}(v, \epsilon_X), \quad \text{ch}_5 = \text{density}(X, \epsilon_v)$$ -->
<!-- $$\text{ch}_6 = \text{density}(v, \epsilon_Y), \quad \text{ch}_7 = \text{density}(Y, \epsilon_v)$$ -->
<!---->
<!-- where $\epsilon_X$ is the multivariate kernel regression residual of $X$ on all other variables (bandwidth 0.5). These encode causal direction asymmetry in 2D form. If $v \to X$ is the true causal direction, regressing $X$ on $v$ and others captures the effect, so $\epsilon_X$ is noise, and $\text{density}(v, \epsilon_X)$ is a uniform cloud. The reverse direction residual $\epsilon_v$ will show structure, making $\text{density}(X, \epsilon_v)$ non-uniform. The model learns to read this asymmetry between ch4 and ch5. -->
<!---->
<!-- **Construction details:** Each channel is built with min-max normalization to $[0, 1]$, then a $32 \times 32$ histogram, then Gaussian smoothing with $\sigma = 4.0$ for raw channels and $\sigma = 2.0$ for ANM channels. The finer sigma for ANM channels is intentional — ANM residuals have sharper local structure that coarser smoothing would wash out. The density is normalized to sum to 1 (joint density, not conditional/row-normalized). -->
<!---->
<!-- --- -->
<!---->
<!-- ## Architecture: Dual-Pipeline with Edge Context Fusion -->
<!---->
<!-- The most important architectural lesson from the v26 experiments was that **the 2D node representation alone is insufficient**. A pure Conv2D classifier on the 8-channel image, with no graph context, scored ~51–53%. This is not much better than chance on 8 classes. -->
<!---->
<!-- The reason is that the node image only encodes the relationship of $v$ to $X$ and $Y$ — it has no information about the relationships among all other variables in the graph. Causal role classification is inherently a graph-level task: whether $v$ is a Confounder or a Mediator depends on the full graph structure, not just the $(v, X, Y)$ triplet. -->
<!---->
<!-- The solution is to fuse the 2D node representation with the v11 edge pipeline as a context encoder. -->
<!---->
<!-- **Edge pipeline (context encoder):** The full v11 edge pipeline runs on the graph — Conv1D over 8-channel sorted curves, structural self-attention over all $O(p^2)$ edge pairs. This produces a $d$-dimensional embedding per edge. The four embeddings corresponding to edges $(v \to X, v \to Y, X \to v, Y \to v)$ are aggregated into a node context vector. Crucially, the edge pipeline also has its own classification head (edge head) with an auxiliary loss — this is not used at inference, only at training to force the edge pipeline to learn meaningful representations rather than collapse. -->
<!---->
<!-- **Node pipeline:** The 8-channel $32 \times 32$ image goes through a hierarchical Conv2D encoder with downsampling, producing a $d$-dimensional node embedding. -->
<!---->
<!-- **Fusion:** The node embedding and edge context vector are concatenated and passed to the final node classification head. -->
<!---->
<!-- **Loss:** $\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{node}}$, with $\lambda = 0.7$. Favoring the edge loss slightly gave better results than equal weighting, because the edge pipeline needs strong gradient signal to learn the context representations that the node pipeline depends on. -->
<!---->
<!-- --- -->
<!---->
<!-- ## Ablations and What Didn't Work -->
<!---->
<!-- **Pure node 2D (no edge context): ~51–53%.** As discussed above, graph context is not optional. The edge pipeline is load-bearing as a context encoder, not just an auxiliary head. -->
<!---->
<!-- **12-channel (adding kernel regression density): 80.42% vs. 8-channel 80.47%.** The natural extension of v11's multi-bandwidth kernel channels to the 2D setting — compute density$(v$, kernel coeff$(u, v))$ — did not help. The reason is an axis mismatch: in the 1D sorted curve, the kernel regression coefficient is a scalar computed per observation and can be sorted alongside the raw values. In the 2D density, the axes represent the raw variable values, and kernel coefficients live in a different space. Placing a kernel coefficient on the x-axis of a histogram while the y-axis is a raw variable value creates a mixed-axis image where Conv2D's spatial inductive bias is inappropriate — the model cannot extract meaningful local patches from a grid where the axes have incompatible units. -->
<!---->
<!-- **Learnable smoothing (v26f): 79.37%.** Instead of hand-tuning the dual sigma ($4.0$ for raw, $2.0$ for ANM), the idea was to use a trainable depthwise Conv2D layer to learn per-channel smoothing kernels from near-raw histograms ($\sigma = 0.5$ base). This regressed by over 1%. The likely reason: at $\sigma = 0.5$ the histograms are very noisy (each of 1000 points contributes to only one bin), and the learnable smoothing layer has to both denoise and extract features simultaneously. The fixed hand-tuned sigmas act as a preprocessing step that separates noise reduction from feature learning. This is also evidence that the dual sigma values ($4.0$ vs $2.0$) are doing meaningful work — they are not arbitrary but reflect the different noise characteristics of raw scatter vs. ANM residuals. -->
<!---->
<!-- --- -->
<!---->
<!-- ## The Translation Invariance Caveat -->
<!---->
<!-- Conv2D assumes **translation equivariance**: a feature detector that fires at position $(i, j)$ will also fire at $(i+1, j)$, weighted equally in the learned kernel. This is the right inductive bias for natural images, where a cat's ear looks the same regardless of where it appears in the frame. -->
<!---->
<!-- For scatter density images, this assumption does not hold. Position in the density image carries absolute meaning: the top-right corner of $\text{density}(v, X)$ represents "high $v$, high $X$" — a fundamentally different configuration from the bottom-left corner "low $v$, low $X$." A confounder produces a specific blob pattern in a specific region of the joint density, and that region matters. -->
<!---->
<!-- Despite this theoretical mismatch, Conv2D still learns useful features. The likely explanation is that the relevant causal signal is primarily encoded in the *shape and structure* of the density — whether it is uniform, elongated, curved, or asymmetric — rather than its absolute position. The Gaussian smoothing at $\sigma = 4.0$ also spreads density across a large portion of the grid, reducing the dependence on absolute position. And with enough data the model can learn to partially compensate by learning position-specific filters implicitly. -->
<!---->
<!-- However, this is a genuine theoretical limitation. A position-aware architecture — such as a Vision Transformer with absolute position embeddings, or a model that explicitly includes the grid coordinates as additional input channels — would be more faithful to the representation. This is an open direction. -->
<!---->
<!-- --- -->
<!---->
<!-- ## Results -->
<!---->
<!-- | Configuration | Local BA | LB | -->
<!-- |---|---|---| -->
<!-- | v25a (4ch raw density, no ANM) | 79.56% | — | -->
<!-- | v25b (4ch raw density) | 78.99% | — | -->
<!-- | v26b (8ch, early fusion) | 79.17% | — | -->
<!-- | v26c (pure node2D, no edge context) | 51.51% | — | -->
<!-- | v26d (favor edge loss, $\lambda$=0.7 inverted) | 80.37% | — | -->
<!-- | v26e 8ch base | **80.47%** | — | -->
<!-- | v26e 12ch (+ kernel density channels) | 80.42% | — | -->
<!-- | v26f (learnable smoothing) | 79.37% | — | -->
<!-- | v26e 8ch + XY aug | **81.19%** | **80.96%** | -->
<!-- | v26e 12ch + XY aug | 80.42% | — | -->
<!---->
<!-- The best v26 result (81.19%) is comparable to v11+ (81.14%). More importantly, the two models are making different predictions — v11 reads 1D functional curves while v26 reads 2D joint distributions. This suggests that ensembling the two representations could yield further gains. -->
<!---->
<!-- --- -->
<!---->
<!-- ## v11 vs. v26: Different Models, Complementary Signal -->
<!---->
<!-- The two representations are fundamentally different in what they encode. -->
<!---->
<!-- v11 (1D sorted curves) is sensitive to the *functional shape* of $E[K \mid X]$ — the nonlinear trend, curvature, and heteroscedasticity of the conditional mean. It is optimal for detecting asymmetries in the functional relationship between variables. -->
<!---->
<!-- v26 (2D density) is sensitive to the *joint distributional shape* of $(K, X)$ — the spread, correlation structure, and tail behavior of the joint distribution. It is more natural for detecting whether two variables are marginally independent (uniform density) vs. dependent (structured density). -->
<!---->
<!-- Collider classification illustrates this difference. A Collider ($X \to K \leftarrow Y$) is marginally independent of both $X$ and $Y$, but conditioning on $K$ induces dependence. The 1D curve for a Collider looks like noise — no functional trend. The 2D density for a Collider also looks close to uniform marginally, but the residual channels (ch4–7) will show the collider bias pattern when $K$ is conditioned on. Each representation has a different angle on the same underlying structure. -->
<!---->
<!-- --- -->
<!---->
<!-- ## Summary -->
<!---->
<!-- The v26 contribution is framing 2D scatter density as a visual conditional independence test: the joint density image of $(K, X)$ is a nonparametric representation of their relationship, and Conv2D learns to classify causal roles by recognizing density patterns that correspond to d-separation structures in the graph. -->
<!---->
<!-- The key architectural lesson is that the node image alone is insufficient — it requires the edge pipeline as a graph context encoder, fused via a dual-pipeline architecture with an auxiliary edge classification loss. -->
<!---->
<!-- The main caveats are: (1) Conv2D's translation equivariance assumption does not hold for density images where position carries absolute meaning; (2) kernel regression density channels do not transfer from 1D to 2D due to axis mismatch; (3) learnable smoothing fails because it conflates denoising with feature learning. -->
<!---->
<!-- Both v11 and v26 reach comparable performance (~81%) but capture different aspects of causal structure, making them strong candidates for ensembling. -->

> This writeup describes the solution I developed for the [ADIA Lab Causal Discovery Challenge](https://hub.crunchdao.com/competitions/causality-discovery) as part of my BSc thesis at VNU-UET (2026). The solution achieves **81.14% balanced accuracy** locally and **80.82% on the CrunchDAO public leaderboard**, surpassing the original competition top-1 result of 76.70% by +4.12%. A second post covers the Conv2D / 2D scatter density extension (v26).

---

## Disclaimer

I had prior knowledge of the competition through the ADIA Lab survey paper, which describes the data-generating process. I did **not** use this to generate synthetic training data — that would invalidate the scientific contribution.

My solution is built directly on top of the [top-1 competition writeup](https://thetourney.github.io/adia-report/) as a baseline. I also drew EDA insights and the XY augmentation idea from the [3rd-place writeup](https://stream-physician-14c.notion.site/ADIA-Lab-Causal-Discovery-Challenge-Rank3-Solution-1397f010c9428099aa82e4503cad1c20). The novelty claims in this post are incremental improvements on that foundation — I'd rather be honest about that than oversell it.

This blog post is written by Claude with me as a proofreader and editor. The technical content is my work, but the writing style is a collaboration.

All source code is available in this [repository](https://github.com/aqu4holic/Graduation-Thesis-BsC-2026/). All experiment results are in this [Google Sheet](https://docs.google.com/spreadsheets/d/13IvbOsEDasD2m-yo4f6JgL_T6YuEoQXT5RCv_9JcHBE/edit?usp=sharing).

---

## Problem Setup

Given $n = 1000$ observations of $p \in \{3, \ldots, 10\}$ variables, where a causal edge $X \to Y$ is always known to exist, classify every other variable $K$ into one of **8 causal roles**:

| Role | Graph motif |
|---|---|
| Confounder | $X \leftarrow K \rightarrow Y$ |
| Mediator | $X \rightarrow K \rightarrow Y$ |
| Collider | $X \rightarrow K \leftarrow Y$ |
| Cause of X | $K \rightarrow X$, no edge to $Y$ |
| Cause of Y | $K \rightarrow Y$, no edge to $X$ |
| Consequence of X | $X \rightarrow K$, no edge to $Y$ |
| Consequence of Y | $Y \rightarrow K$, no edge to $X$ |
| Independent | No edge to $X$ or $Y$ |

![8 labels](/assets/img/posts/2026-04-15/causal_labels.png)
_The 8 causal role classes (image from the [ADIA Lab paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6125566))_

**Metric:** balanced accuracy — mean per-class recall. Overall accuracy is never reported. The hardest classes are Collider and Mediator; they count equally with the easy ones.

---

## Baseline: The Top-1 Solution

### Input Representation

The top-1 solution's core insight is representing each variable pair as a **sorted observation curve** rather than a set of scalar statistics.

For each directed pair $(u, v)$ where $u \neq v$, sort all 1000 observations by $u$'s values to get a permutation index `sort_idx`. Reading every other variable at that permutation gives a length-1000 functional curve of how $v$ behaves as $u$ increases — the conditional curve $v \mid u$. Any scalar summary permanently discards the shape of this curve. Conv1D reads it directly.

The baseline builds **3 channels per directed edge $(u, v)$**, all at the sorted permutation:

| Channel | What it is |
|---|---|
| $\mathbf{c}_1$ | $u$ sorted by $u$ — the sorted values of the source variable |
| $\mathbf{c}_2$ | $v$ sorted by $u$ — the conditional curve of $v$ given $u$ |
| $\mathbf{c}_3$ | Multivariate kernel regression coefficient $\hat{\beta}^{(h=0.5)}_{u \to v}$ at the sorted permutation — predicts $v$ from all other variables simultaneously |

For a graph with $p$ variables this produces $E = p(p-1)$ directed edges, each a tensor of shape $(3, 1000)$.

Each edge also has a **type label** (7 types, encoding that edge's structural relationship to the anchor $X \to Y$, e.g. $X \to v$, $v \to Y$, other $\to$ other, etc.).

### Pipeline

**Step 1 — Conv1D encoder:** Each $(3, 1000)$ edge tensor goes through a Stem layer (linear projection from 3 to $d$ channels) followed by 5 residual Conv1D blocks, then AdaptiveAvgPool1d. Output: one $d$-dim embedding per edge.

**Step 2 — Edge type fusion:** The 7 edge types are projected to $d$-dim vectors via a learned embedding table. These are merged with the Conv1D output via a linear fusion layer. Output: one $d$-dim embedding per edge encoding both content and structural type.

**Step 3 — Self-attention:** Two self-attention layers run over all $E$ edge embeddings simultaneously. Each edge can attend to all other edges, enabling global graph reasoning. Output: $E$ attended edge embeddings.

**Step 4 — Two heads:**
- **Edge head** (`Linear(d, 2)`): predicts whether a directed causal edge $u \to v$ actually exists in the ground truth DAG — binary classification, trained with cross-entropy against the adjacency matrix. This is an **auxiliary loss**, discarded at inference.
- **Node head** (`Linear(d, 8)`): for each non-$X$/$Y$ node $v$, gathers the 4 attended edge embeddings at positions $(v \to X), (v \to Y), (X \to v), (Y \to v)$, merges them via a linear fusion, and produces the 8-class causal role prediction.

The combined training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{node}} + \lambda \cdot \mathcal{L}_{\text{edge, binary}}$$

The edge head's binary objective (edge exists or not) is well-defined and has clean ground truth from the adjacency matrix. It forces the Conv1D + attention stack to produce embeddings that are sensitive to actual causal structure, ensuring the node head receives informative input rather than unconstrained representations.

**My reimplementation:** 73.96% local / 73.6% leaderboard (vs. claimed 76.70%). The gap is likely due to implementation details not fully specified in the writeup.

![Baseline architecture](https://thetourney.github.io/adia-report/assets/img/model.png)
_The baseline architecture (image from the [top-1 writeup](https://thetourney.github.io/adia-report/))_

---

## Contribution 1: Multi-Bandwidth Kernel Channels

The baseline uses a single bandwidth $h=0.5$ for the kernel regression coefficient. A single bandwidth creates a bias-variance tradeoff — coarse $h$ smooths away local nonlinear structure, fine $h$ is noisy.

I add coefficients at two additional bandwidths, giving the model simultaneous access to local and global conditional structure. The total channel count grows from 3 to 5:

$$[\text{sort}_u(u),\ \text{sort}_u(v),\ \hat{\beta}^{(h=0.2)},\ \hat{\beta}^{(h=0.5)},\ \hat{\beta}^{(h=1.0)}]$$

All channels are at the same sorted permutation, so they are spatially aligned sequences that Conv1D can jointly process.

**Result:** 73.96% → 75.44% (+1.48%)

---

## Contribution 2: ANM Residual Channels

The **Additive Noise Model** [Hoyer et al., 2009] gives a testable prediction: if $u$ truly causes $v$, then the residuals of regressing $v$ on $u$ and other variables should be statistically independent of $u$. The reverse direction residual will not be independent — it will retain structure. This asymmetry is a direct signal of causal direction.

I compute multivariate kernel regression residuals for each directed edge $(u, v)$ at 3 bandwidths and add them as 3 additional channels at the sorted permutation:

$$[\epsilon_{u \to v}^{(\sigma=1.0)},\ \epsilon_{u \to v}^{(\sigma=2.0)},\ \epsilon_{u \to v}^{(\sigma=4.0)}]$$

This brings the total to **8 channels: 2 raw sorted values + 3 kernel coefficients + 3 ANM residuals**, all length-1000, all sorted by $u$.

**Why this helps specific classes:** A Collider ($X \to K \leftarrow Y$) and a Consequence of X ($X \to K$) both have $X$ causing $K$. For the correct direction, regressing $K$ on $X$ leaves residuals that are near-independent of $X$ — a flat residual curve when sorted by $X$. The reverse direction residual retains structure. The model learns to read this asymmetry across the 8 residual channels. Empirically, Collider and Consequence of X recall improved most.

**Result:** 75.44% → 76.94% (+1.50%). Combined with XY augmentation: **80.26% / 80.22% LB**.

![The input tensor](/assets/img/posts/2026-04-15/input_tensors.svg)
_The 8-channel edge representation_

---

## Contribution 3: XY Augmentation

*Idea from [top-3 writeup](https://stream-physician-14c.notion.site/ADIA-Lab-Causal-Discovery-Challenge-Rank3-Solution-1397f010c9428099aa82e4503cad1c20#1457f010c94280cfb541c60cc9f55b97).*

In the original setup, the model always sees each graph with fixed $(X, Y)$ labels. This risks learning positional shortcuts — "when I see this pattern, $X$ is usually at index 0" — rather than the underlying topology.

**XY augmentation** relabels graphs so the same underlying structure is presented from multiple anchor perspectives. Any pair of variables with a directed causal edge between them can be relabeled as the new $(X', Y')$ anchor, and all other variables' roles are recomputed accordingly. A graph with $p$ variables yields roughly 11 valid relabelings on average. This forces the model to learn the topological role of each variable relative to the anchor edge, not relative to fixed label positions.

The effective training set grows from 25K graphs to 263K augmented samples.

**Interaction with structural bias:** XY augmentation gave +3.32% on v8b but only +1.55% on v11. Both techniques encode the same information — that $X$ and $Y$ are topological anchors, not positional constants — but one does it through data diversity and the other through architecture. Once the structural bias is in place, augmentation has less new information to contribute. The partial redundancy is indirect evidence that the structural bias mechanism is functioning correctly.

![Label swapping augmentation](https://stream-physician-14c.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe73d89a4-7021-4a2c-ad81-0287d8dd6a2a%2F80d93c17-7c9f-4143-98c7-ebf6bd750adc%2Fd9a9b16f247d29f0aed71e804d58264a_720.png?table=block&id=1457f010-c942-80d0-9cc5-d5924505db6c&spaceId=e73d89a4-7021-4a2c-ad81-0287d8dd6a2a&width=2000&userId=&cache=v2)
_Demonstration of the augmentation process (image from [top-3 writeup](https://stream-physician-14c.notion.site/ADIA-Lab-Causal-Discovery-Challenge-Rank3-Solution-1397f010c9428099aa82e4503cad1c20))_

---

## Contribution 4: Structural Attention Bias

This is the only architectural change that produced a significant positive result.

Standard self-attention treats every pair of edge embeddings $(e_i, e_j)$ identically in computing attention scores. But in a causal graph, the topological relationship between two directed edges carries strong prior information. Two edges forming a chain $X \to K \to Y$ should interact differently from two completely unrelated edges.

I define **6 topological relationship types** between any pair of directed edges $(u_1 \to v_1)$ and $(u_2 \to v_2)$:

| Type | Condition |
|---|---|
| Reverse | $u_1 = v_2$ and $v_1 = u_2$ |
| Shared source (fork) | $u_1 = u_2$, not reverse |
| Shared target (collider) | $v_1 = v_2$, not reverse |
| Forward chain | $v_1 = u_2$, not reverse |
| Backward chain | $u_1 = v_2$, not reverse |
| Unrelated | none of the above |

A **learned scalar bias** $b_{\tau, h} \in \mathbb{R}$ per type $\tau$ per attention head $h$ is added to the attention logit before softmax:

$$A_{ij}^{(h)} = \frac{\mathbf{q}_i^{(h)} \cdot \mathbf{k}_j^{(h)} + b_{\tau(i,j), h}}{\sqrt{d_h}}$$

The biases are zero-initialized, so the model starts from standard attention and learns to prioritize topological relationships during training. **Total additional parameters: 24 scalars** (6 types × 4 heads).

**Why this works where other architectural changes failed:** Every other architectural addition I tried (cross-attention branches, dual-path transformers, node-centric pooling) degraded performance. Those changes added raw capacity without a problem-specific prior, leading to overfitting on 25K samples. The structural bias adds almost no capacity — it only learns *how much to weight* each topological relationship type, with the types themselves defined by causal graph theory. The inductive bias is entirely appropriate to the problem.

**Result:** v8b 76.94% → v11 **79.59%** (+2.65%). With XY augmentation: **81.14% local / 80.82% LB**.

![Structural bias](/assets/img/posts/2026-04-15/structural_bias.svg)
_The structural attention bias: a learned scalar per (type, head) pair is added to the attention logit_

---

## What Didn't Work

### ML fullstack (v13): 72.64%

I built a complete machine learning pipeline — 300+ engineered features (pairwise correlations, partial correlations, mutual information, outputs from PC, LiNGAM, NOTEARS, ANM), three gradient boosting models, a GNN refinement stage, and a stacking ensemble. Result: **72.64%** — below the simple Conv1D baseline at 73.96%.

**Why:** Scalar feature compression is irreversible. Once you summarize a 1000-point conditional expectation curve into a single number, you permanently discard its shape — curvature, heteroscedasticity, nonlinearity. That shape is exactly what distinguishes a Mediator from a Consequence of X. Conv1D reads the full curve; no amount of scalar engineering can recover what was thrown away. I also tried injecting these scalars as a parallel tower into the deep learning model — this also failed, because broadcasting a scalar across a length-1000 sequence gives it no structural relationship to the sequence positions.

### Architectural complexity (v3, v4, v6): regressed 1–2%

Cross-attention branches and dual-path transformers all degraded performance. Architectural complexity only helps when it encodes an appropriate inductive bias. General capacity additions overfit on 25K training samples.

### Node-centric attention (v9, v9b): lost ~1.5%

Compressing the $O(p^2)$ edge context to node-level summaries consistently underperformed. Full edge self-attention is load-bearing: global graph reasoning requires seeing all pairwise relationships simultaneously.

---

## Score Progression

| Version | Local BA | LB | Key change |
|---|---|---|---|
| Baseline reimplementation | 73.96% | 73.6% | — |
| + multi-bandwidth kernel (v5m) | 75.44% | 74.98% | +1.48% |
| + ANM residuals (v8b) | 76.94% | — | +1.50% |
| + XY augmentation (v8b+) | 80.26% | 80.22% | +3.32% |
| + structural bias, no aug (v11) | 79.59% | — | +2.65% over v8b |
| + XY augmentation (v11+) | **81.14%** | **80.82%** | best result |

---

## Summary

Four contributions, each with a clear mechanistic motivation:

1. **Multi-bandwidth kernel channels** — simultaneous access to local and global conditional structure
2. **ANM residual channels** — direct causal direction signal, most impactful for Collider and Consequence classes
3. **XY augmentation** — forces topology learning over position learning, +3.32% on v8b
4. **Structural attention bias** — encodes causal graph priors into attention with 24 parameters, +2.65% over v8b

The core lesson: *representational form matters more than model complexity*. Full functional curves contain information that no scalar feature set can recover. Architectural changes only help when they carry an appropriate inductive bias — general capacity additions hurt.

---

## References

- thetourney. *ADIA Lab Causal Discovery Challenge — 1st Place Solution*. 2024. [link](https://thetourney.github.io/adia-report/)
- mutian-hong. *ADIA Lab Causal Discovery Challenge — 3rd Place Solution*. 2024. [link](https://stream-physician-14c.notion.site/ADIA-Lab-Causal-Discovery-Challenge-Rank3-Solution-1397f010c9428099aa82e4503cad1c20)
- Hoyer, P. et al. *Nonlinear Causal Discovery with Additive Noise Models*. NeurIPS 2009.
- Pearl, J. *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2009.
- Olivetti, E. et al. *Can Machines Learn Causal Structure? Evidence from ADIA Lab's Causal Discovery Challenge*. SSRN 2025.
