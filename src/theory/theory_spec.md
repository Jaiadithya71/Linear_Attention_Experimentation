# Theoretical Specification: Error-Controlled Adaptive-Rank Linear Attention

**Team 3 — Mathematical Foundation & Error Certificate Framework**  
*Lead Author: Person 1 (Theory Lead)*  
*Target Architecture: Positive-Kernel Linear Attention with Dynamic Rank Allocation*

---

## 1. Mathematical Formulation & Notation

Let input sequences be represented as:
- Queries $Q \in \mathbb{R}^{N \times d_k}$
- Keys $K \in \mathbb{R}^{N \times d_k}$
- Values $V \in \mathbb{R}^{N \times d_v}$

where $N$ denotes the sequence length, $d_k$ is the query/key dimension, and $d_v$ is the value dimension.

### 1.1 Exact Softmax Attention
The standard scaled dot-product softmax attention matrix and output are given by:
$$S_{i, j} = \frac{\langle q_i, k_j \rangle}{\sqrt{d_k}}$$
$$A_{i, j} = \frac{\exp(S_{i, j})}{\sum_{m=1}^N \exp(S_{i, m})}$$
$$Y^S = A V \in \mathbb{R}^{N \times d_v}$$

Row $i$ of the output is:
$$y_i^S = \frac{\sum_{j=1}^N \exp(S_{i, j}) v_j}{\sum_{m=1}^N \exp(S_{i, m})}$$

### 1.2 Kernelized Linear Attention
Linear attention replaces the exponential interaction with an inner product of positive feature maps $\phi: \mathbb{R}^{d_k} \to \mathbb{R}_+^r$:
$$K_{\text{lin}}(q_i, k_j) = \langle \phi(q_i), \phi(k_j) \rangle$$

By exploiting associativity of matrix multiplication, the output is computed in $O(N r d_v)$ time:
$$y_i^L = \frac{\sum_{j=1}^N \langle \phi(q_i), \phi(k_j) \rangle v_j}{\sum_{m=1}^N \langle \phi(q_i), \phi(k_m) \rangle} = \frac{\phi(q_i)^T \sum_{j=1}^N \phi(k_j) v_j^T}{\phi(q_i)^T \sum_{m=1}^N \phi(k_m)}$$

In recurrent state form:
$$S_{\text{state}} = \sum_{j=1}^N \phi(k_j) v_j^T \in \mathbb{R}^{r \times d_v}, \quad z = \sum_{j=1}^N \phi(k_j) \in \mathbb{R}^r$$
$$y_i^L = \frac{\phi(q_i)^T S_{\text{state}}}{\phi(q_i)^T z}$$

---

## 2. Derivation of the Output Error Certificate Bound

Let $\hat{K}(q, k)$ be an $r$-dimensional positive kernel approximation to the unnormalized softmax kernel $K_{\text{softmax}}(q, k) = \exp\left(\frac{\langle q, k \rangle}{\sqrt{d_k}}\right)$.

Define the element-wise kernel approximation error as:
$$|\hat{K}(q_i, k_j) - K(q_i, k_j)| \le \epsilon_r(X) \quad \forall i, j \in [N]$$

Let the true partition function for token $i$ be:
$$Z_i = \sum_{j=1}^N K(q_i, k_j)$$
and the approximate partition function be:
$$\hat{Z}_i = \sum_{j=1}^N \hat{K}(q_i, k_j)$$
Then:
$$|\hat{Z}_i - Z_i| \le \sum_{j=1}^N |\hat{K}(q_i, k_j) - K(q_i, k_j)| \le N \epsilon_r(X)$$

### 2.1 Numerator Deviation
Let $N_i = \sum_{j=1}^N K(q_i, k_j) v_j$ and $\hat{N}_i = \sum_{j=1}^N \hat{K}(q_i, k_j) v_j$. Assuming $\|v_j\|_\infty \le V_{\max}$:
$$\|\hat{N}_i - N_i\|_\infty \le \sum_{j=1}^N |\hat{K}(q_i, k_j) - K(q_i, k_j)| \|v_j\|_\infty \le N \epsilon_r(X) V_{\max}$$

### 2.2 Output Error Upper Bound
The difference in attention output vector $y_i$ is:
$$\|y_i^L - y_i^S\|_\infty = \left\| \frac{\hat{N}_i}{\hat{Z}_i} - \frac{N_i}{Z_i} \right\|_\infty = \left\| \frac{\hat{N}_i Z_i - N_i \hat{Z}_i}{\hat{Z}_i Z_i} \right\|_\infty$$

Adding and subtracting $N_i Z_i$:
$$\hat{N}_i Z_i - N_i \hat{Z}_i = (\hat{N}_i - N_i) Z_i - N_i (\hat{Z}_i - Z_i)$$
Taking norms:
$$\|\hat{N}_i Z_i - N_i \hat{Z}_i\|_\infty \le \|\hat{N}_i - N_i\|_\infty Z_i + \|N_i\|_\infty |\hat{Z}_i - Z_i|$$
Since $\|N_i\|_\infty \le Z_i V_{\max}$:
$$\|\hat{N}_i Z_i - N_i \hat{Z}_i\|_\infty \le N \epsilon_r V_{\max} Z_i + Z_i V_{\max} (N \epsilon_r) = 2 N \epsilon_r V_{\max} Z_i$$

Dividing by $\hat{Z}_i Z_i$:
$$\|y_i^L - y_i^S\|_\infty \le \frac{2 N \epsilon_r V_{\max}}{\hat{Z}_i} \le \frac{2 N \epsilon_r V_{\max}}{Z_i - N \epsilon_r}$$

### 2.3 Sufficient Condition for Prescribed Tolerance $\delta$
To guarantee $\|y_i^L - y_i^S\|_\infty \le \delta$ for all tokens $i \in [N]$:
$$\frac{2 N \epsilon_r V_{\max}}{Z_i - N \epsilon_r} \le \delta \iff 2 N \epsilon_r V_{\max} \le \delta Z_i - \delta N \epsilon_r$$
$$\iff N \epsilon_r (2 V_{\max} + \delta) \le \delta Z_i$$
$$\implies \epsilon_r(X) \le \frac{\delta \min_i Z_i}{N (2 V_{\max} + \delta)}$$

This is the **Sufficient Output Error Certificate** used in `src/theory/difficulty_metric.py`.

---

## 3. Dynamic Rank Selection Algorithm $r^*(X, \delta)$

### 3.1 Input Difficulty Metric: Attention Entropy
The geometric difficulty of an attention distribution depends on its entropy:
$$\mathcal{H}(q_i, K) = -\sum_{j=1}^N A_{i, j} \ln A_{i, j}$$

- **High Entropy Regime ($\mathcal{H} \approx \ln N$):** Attention weights are diffuse. The partition function $Z_i$ is large, and no single key dominates. Positive random features converge quickly; a small rank ($r=64$) satisfies tolerance $\delta$.
- **Low Entropy / Needle Regime ($\mathcal{H} \ll \ln N$):** A single key dominates ($A_{i, j^*} \approx 1$). Approximating the sharp exponential peak requires high-order polynomial features; error decays slowly ($O(1/\sqrt{r})$), demanding $r \ge 1024$ or exact fallback.

### 3.2 Rank Selection Rule
Given candidate ranks $\mathcal{R} = (64, 256, 1024)$:
$$r^*(X, \delta) = \min \left\{ r \in \mathcal{R} : \hat{\epsilon}_r(X) \le \frac{\delta \min_i Z_i}{N (2 V_{\max} + \delta)} \right\}$$

where the empirical error estimate $\hat{\epsilon}_r(X)$ is computed via FAVOR+ variance bounds:
$$\hat{\epsilon}_r(X) \approx \frac{C_{\text{geom}} \exp\left(\frac{\|q\|_{\max} \|k\|_{\max}}{\sqrt{d_k}}\right)}{\sqrt{r}}$$
with $C_{\text{geom}} = 1 - \frac{\bar{\mathcal{H}}}{\ln N}$.

---

## 4. Connection to SOTA Literature

1. **Katharopoulos et al. (ICML 2020):** Proposed $\phi(x) = \text{elu}(x) + 1$, establishing $O(N)$ autoregressive inference but failing on sharp retrieval tasks due to state compression.
2. **Choromanski et al. (ICLR 2021) - Performer / FAVOR+:** Positive random features with orthogonal coupling eliminate variance while preserving unnormalized expectation $\mathbb{E}[\hat{K}] = K_{\text{softmax}}$.
3. **Likhosherstov et al. (2023) - FAVOR#:** Demonstrates that deterministic positive polynomial feature maps minimize maximum error bounds over compact spheres, providing tighter certificates than purely random features.
4. **Sui & Zhang (2026):** Proves that the exact softmax interaction rank scales geometrically with temperature $\tau = 1/\sqrt{d_k}$ and query-key norm radius $R$. For sequences with $R \le \sqrt{d_k}$, rank $r = O(d_k)$ suffices; for $R \gg \sqrt{d_k}$, required rank grows as $r = \Omega(N)$, explaining the empirical collapse observed at $N \ge 1,024$.
