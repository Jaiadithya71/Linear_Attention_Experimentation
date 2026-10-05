# Master Research Synthesis: Linear Attention vs. Softmax Attention

**Project:** Adaptive Attention Computation: Balancing Efficiency and Retrieval in Long-Context Transformers  
**Research Team:** Team 3 (6 Engineers & Researchers)  
**Hardware & Environment:** NVIDIA Tesla T4 (Dual T4 16GB, Kaggle / Colab), `torch.float32`, PyTorch 2.11.0+cu128  
**Document Purpose:** Complete, easily readable synthesis of all project artifacts, mathematical formulations, empirical results, and research roadmap.

---

## Table of Contents

1. [Executive Summary & High-Level Narrative](#1-executive-summary--high-level-narrative)
2. [Project Artifact Inventory & Map](#2-project-artifact-inventory--map)
3. [The Core Dilemma: Quadratic Softmax vs. Linear Attention](#3-the-core-dilemma-quadratic-softmax-vs-linear-attention)
4. [Mathematical Foundations & Corrections](#4-mathematical-foundations--corrections)
   - [The Corrected 3-Token Analytical Benchmark](#the-corrected-3-token-analytical-benchmark)
   - [Associativity: Why Softmax Cannot Be Reordered](#associativity-why-softmax-cannot-be-reordered)
   - [Error-Controlled Adaptive-Rank Theorem](#error-controlled-adaptive-rank-theorem)
5. [Empirical Benchmark Results (Tesla T4)](#5-empirical-benchmark-results-tesla-t4)
   - [Efficiency & Latency Scaling](#efficiency--latency-scaling)
   - [VRAM Consumption & OOM Regimes](#vram-consumption--oom-regimes)
   - [The Retrieval Accuracy Collapse](#the-retrieval-accuracy-collapse)
   - [Output Deviation & Logit Scale Sensitivity](#output-deviation--logit-scale-sensitivity)
   - [Dimension Sweeps: When Does Linear Attention Win?](#dimension-sweeps-when-does-linear-attention-win)
6. [Can Linear Attention Be Rescued? (FAVOR+ & Sharpness Sweeps)](#6-can-linear-attention-be-rescued-favor--sharpness-sweeps)
7. [The Formal Pre-Registered Acceptance Test](#7-the-formal-pre-registered-acceptance-test)
8. [Root Cause Analysis: Why Kernelized Linear Attention Collapses](#8-root-cause-analysis-why-kernelized-linear-attention-collapses)
9. [The Architectural Pivot: Adaptive Attention & Next Steps](#9-the-architectural-pivot-adaptive-attention--next-steps)

---

## 1. Executive Summary & High-Level Narrative

The project investigates whether **Kernelized Linear Attention** can replace standard **Softmax Attention** to eliminate the quadratic $O(N^2)$ computational and memory bottleneck in long-context Transformer models, while preserving attention quality and key-value retrieval capability.

```
                    ┌──────────────────────────────────────────────┐
                    │ Standard Softmax Attention                   │
                    │ Complexity: O(N² d) time, O(N²) memory       │
                    │ High retrieval accuracy (100%), but OOMs     │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Kernel Trick & Associativity
                        (QKᵀ)V  vs  φ(Q)(φ(K)ᵀV)
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Kernelized Linear Attention                  │
                    │ Complexity: O(N r d) time, O(r d) memory     │
                    │ 164x speedup at N=65,536 (9.7 ms vs 1602 ms) │
                    └──────────────────────┬───────────────────────┘
                                           │
                                  Empirical Findings:
                         Severe Retrieval Collapse (14% acc)
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ The Research Pivot: Adaptive Attention       │
                    │ Error-controlled dynamic rank r(X, δ)        │
                    │ Hybrid routing & Gated DeltaNet baselines    │
                    └──────────────────────────────────────────────┘
```

### The Three Major Findings:
1. **Computational Scaling Claim Confirmed:**  
   Reordered linear attention ($\text{ReLU}+1$) achieves true $O(N)$ execution scaling (empirical time exponent $\alpha \approx 0.80$ vs $1.81$ for softmax). At sequence length $N = 65,536$, Linear Attention executes in **$9.74\text{ ms}$** compared to **$1,601.74\text{ ms}$** for exact PyTorch SDPA (a **$164.4\times$ speedup**). Naive softmax crashes with Out-of-Memory (OOM) at $N = 32,768$, whereas linear attention consumes only **$257.1\text{ MB}$** of extra peak memory at $N = 65,536$.
2. **Quality & Retrieval Capability Collapsed:**  
   In a standardized 16-class synthetic Key-Value Retrieval benchmark (chance level $6.2\%$), Softmax achieves **$100\%$ accuracy** across all sequence lengths. Linear Attention collapses to **$14.2\%$ accuracy** at $N = 1,024$ and **$9.5\%$** at $N = 4,096$. Even increasing feature rank to $r = 1,024$ ($16\times$ feature capacity via FAVOR+) only rescues retrieval to **$26.7\%$**.
3. **The Acceptance Criterion Failed:**  
   Under a pre-registered non-inferiority margin ($\delta = 0.05$) and required speedup ($\ge 2.0\times$), Linear Attention failed the combined test due to an accuracy deficit $\Delta = -0.86\text{ to } -0.91$. Furthermore, modern memory-efficient exact attention (PyTorch SDPA / FlashAttention) runs up to $N = 65,536$ with $O(N)$ memory and zero retrieval loss, establishing itself as the mandatory engineering control.

---

## 2. Project Artifact Inventory & Map

The project directory consists of seven original artifacts spanning the research lifecycle, now converted into structured Markdown files and figures:

| Original File | Formatted Markdown in `docs_markdown/` | Role & Description |
| :--- | :--- | :--- |
| `DOC-20261005-WA0006.pdf` | [`01_project_plan_and_specification.md`](01_project_plan_and_specification.md) | **Sprint Plan & Tech Spec:** 6-engineer task breakdown, contract-first interfaces, 9.5-hour execution schedule, Kaggle T4 guardrails. |
| `Linear_Attention_Research_Team_3.docx` | [`02_research_proposal_initial.md`](02_research_proposal_initial.md) | **Initial Proposal:** Initial research question, null/alternative hypotheses, toy 3-token calculation (contains the uncorrected math). |
| `Linear_Attention_Research_Team_3_Revised.docx` | [`03_research_proposal_revised.md`](03_research_proposal_revised.md) | **Revised Proposal:** Pre-registered design, mathematical corrections ($1/\sqrt{d_k}$, associativity limits, true baseline comparison), non-inferiority test setup. |
| `Mathematically_Correct_Linear_Attention_Hypothesis (2).docx` | [`04_mathematical_hypothesis_and_framework.md`](04_mathematical_hypothesis_and_framework.md) | **Mathematical Framework:** Error-controlled adaptive rank derivation, uniform kernel error theorem, sufficient condition $\epsilon_r(X)$, references to FAVOR# and Sui & Zhang (2026). |
| `Team_3_Linear_Attention_PDCA_EN.docx` | [`05_team_3_pdca_framework.md`](05_team_3_pdca_framework.md) | **PDCA Research Workbook:** Plan-Do-Check-Act framework, 3 implementation paths ($S, K, L$), causal leakage guardrails, acceptance checklist. |
| `Team3_Linear_Attention_Executive_Summary.pptx` | [`06_team_3_presentation_slides.md`](06_team_3_presentation_slides.md) | **Executive Presentation Deck:** 7-slide departmental slide deck summarizing methodology, benchmarks, failure modes, and next steps. |
| `Untitled0.ipynb` | [`07_experiments_and_notebook_code.md`](07_experiments_and_notebook_code.md) | **Full Benchmark Code & Logs:** Complete PyTorch implementation, CUDA timing harness, memory tracking, retrieval suite, FAVOR+ sweeps, and generated figures. |
| — | [`08_two_person_execution_plan.md`](08_two_person_execution_plan.md) | **Two-Person Sprint Plan:** Restructured execution plan consolidating the 6 original roles into a 2-person research & testing pair. |

---

## 3. The Core Dilemma: Quadratic Softmax vs. Linear Attention

### Standard Softmax Attention
For queries $Q \in \mathbb{R}^{N \times d_k}$, keys $K \in \mathbb{R}^{N \times d_k}$, and values $V \in \mathbb{R}^{N \times d_v}$:
$$A^S = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$$
$$Y^S = A^S V \in \mathbb{R}^{N \times d_v}$$
- **Time Complexity:** $O(N^2 d_k + N^2 d_v)$
- **Intermediate Memory:** $O(N^2)$ to store the attention matrix $A^S$ (or $O(N)$ when using tiled online-softmax in FlashAttention/SDPA).

### Kernelized Linear Attention (Katharopoulos et al., 2020)
By replacing the softmax normalization with a non-negative feature map $\phi(x): \mathbb{R}^{d_k} \to \mathbb{R}^r$:
$$\kappa(q_i, k_j) = \phi(q_i)^T \phi(k_j)$$
The full attention output for row $i$ becomes:
$$y_i^L = \frac{\sum_{j=1}^N \phi(q_i)^T \phi(k_j) v_j}{\sum_{j=1}^N \phi(q_i)^T \phi(k_j)}$$
Using matrix associativity $(\Phi_Q \Phi_K^T) V = \Phi_Q (\Phi_K^T V)$:
$$Y^L = \tilde{D}^{-1} \Phi_Q \left(\Phi_K^T V\right)$$
where $\tilde{D} = \text{diag}\left(\Phi_Q (\Phi_K^T \mathbf{1}_N)\right)$.

- **Intermediate State:** $S_V = \Phi_K^T V \in \mathbb{R}^{r \times d_v}$ and $z = \Phi_K^T \mathbf{1}_N \in \mathbb{R}^{r \times 1}$.
- **Time Complexity:** $O(N r d_v + N r d_k)$
- **Memory Complexity:** $O(r d_v)$ recurrent state.

---

## 4. Mathematical Foundations & Corrections

### The Corrected 3-Token Analytical Benchmark
A major milestone of Team 3 was catching and correcting the algebraic error in the initial draft.

Given toy inputs ($d_k = d_v = 2$, $N = 3$):
$$Q = \begin{bmatrix} 1 & 2 \\ 2 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 10 & 0 \\ 0 & 10 \\ 5 & 5 \end{bmatrix}$$

1. **The Correction:**
   - **Correct $Q K^T$:**
     $$Q K^T = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 1 & 3 \\ 1 & 1 & 2 \end{bmatrix}$$
   - *Error caught:* The initial proposal reported $[[7, 8, 10], [8, 7, 10], [6, 6, 8]]$ as $Q K^T$. That matrix was actually $\phi(Q)\phi(K)^T$ using $\phi(x) = \text{ReLU}(x) + 1$.
2. **Correct Scaled Softmax Output (Row 1):**
   - Scaled scores: $Q K^T / \sqrt{2} = [0.70711, 1.41421, 2.12132]$
   - Softmax weights: $[0.14003, 0.28400, 0.57598]$
   - Output: $y_1^S = 0.14003(10,0) + 0.28400(0,10) + 0.57598(5,5) = \mathbf{[4.28017, 5.71983]}$
3. **Linear Attention Output (Row 1):**
   - Feature map $\phi(x) = \text{ReLU}(x) + 1$:
     $$\Phi_K^T V = \begin{bmatrix} 30 & 20 \\ 20 & 30 \end{bmatrix}, \quad \Phi_K^T \mathbf{1} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$$
   - Numerator: $\phi(q_1) (\Phi_K^T V) = (2, 3) \begin{bmatrix} 30 & 20 \\ 20 & 30 \end{bmatrix} = [120, 130]$
   - Denominator: $(2, 3) \cdot [5, 5]^T = 25$
   - Output: $y_1^L = [120/25, 130/25] = \mathbf{[4.8, 5.2]}$
4. **Discrepancy:**
   The relative Frobenius error between Linear Attention and true Scaled Softmax on this toy example is **$8.4\%$**. Linear attention smooths weights out ($(0.28, 0.32, 0.40)$) compared to Softmax's sharp exponential focus ($0.576$ on token 3).

### Associativity: Why Softmax Cannot Be Reordered
The fundamental reason linear attention requires replacing softmax with a kernel feature map is algebraic:
$$\text{softmax}(Q K^T) V \ne \text{reorderable}$$
The row-wise denominator $\sum_j \exp(q_i^T k_j / \sqrt{d_k})$ couples every query $q_i$ nonlinearly with every key $k_j$, making factoring impossible. Kernelization decouples the inner product:
$$\exp\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right) \approx \phi(q_i)^T \phi(k_j)$$
allowing associativity:
$$\sum_j \left(\phi(q_i)^T \phi(k_j)\right) v_j = \phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)$$

### Error-Controlled Adaptive-Rank Theorem
From [`04_mathematical_hypothesis_and_framework.md`](04_mathematical_hypothesis_and_framework.md):
- Let $\kappa_{ij} = \exp(q_i^T k_j / \sqrt{d_k})$ and $\tilde{\kappa}_{ij} = \phi_r(q_i)^T \phi_r(k_j)$.
- Suppose $|\tilde{\kappa}_{ij} - \kappa_{ij}| \le \epsilon_r$ uniformly for all $i, j$, with $\|v_j\| \le V_{\max}$ and row normalizer $Z_i = \sum_j \kappa_{ij}$.
- The output error satisfies:
  $$\|y_i^L - y_i^S\| \le \frac{2 N \epsilon_r V_{\max}}{Z_i - N \epsilon_r}$$
- **Sufficient Condition:** For a prescribed tolerance $\|y_i^L - y_i^S\| \le \delta$, the feature approximation error must satisfy:
  $$\epsilon_r \le \frac{\delta Z_i}{N(2 V_{\max} + \delta)}$$
- **Adaptive-Rank Rule:**
  $$r^*(X, \delta) = \min \left\{ r \in \mathcal{R} : \epsilon_r(X) \le \frac{\delta \min_i Z_i}{N(2 V_{\max} + \delta)} \right\}$$

---

## 5. Empirical Benchmark Results (Tesla T4)

All tests executed under identical conditions: NVIDIA Tesla T4, `torch.float32`, $B=1, H=4, d_k=d_v=r=64$.

### Efficiency & Latency Scaling

![Efficiency Scaling Curves](figures/fig1_efficiency_scaling_curves.png)

| Sequence Length ($N$) | Naive Softmax (ms) | Flash / SDPA Exact (ms) | Sparse (Local $w=64$) (ms) | Linear ($\text{ReLU}+1$) (ms) | Speedup vs Naive Softmax | Speedup vs Flash / SDPA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 0.31 | 0.22 | 1.03 | 0.57 | 0.54× | 0.39× |
| **256** | 0.27 | 0.24 | 0.88 | 0.44 | 0.62× | 0.55× |
| **512** | 0.38 | 0.40 | 0.94 | 0.48 | 0.80× | 0.84× |
| **1,024** | 1.11 | 1.00 | 0.69 | **0.39** | **2.86×** | **2.57×** |
| **2,048** | 3.28 | 3.63 | 0.92 | **0.54** | **6.06×** | **6.72×** |
| **4,096** | 11.19 | 9.81 | 1.38 | **0.86** | **12.97×** | **11.36×** |
| **8,192** | 37.38 | 29.03 | 2.29 | **1.38** | **27.00×** | **20.97×** |
| **16,384** | 175.61 | 113.95 | 5.32 | **4.01** | **43.77×** | **28.40×** |
| **32,768** | **OOM** | 369.45 | 8.50 | **4.91** | — | **75.32×** |
| **65,536** | **OOM** | 1,601.74 | 16.86 | **9.74** | — | **164.43×** |

- **Crossover Point:** Reordered Linear Attention becomes faster than Naive Softmax and Flash/SDPA at **$N \approx 1,024$**.
- **Empirical Scaling Exponent (Time Slope):**
  - Naive Softmax: **$1.81$**
  - Flash / SDPA: **$1.74$**
  - Sparse ($w=64$): **$0.79$**
  - Linear Attention: **$0.80$** (sub-linear to linear in practice due to GPU core saturation)

### VRAM Consumption & OOM Regimes

| Sequence Length ($N$) | Naive Softmax Peak (MB) | Flash / SDPA Peak (MB) | Sparse Peak (MB) | Linear Peak (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 0.2 | 0.1 | 0.8 | 0.3 |
| **512** | 8.5 | 0.5 | 6.5 | 2.1 |
| **1,024** | 33.0 | 1.0 | 13.0 | 4.1 |
| **4,096** | 516.0 | 4.0 | 52.0 | 16.1 |
| **16,384** | 8,208.0 | 16.0 | 208.1 | 64.3 |
| **32,768** | **OOM** ($> 16\text{ GB}$) | 32.0 | 416.1 | 128.6 |
| **65,536** | **OOM** | 64.0 | 832.3 | **257.1** |

- Naive Softmax memory explodes with slope **$1.99$** ($O(N^2)$).
- Flash / SDPA and Linear Attention maintain a memory scaling slope of **$1.0$** ($O(N)$ total memory including input/output tensors).
- Linear Attention's internal recurrent state is strictly constant: $S_V \in \mathbb{R}^{H \times r \times d_v} = 4 \times 64 \times 64 = 16,384$ floats per batch element ($64\text{ KB}$).

### The Retrieval Accuracy Collapse

![Key-Value Retrieval Accuracy Curves](figures/fig2_retrieval_accuracy_curves.png)

Synthetic benchmark: $N$ keys with 16 balanced orthonormal classes ($C = 16$, chance = $6.25\%$). Query is a noisy target key; all other $N-1$ keys are distractors.

| Sequence Length ($N$) | Naive Softmax | Flash / SDPA Exact | Sparse (Global Target) | Sparse (Local Target) | Linear ($\text{ReLU}+1$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 100.0% | 100.0% | 100.0% | 100.0% | **57.7%** |
| **256** | 100.0% | 100.0% | 64.3% | 100.0% | **27.2%** |
| **512** | 100.0% | 100.0% | 39.2% | 100.0% | **17.4%** |
| **1,024** | 100.0% | 100.0% | 23.7% | 100.0% | **14.2%** |
| **2,048** | 100.0% | 100.0% | 15.1% | 100.0% | **11.6%** |
| **4,096** | 100.0% | 100.0% | 10.7% | 100.0% | **9.5%** |

- Softmax maintains **$100.0\%$ accuracy** at all sequence lengths.
- Linear Attention drops to **$14.2\%$ at $N=1,024$** and approaches chance level (**$9.5\%$ vs $6.25\%$**) at $N=4,096$.
- Sparse Attention maintains **$100\%$** if the needle is within the local window ($w=64$), but drops rapidly for global targets.

### Output Deviation & Logit Scale Sensitivity

Relative Frobenius error $\|Y^L - Y^S\|_F / \|Y^S\|_F$:

| Scale Factor | $N = 64$ | $N = 256$ | $N = 1,024$ | $N = 4,096$ |
| :---: | :---: | :---: | :---: | :---: |
| **Scale 0.5 (Diffused)** | 0.2466 | 0.2359 | 0.2443 | 0.2388 |
| **Scale 1.0 (Standard)** | 0.7733 | 0.7748 | **0.7903** | 0.7867 |
| **Scale 2.0 (Sharp)** | 0.9797 | 0.9928 | **0.9976** | 0.9992 |

As logits become sharper ($\text{scale} \ge 1.0$), Softmax places almost all probability mass on the single best matching key, while the linear kernel ($\text{ReLU}+1$) cannot produce peaky weights, leading to near-total relative error ($> 99\%$).

### Dimension Sweeps: When Does Linear Attention Win?

At fixed $N = 4,096$, sweeping head dimension $d \in \{16, 32, 64, 128, 256, 512\}$:
- Softmax complexity: $O(N^2 d)$
- Linear complexity: $O(N d^2)$

| Head Dim $d$ | Softmax Latency (ms) | Linear Latency (ms) | Speedup vs Softmax |
| :---: | :---: | :---: | :---: |
| **16** | 11.37 | 0.76 | **15.04×** |
| **32** | 9.17 | 0.71 | **13.01×** |
| **64** | 10.14 | 0.82 | **12.31×** |
| **128** | 15.32 | 1.58 | **9.69×** |
| **256** | 31.40 | 3.14 | **10.00×** |
| **512** | 59.42 | 7.36 | **8.07×** |

Linear attention's speedup degrades as $d$ increases, because the recurrent state update scales quadratically with head dimension ($d^2$). Linear attention dominates only when $N \gg d$.

---

## 6. Can Linear Attention Be Rescued? (FAVOR+ & Sharpness Sweeps)

### 1. FAVOR+ Random Feature Dimension Sweep ($N = 1,024$)
Tested Choromanski et al.'s Positive Random Features (FAVOR+) with varying random feature dimension $r$:

| Attention Variant | Feature State Size ($r \times d_v$) | Key-Value Retrieval Accuracy |
| :--- | :---: | :---: |
| Linear FAVOR+ ($r = 64$) | 4,096 floats | 10.1% |
| Linear FAVOR+ ($r = 256$) | 16,384 floats | 15.6% |
| Linear FAVOR+ ($r = 1024$) | 65,536 floats | **26.7%** |
| Linear $\text{ReLU}+1$ ($r = 64$) | 4,096 floats | 14.2% |
| **Softmax (Exact)** | 131,072 floats | **100.0%** |

*Verdict:* Quadrupling feature rank ($r = 64 \to 256 \to 1024$) yields diminishing returns ($10.1\% \to 26.7\%$). To reach Softmax-level retrieval, the rank $r$ would have to approach $N$, completely negating the $O(N)$ efficiency advantage.

### 2. Query/Key Signal-Strength Sweep ($N = 1,024$)
Varying key scale from 2 to 24:
- **Softmax:** Accuracy rises from $32.4\%$ (scale 2) to **$100.0\%$** (scale 8 and above).
- **Linear Attention:** Accuracy starts at $8.3\%$ (scale 2) and peaks at only **$17.5\%$** (scale 24). Linear attention cannot leverage large logit scales because positive polynomial/exponential kernels without softmax exponentiation sum diffusely over distractors.

---

## 7. The Formal Pre-Registered Acceptance Test

![Efficiency-Quality Trade-Off and Non-Inferiority Analysis](figures/fig3_tradeoff_and_noninferiority.png)

### Acceptance Rules:
1. **Speed Criterion:** Latency speedup $\ge 2.0\times$ vs naive softmax over the target sequence range.
2. **Quality Criterion (Non-Inferiority):** Paired retrieval accuracy difference $\Delta = \text{acc}_{\text{method}} - \text{acc}_{\text{softmax}}$ must satisfy:
   $$95\% \text{ Bootstrap CI Lower Bound} > -\delta \quad (\delta = 0.05)$$
3. **Combined Claim:** Both criteria must pass simultaneously.

### Summary at $N = 4,096$:

| Method | Speedup vs Naive | Speed Status | Quality Difference ($\Delta$) | 95% CI | Non-Inferior ($\delta = 0.05$)? | Combined Claim Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear ($\text{ReLU}+1$)** | **$12.97\times$** | **PASS** | **$-0.905$** | $[-0.909, -0.900]$ | **FAIL** | **FAILS** |
| **Sparse ($w=64$, Global)** | **$8.14\times$** | **PASS** | **$-0.893$** | $[-0.895, -0.891]$ | **FAIL** | **FAILS** |
| **Sparse ($w=64$, Local)** | **$8.14\times$** | **PASS** | **$0.000$** | $[0.000, 0.000]$ | **PASS** | **PASSES (Local only)** |
| **Flash / SDPA** | $1.14\times$ | Fail ($\ge 2\times$) | **$0.000$** | $[0.000, 0.000]$ | **PASS** | **Passes Quality Only** |

**Conclusion:** Pure kernelized linear attention fails the acceptance test.

---

## 8. Root Cause Analysis: Why Kernelized Linear Attention Collapses

The empirical breakdown of kernelized linear attention is rooted in **three fundamental architectural failure modes**:

1. **The Fixed-Capacity Memory Bottleneck:**  
   Softmax attention retains access to all $N$ individual key-value pairs. In contrast, linear attention compresses all $N$ tokens into an $r \times d_v$ matrix $S = \sum_{j=1}^N \phi(k_j) v_j^T$. As $N \to \infty$, an unbounded number of tokens are packed into fixed dimension $r$, causing massive key collisions and semantic crosstalk.
2. **Loss of Selective Sharpening (The Softmax Margin):**  
   Softmax applies an exponential non-linearity *after* the inner product $\exp(q_i^T k_j / \tau)$. This creates an extreme winner-take-all filtering effect where distractors with small logit margins are suppressed to near-zero. Kernelized linear attention computes $\phi(q_i)^T \phi(k_j)$; since $\phi(\cdot) \ge 0$, distractors always contribute positive mass to the accumulator, drowning out the target signal as $N$ grows.
3. **Permutation Equivariance & Long-Range Smearing:**  
   In non-causal attention without decay or gating, all tokens are summed with equal weight regardless of position. Distant noise tokens accumulate additively, degrading the signal-to-noise ratio linearly with sequence length.

---

## 9. The Architectural Pivot: Adaptive Attention & Next Steps

Faced with these empirical results, the team initiated a strategic pivot outlined in [`01_project_plan_and_specification.md`](01_project_plan_and_specification.md) and [`04_mathematical_hypothesis_and_framework.md`](04_mathematical_hypothesis_and_framework.md):

```
                        ┌───────────────────────────────┐
                        │      Input Query & Context    │
                        └──────────────┬────────────────┘
                                       │
                         Difficulty / Margin Estimator
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
        Low Complexity Context                    High Complexity Context
        (Diffused / Summarization)                (Precise Retrieval / Needles)
                  │                                         │
                  ▼                                         ▼
        Linear Kernel Path                        Exact / High-Rank Path
        r = 64 (O(N) Fast Path)                   r = 1024 or FlashAttention SDPA
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       ▼
                         Output Feature Representation
```

### The 5-Phase Roadmap:
1. **SOTA Recurrent Baselines (Gated DeltaNet):**  
   Implement recurrent linear architectures with data-dependent update gates and delta-rule state corrections ($S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$), which solve associative recall while preserving $O(N)$ inference.
2. **Adaptive-Rank Kernel (`AdaptiveRankAttention`):**  
   Dynamically assign feature rank $r \in \{64, 256, 1024\}$ per attention head or token block based on the mathematical error certificate $\epsilon_r(X)$.
3. **Hybrid Sparse-Linear Routing:**  
   Combine local window attention ($w = 64$, which proved $100\%$ accurate locally) with an error-controlled linear global summary state.
4. **Pretrained LLM Patching:**  
   Patch attention blocks in `Qwen/Qwen2.5-0.5B` using monkey-patch harnesses to benchmark real prefill and generation latency on Kaggle T4.
5. **Time-Series / Downstream Benchmarks:**  
   Evaluate long-context forecasting and retrieval on standard benchmarks (TiRex / LongBench).
