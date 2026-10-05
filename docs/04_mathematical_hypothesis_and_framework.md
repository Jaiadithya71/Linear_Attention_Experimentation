# Error-Controlled Adaptive-Rank Linear Attention
## Compressed Mathematical Hypothesis and Research Framework

Error-Controlled Adaptive-Rank Linear Attention

Compressed Mathematical Hypothesis and Research Framework


## 1. Research Question

Can the quadratic interaction in softmax attention be replaced by a linear-complexity kernel formulation while guaranteeing a prescribed approximation tolerance, with the required feature dimension selected according to the input?


$${Y}^{S}=softmax\left(\frac{Q{K}^{T}}{\sqrt{{d}_{k}}}\right)V$$


## 2. Central Hypothesis

H1 — Error-Controlled Adaptive-Rank Hypothesis

For a prescribed output tolerance $δ$, the minimum feature dimension required to approximate normalized softmax attention is input-dependent. Therefore, an input-conditioned feature rank can achieve the required quality with less feature computation than a conservative globally fixed rank on inputs that admit a lower approximation dimension.


$${r}^{∗}\left(X,δ\right)=min\left{r∈ℛ:B\left(X,r\right)≤δ\right}$$

$B\left(X,r\right)$ denotes a valid upper bound on the attention-output error produced by an $r$-dimensional positive-kernel approximation.


## 3. Exact Softmax Attention

For $Q∈{ℝ}^{N×{d}_{k}}$, $K∈{ℝ}^{N×{d}_{k}}$, and $V∈{ℝ}^{N×{d}_{v}}$, define:


$$S=\frac{Q{K}^{T}}{\sqrt{{d}_{k}}}$$


$${A}^{S}={softmax}_{row}\left(S\right)$$


$${Y}^{S}={A}^{S}V$$

Equivalently, for query i:


$${κ}_{ij}=\exp \left(\frac{{{q}_{i}}^{T}{k}_{j}}{\sqrt{{d}_{k}}}\right)$$


$${Z}_{i}=∑_{j=1}^{N} {κ}_{ij}$$


$${y}_{i}^{S}=\frac{1}{{Z}_{i}}∑_{j=1}^{N} {κ}_{ij}{v}_{j}$$

The dense score/attention matrix contains ${N}^{2}$ pairwise interactions, giving $O\left({N}^{2}{d}_{k}+{N}^{2}{d}_{v}\right)$ arithmetic for the direct formulation and $O\left({N}^{2}\right)$ interaction storage.


## 4. Kernelized Linear Attention

Let ${φ}_{r}:{ℝ}^{\left({d}_{k}\right)}→{ℝ}^{r}$ be a non-negative feature map and define:


$${\tilde{κ}}_{ij}={\left({φ}_{r}\left({q}_{i}\right)\right)}^{T}{φ}_{r}\left({k}_{j}\right)≥0$$


$${\tilde{Z}}_{i}=∑_{j=1}^{N} {\tilde{κ}}_{ij}$$


$${y}_{i}^{L}=\frac{1}{{\tilde{Z}}_{i}}∑_{j=1}^{N} {\tilde{κ}}_{ij}{v}_{j}$$

Let ${Φ}_{Q}∈{ℝ}^{N×r}$ and ${Φ}_{K}∈{ℝ}^{N×r}$ contain the feature vectors. Then:


$${Y}^{L}={\tilde{D}}^{−1}{Φ}_{Q}\left({{Φ}_{K}}^{T}V\right)$$


$$\tilde{D}=diag\left({Φ}_{Q}{{Φ}_{K}}^{T}{1}_{N}\right)$$

The equality follows from associativity:


$$\left({Φ}_{Q}{{Φ}_{K}}^{T}\right)V={Φ}_{Q}\left({{Φ}_{K}}^{T}V\right)$$

Thus the $N×N$ matrix need not be formed. The dominant matrix products cost $O\left(Nr{d}_{v}\right)$, while feature construction contributes $O\left(Nr{d}_{k}\right)$. The kernel state ${{Φ}_{K}}^{T}V$ has only $r·{d}_{v}$ entries per head.


## 5. Connection to Softmax: Error Theorem

The linear formulation is not automatically equivalent to softmax. The approximation must therefore be controlled. Let:


$${κ}_{ij}=\exp \left(\frac{{{q}_{i}}^{T}{k}_{j}}{\sqrt{{d}_{k}}}\right)$$


$${\tilde{κ}}_{ij}={\left({φ}_{r}\left({q}_{i}\right)\right)}^{T}{φ}_{r}\left({k}_{j}\right)$$


$${n}_{i}=∑_{j}^{} {κ}_{ij}{v}_{j}$$$${\tilde{n}}_{i}=∑_{j}^{} {\tilde{κ}}_{ij}{v}_{j}$$


$${y}_{i}^{S}=\frac{{n}_{i}}{{Z}_{i}}$$$${y}_{i}^{L}=\frac{{\tilde{n}}_{i}}{{\tilde{Z}}_{i}}$$

Assume, for every i,j:

$\|{\tilde{κ}}_{ij}−{κ}_{ij}\|≤{ε}_{r}$

${κ}_{ij}≥0  and  {\tilde{κ}}_{ij}≥0$

${\left‖{v}_{j}\right‖}_{j}≤{V}_{max}$

${Z}_{i}>N{ε}_{r}$

Then:


$$\|{\tilde{Z}}_{i}−{Z}_{i}\|≤N{ε}_{r}$$


$${\left‖{\tilde{n}}_{i}−{n}_{i}\right‖}_{i}≤N{ε}_{r}{V}_{max}$$


$${\tilde{Z}}_{i}≥{Z}_{i}−N{ε}_{r}$$

Using ${y}_{i}^{L}−{y}_{i}^{S}=\frac{{\tilde{n}}_{i}−{n}_{i}}{{\tilde{Z}}_{i}}+{n}_{i}\left(\frac{1}{{\tilde{Z}}_{i}}−\frac{1}{{Z}_{i}}\right)$, together with ${\left‖{n}_{i}\right‖}_{i}≤{Z}_{i}{V}_{max}$, gives:


$${\left‖{y}_{i}^{L}−{y}_{i}^{S}\right‖}_{i}≤\frac{2N{ε}_{r}{V}_{max}}{{Z}_{i}−N{ε}_{r}}$$

Therefore, a sufficient condition for ${\left‖{y}_{i}^{L}−{y}_{i}^{S}\right‖}_{i}≤δ$ is:


$${ε}_{r}≤\frac{δ{Z}_{i}}{N\left(2{V}_{max}+δ\right)}$$

This is the mathematical bridge from kernel approximation to attention-output accuracy.


## 6. Adaptive-Rank Rule

If a valid bound ${ε}_{r}\left(X\right)$ is available for the selected feature construction (uniform over all rows $i$), choose the smallest rank satisfying the sufficient condition:


$${r}^{∗}\left(X,δ\right)=min\left{r:{ε}_{r}\left(X\right)≤\frac{δ\{min}_{i} min}{N\left(2{V}_{max}+δ\right)}\right}$$

This produces a quality-controlled computational budget rather than treating $r$ as a fixed hyperparameter. The theoretical task is to obtain a valid ${ε}_{r}\left(X\right)$ bound with explicit dependence on feature dimension, input geometry and confidence.


## 7. Testable Predictions

Required rank varies with input geometry, query–key scale/temperature and attention concentration.

Increasing $r$ reduces approximation error over a measurable operating regime.

Some inputs satisfy a fixed tolerance at substantially smaller $r$ than others.

Adaptive rank can reduce average feature-state computation while remaining within the prescribed tolerance.

If difficult inputs require $r$ comparable to $N$, or rank-selection overhead removes the savings, the hypothesis is not supported in that regime.


## 8. Research Position and Relevant Prior Work

The novelty claim must be narrow. Kernelized linear attention and random-feature approximation of softmax are established. The proposed research is the combination of an explicit output-error certificate with an input-conditioned minimum-rank rule and its measured efficiency boundary.

Katharopoulos et al. (2020), Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention — establishes kernelized linear attention through associativity and linear sequence-length complexity.

Choromanski et al. (2020/2021), Rethinking Attention with Performers — establishes FAVOR+ positive random features for linear-time/space approximation of softmax attention with theoretical guarantees.

Likhosherstov et al. (2023), FAVOR# — develops sharper positive random-feature approximations and variance reduction for softmax/Gaussian kernels.

Sui & Zhang (2026), The Approximation Rank of Softmax Attention — directly studies approximation rank and its dependence on geometry and temperature; this makes a generic 'adaptive rank is novel' claim inappropriate and motivates a more specific certificate-to-computation contribution.


## 9. Experimental Validation

Compare exact softmax, explicit kernel attention and reordered kernel attention.

Verify explicit and reordered forms numerically agree before quality comparisons.

Sweep $r$ and measure ${E}_{r}=\frac{{\left‖{Y}_{r}^{L}−{Y}^{S}\right‖}_{r}}{{\left‖{Y}^{S}\right‖}_{F}}$.

Compare certified/selected rank against the empirical minimum rank satisfying the target $δ$.

Measure latency and peak memory, including rank-selection overhead.

Evaluate retrieval accuracy using fixed-dimensional value codes and controlled distractor margins.

The final result should report both successful operating regimes and failure boundaries.


## 10. Final Research Hypothesis

H1: For normalized softmax attention, there exists an input-conditioned feature dimension ${r}^{∗}\left(X,δ\right)$, selected from a valid approximation-error certificate, such that positive-kernel linear attention satisfies a prescribed output tolerance $δ$ while retaining linear sequence-length complexity. Relative to a conservative fixed-rank approximation, this adaptive allocation can reduce average feature computation over a defined operating regime.


## Selected References

Katharopoulos et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. https://arxiv.org/abs/2006.16236

Choromanski et al. (2020). Rethinking Attention with Performers. https://arxiv.org/abs/2009.14794

Likhosherstov et al. (2023). FAVOR#: Sharp Attention Kernel Approximations via New Classes of Positive Random Features. https://arxiv.org/abs/2302.00787

Sui & Zhang (2026). The Approximation Rank of Softmax Attention: Sharp Geometric Laws and Robust Interaction Dimension. https://arxiv.org/abs/2608.28150
