# Kernelized Linear Attention vs. Softmax Attention: An Efficiency–Quality Boundary Analysis

Team 3 research proposal — revised version (corrected mathematics, pre-registered design)

# 1. Title

Kernelized Linear Attention vs. Softmax Attention: An Efficiency–Quality Boundary Analysis

The original title asserted that attention quality is preserved. That is the result to be tested, so the title now states the method of study, not the outcome.

# 2. Research Problem

Standard softmax attention forms an N × N score matrix, so its compute grows quadratically with sequence length N. Kernelized linear attention replaces the softmax with a feature map φ so that keys and values can be accumulated into a fixed-size state, giving cost linear in N for fixed dimensions. The state is smaller, so it can mix details that a task needs to keep distinct. This study measures where the efficiency gain holds and what quality it costs.

Framing: linear attention is an established direction (Katharopoulos et al., 2020). This work is a reproduction and boundary analysis, not a claim of a new mechanism. Exact attention is also available in memory-efficient form (FlashAttention), so avoiding a stored N × N matrix is not unique to linear attention.

# 3. Research Questions

RQ1 (efficiency): At fixed dimensions, hardware, precision and implementation, how do latency and peak memory scale with N for each method, and where is the crossover?

RQ2 (quality): How do output deviation and key–value retrieval accuracy of linear attention compare with standard scaled softmax attention?

# 4. Hypotheses (pre-registered)

Define R = t_linear / t_softmax (median latency ratio) and Δ = acc_linear − acc_softmax (paired retrieval accuracy difference). Register a speed target R* < 1 and a non-inferiority margin δ > 0 before testing.

A combined claim (“faster with acceptable quality”) requires rejecting both null hypotheses. A nonsignificant difference in quality does not establish equivalence; only the non-inferiority test does. Memory is reported separately as measured peak memory, not folded into the speed claim.

# 5. Variables

Independent variable: attention method — S (standard scaled softmax), K (explicit kernel, small tests only), L (reordered linear); also sequence length N.

Dependent variables:

Latency (median, quantiles, repetition count)

Peak memory (state whether allocated, reserved or process memory, and whether inputs/outputs are included)

Relative Frobenius output error between L and S

Key–value retrieval accuracy with distractors

Failure lengths (out-of-memory or unrun points are recorded)

Theoretical operation counts are reported separately; they are not measured performance. “Task performance” from the original draft is removed because no downstream task is defined; it is a later extension.

Controls:

Batch size, number of heads, d_k, d_v, feature dimension r

Hardware, dtype/precision, framework and library versions, compiler settings

Identical Q, K, V inputs and random seeds shared across methods

Masking: noncausal first; causal as a separate experiment

Baseline implementation named (naive vs. optimized exact)

“Model architecture” is dropped as a control because this is a layer-level study with no full model.

# 6. Methodology

Fix d_k, d_v, r, batch, heads, precision and device; generate Q, K, V per seed.

Implement S: softmax(QKᵀ/√d_k)V. Implement K: form φ(Q)φ(K)ᵀ explicitly, row-normalize, multiply V. Implement L: build φ(K)ᵀV and φ(K)ᵀ1, then apply to φ(Q). Never build an N × N matrix inside L.

Unit checks: reproduce the three-token example in high precision; test random, zero, negative and single-token inputs; check denominator shapes, broadcasting, NaN/Inf, and epsilon placement. K and L must agree within floating-point tolerance before any comparison with S.

Efficiency run: N = 64, 256, 512, 1024, then geometric growth until relevant lengths or OOM. Warm up, repeat, interleave methods, synchronize the GPU, exclude data generation and transfers, and report compilation separately.

Output deviation: relative Frobenius error ‖O_L − O_S‖_F / ‖O_S‖_F with a documented safeguard for near-zero reference norms, across predefined input scales, seeds and lengths.

Retrieval: generate distinguishable keys bound to fixed-dimensional value codes; add distractors as N grows; decode with a fixed rule (class codes or nearest neighbor). Confirm first that S solves the task. Do not use N-dimensional one-hot values, which change d_v with N.

Statistics: at least five seeds; paired differences with confidence intervals, resampling at the independent generation unit. Repeated timings measure timing noise; independently generated tasks measure quality uncertainty. They are not interchangeable.

Analyze the efficiency–quality trade-off and report the crossover length and supported range.

# 7. Expected Outcome (conditional)

Under fixed dimensions, linear attention is expected to scale better with N and to overtake softmax beyond a crossover length. Quality may degrade when many similar keys interfere in the fixed-size state, and this depends on the feature map, r, and input scale. If gains appear only against a naive baseline, the conclusion is restricted accordingly.

# 8. Mathematical Experiment (corrected)

Question: can the N × N attention matrix be avoided, and what changes when it is?

## 8.1 Standard baseline

Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

Softmax is applied row-wise over keys. The original draft called this “scaled” but omitted 1/√d_k; the equation and the implementation must both include it.

## 8.2 Kernelized linear attention

LinearAttention(Q,K,V) = φ(Q)(φ(K)ᵀV) / φ(Q)(φ(K)ᵀ1)

Two distinct facts are used here. (i) Without softmax, associativity holds: (QKᵀ)V = Q(KᵀV). (ii) With softmax it does not: softmax(QKᵀ)V cannot be reordered. The linear method therefore changes the kernel, and for the same kernel the explicit and reordered computations must agree numerically while differing from softmax in general.

## 8.3 Step 1 — Three tokens (d_k = d_v = 2)

Q = [[1,2],[2,1],[1,1]]   K = [[1,0],[0,1],[1,1]]   V = [[10,0],[0,10],[5,5]]

## 8.4 Step 2 — Standard scores (the corrected step)

QKᵀ = [[1,2,3],[2,1,3],[1,1,2]]

Check for row 1: q₁ = (1,2) against k₁=(1,0), k₂=(0,1), k₃=(1,1) gives 1, 2, 3. The matrix [[7,8,10],[8,7,10],[6,6,8]] in the original draft is not QKᵀ; it is φ(Q)φ(K)ᵀ (Section 8.6).

## 8.5 Step 3 — Scaled softmax, worked for query 1

Scale by 1/√2: scores = (0.70711, 1.41421, 2.12132).

exp(scores) = (2.02811, 4.11325, 8.34214), sum = 14.48351

weights = (0.14003, 0.28400, 0.57598)

output₁ = 0.14003·(10,0) + 0.28400·(0,10) + 0.57598·(5,5) = (4.28017, 5.71983)

By the symmetry of the example, output₂ = (5.71983, 4.28017). For query 3, the scores (1,1,2)/√2 give equal weight on tokens 1 and 2, so output₃ = (5,5).

## 8.6 Step 4 — Feature map

φ(x) = ReLU(x) + 1 is used only as an illustrative map, not as a universal definition of linear attention, and it does not by itself guarantee a good softmax approximation.

φ(Q) = [[2,3],[3,2],[2,2]]   φ(K) = [[2,1],[1,2],[2,2]]

φ(Q)φ(K)ᵀ = [[7,8,10],[8,7,10],[6,6,8]]   (mapped scores)

## 8.7 Step 5 — Reordered linear computation (no 3 × 3 matrix)

φ(K)ᵀV = [[30,20],[20,30]]      φ(K)ᵀ1 = [5,5]ᵀ

numerators  φ(Q)(φ(K)ᵀV) = [[120,130],[130,120],[100,100]]

denominators φ(Q)(φ(K)ᵀ1) = [25, 25, 20]ᵀ

output = [[4.8,5.2],[5.2,4.8],[5,5]]

Query 1 check: (2,3)·[[30,20],[20,30]] = (60+60, 40+90) = (120,130); denominator 2·5+3·5 = 25; 120/25 = 4.8, 130/25 = 5.2. The explicit kernel route (row-normalizing φ(Q)φ(K)ᵀ, weights (0.28, 0.32, 0.40) for query 1, then multiplying by V) gives the same output, confirming K ≈ L. The compact state is r × d_v = 2 × 2. In this toy case N = 3 and r = 2 are similar in size, so no speed gain is implied; linear cost O(N·r·d_v) beats O(N²·d) only when N is large relative to r.

## 8.8 Step 6 — Comparison (first output row, deterministic recalculation)

The original draft compared linear attention with the mapped unscaled softmax value (4.64, 5.36). The correct baseline is the original scaled softmax (4.28, 5.72). Against that baseline the relative Frobenius error of the linear output is about 8.4%, larger than the gap the original draft implied. Effective attention weights also differ in shape: linear gives (0.28, 0.32, 0.40) for query 1, while softmax concentrates 0.576 on token 3. This is a deterministic recalculation on one toy example, not a performance result.

# 9. Causal Extension (separate experiment)

An autoregressive linear state must accumulate only current and earlier keys and values; using full-sequence KᵀV leaks future information. Test that changing future values cannot change earlier outputs. Report prefill and token-by-token decoding separately.

# 10. Meaning of Long-Range Retrieval

Noncausal attention without positional encoding is permutation-equivariant, so moving the target farther away does not test long-range memory. Define a positional encoding or a causal accumulation task, and vary distance separately from the number of distractors.

# 11. Experimental Sequence

The original lengths (N = 3 to 1,000) are too small to show scaling, since fixed overheads dominate. Use N = 64, 256, 512, 1024 and continue geometrically. Include OOM and unrun points in the results table.

# 12. Acceptance Criteria and Decisions

Speed: R < R* across the predefined length range; if short inputs are slower and long inputs faster, report the crossover and supported range.

Quality: lower bound of the interval for Δ exceeds −δ. Small output error does not establish preserved quality if retrieval accuracy falls.

If K and L disagree, fix the computation and normalization before discussing quality.

If speed improves but retrieval worsens, report the trade-off and inspect interference, input scale and r; increasing r includes its extra cost.

If an optimized exact baseline beats the naive comparison, revise the engineering conclusion and keep the unfavorable control.

# 13. Limitations

Results apply only to the tested hardware, precision, feature map, dimensions and length range. Output similarity is not downstream task quality. Replacing attention in a pretrained model changes its representation distribution and may need adaptation, which is a separate study.

# 14. Conclusion Template

Under [hardware and precision], [fixed dimensions] and [length range], the specified method had [time ratio and variability] relative to [baseline implementation], with [peak memory]. Retrieval accuracy differed by [estimate and interval], which [meets or fails] the predefined δ. The conclusion applies to [conditions]. Next, test [remaining question].

# 15. References

Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020. https://proceedings.mlr.press/v119/katharopoulos20a.html

Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. https://arxiv.org/abs/2205.14135


## Tables

### Table 1

|  | Speed claim | Quality claim |
| --- | --- | --- |
| H₀ | R ≥ R* over the predefined length range | Δ ≤ −δ (linear is worse by at least δ) |
| H₁ | R < R* over the predefined length range | Δ > −δ (lower confidence bound above −δ) |


### Table 2

| Computation | Output row 1 | Role |
| --- | --- | --- |
| Original scaled softmax | [4.28017, 5.71983] | Standard baseline |
| Original unscaled softmax | [4.22651, 5.77349] | Equation check only |
| Mapped unscaled softmax (softmax of φ(Q)φ(K)ᵀ) | [4.63907, 5.36093] | Value shown in original draft |
| Normalized linear attention | [4.8, 5.2] | Method under study |


### Table 3

| Factor | Design | Record |
| --- | --- | --- |
| Sequence length | Predefined groups, geometric growth | OOM and unrun points |
| Feature dimension r | Fixed first; separate sweep later | Mapping and state cost |
| Input scale | Predefined scales | Shared inputs across methods |
| Seeds | At least five | Seed-level results |
| Baseline | Naive and optimized (e.g., FlashAttention) named separately | Backend and version |

