# Team 3 Linear Attention Research Workbook

Correct the attention comparison, run reproducible efficiency and retrieval experiments, and state the conditions under which benefits hold. Theoretical complexity and a single worked example are not measured performance. Progress through completion criteria without a fixed time limit.

Project lead: To complete    Experiment lead: To complete
Reviewer: To complete    Work date: To complete    Version: 1.0

## 1 Current status

The proposal defines an efficiency-quality trade-off, identifies controls, and gives a three-token calculation. It does not report runtime, memory, retrieval results, or statistical evidence. Section 12 labels feature-mapped scores as the original QK transpose product; correct this first.

## 2 Intuition

Standard attention compares each query against every key and aggregates values. Kernelized linear attention accumulates keys and values into a fixed-dimensional state and lets queries read it. The smaller state can mix details that the task needs to distinguish.

## 3 Plan and hypotheses

RQ1: at fixed dimensions, hardware, precision, and implementation, how do latency and memory scale with sequence length? RQ2: how do output deviation and key-value retrieval accuracy compare with standard scaled softmax attention?

Test efficiency and quality separately. Use a timing ratio R=linear/softmax and a retrieval accuracy difference. A claim of faster computation with acceptable quality must meet both a predefined speed target and a non-inferiority margin δ.

A nonsignificant quality difference does not establish equivalence. Register δ, lengths, primary metrics, and statistics before testing. Linear attention is an established research direction; frame this work as reproduction and boundary analysis unless there is evidence of a new contribution.

# Team 3 Mathematical Corrections

## 4 Correct standard baseline

Standard scaled dot-product attention is softmax(QK^T/sqrt(d_k))V, with row normalization over keys. The proposal calls the operation scaled attention but omits the scale factor; align the equation and implementation.

Given Q=[[1,2],[2,1],[1,1]], K=[[1,0],[0,1],[1,1]], and V=[[10,0],[0,10],[5,5]], the correct QK^T is [[1,2,3],[2,1,3],[1,1,2]].

The reported [[7,8,10],[8,7,10],[6,6,8]] instead equals φ(Q)φ(K)^T for φ(x)=ReLU(x)+1. Label original and mapped inputs separately.

The second row swaps the first row components; the third is [5,5]. These are deterministic recalculations, not performance results. The proposal’s linear intermediate state, denominator, and output are internally consistent.

## 5 Limits of associativity

(QK^T)V=Q(K^TV), but softmax(QK^T)V cannot be reordered this way. The linear path computes φ(Q)(φ(K)^TV), divided by φ(Q)(φ(K)^T1). Each query has a scalar denominator broadcast across value dimensions.

Verify two distinct claims: the explicit and reordered computations for the same kernel should agree numerically; that kernel and softmax generally produce different outputs. ReLU+1 does not by itself establish an accurate softmax approximation.

## 6 Complexity scope

For feature dimension r and value dimension d_v, the compact state is r×d_v and major reordered computation is approximately O(N r d_v), plus feature-mapping cost. Linear refers to N with dimensions fixed; it does not imply superiority for every dimension or implementation.

# Team 3 Do Correctness and Implementation

## 7 Three implementation paths

Use K only for small correctness checks. Do not secretly construct an N×N matrix inside L during timing or large runs. K and L should agree within floating-point tolerance. Differences between S and L reflect the kernel change.

## 8 Unit checks

Reproduce the three-token case in high precision, then check the benchmark dtype. Set tolerances appropriate to precision and scale; report absolute and relative errors.

Test random, zero, negative, and single-token inputs, plus different value dimensions. Check denominator shapes, broadcasting, NaNs, and infinities.

Record epsilon and where it is applied. ReLU+1 gives positive features for finite inputs, but other maps and low precision need stability checks.

Match masks, dropout, batch size, heads, dtype, and device. Complete noncausal attention first; treat causal attention as a separate experiment.

## 9 Causal extension

Autoregressive linear states must accumulate only current and earlier keys and values. Full-sequence K^TV leaks future information. Test that changing future values cannot alter earlier outputs. Report prefill and token-by-token decoding separately.

## 10 Implementation record

Record framework, library and compiler versions, CPU/GPU, precision, compilation, mixed precision, and gradient mode. For optimized attention, save the backend actually selected; masks or dtype may trigger fallback.

A clearly named naive baseline is acceptable initially. Engineering superiority claims should later include supported optimized exact attention. FlashAttention reduces intermediate memory while retaining exact attention, so avoiding a stored full matrix is not unique to linear attention.

# Team 3 Do Performance and Retrieval

## 11 Efficiency experiment

Start with N=64, 256, 512, and 1024. Increase geometrically if hardware permits until reaching relevant lengths or out-of-memory limits. Fix batch size, heads, d_k, d_v, r, and precision. Exclude input generation, transfers, and validation from timing unless explicitly measuring an end-to-end pipeline.

Warm up each path, repeat timings, and report medians, quantiles, and counts. Use GPU events or boundary synchronization to measure execution rather than asynchronous submission. Report compilation and startup separately. Interleave methods to reduce thermal and background-load effects.

Measure peak memory separately per method. State allocated, reserved, or process memory; define baseline, peak reset, and whether inputs/outputs are included. A fixed-size state does not make the whole workload constant-memory because inputs and outputs still grow with N.

## 12 Output deviation

For identical Q, K, and V, calculate relative Frobenius error between L and S, with a documented safeguard for near-zero reference norms. Cover predefined input scales, seeds, and lengths. Output similarity is not downstream task quality.

## 13 Key-value retrieval with distractors

Generate distinguishable keys bound to values. A query targets one key, optionally with predefined noise; add distractors as N grows. Both methods receive identical inputs and use the same fixed decoding rule to identify the retrieved value.

Use fixed-dimensional class codes with balanced classes, or fixed-dimensional random codes and nearest-neighbor decoding. Do not default to N-dimensional one-hot values: increasing d_v with N violates the fixed-dimension scaling comparison. Record collisions and decoding limits.

Fix or systematically vary query/key norms and signal strength. Small random logits can make softmax retrieval poor too. Establish that the standard baseline can solve the intended task before interpreting relative quality.

## 14 Meaning of long-range retrieval

Noncausal attention without positional encoding has the corresponding permutation behavior. Merely moving a target farther away does not test long-range memory. Define positional encoding or a causal accumulation task, and control distance separately from distractor count.

# Team 3 Check Metrics and Decisions

## 15 Experiment matrix

## 16 Primary metrics

Report latency, speedup=softmax_time/linear_time, peak memory, and failure lengths. Report retrieval accuracy differences, relative output errors, and uncertainty. Keep theoretical operations separate from observed time; O(N) is not a measured millisecond saving.

Repeated timings quantify performance noise; independently generated retrieval tasks quantify quality uncertainty. Do not treat these as interchangeable samples. Use paired quality differences and intervals on shared tasks, resampling at the independent generation unit when repetitions are correlated.

## 17 Acceptance criteria

Set quality tolerance δ and speed targets before testing. For Δ=accuracy_linear-accuracy_softmax, non-inferiority requires the interval’s lower bound to exceed -δ. Speed must meet the target over the predefined length range. Both are required for a combined claim.

If short inputs are slower but long inputs faster, report the crossover and supported range. If gains exist only against a naive baseline, restrict the conclusion. Small output error does not establish preserved quality when retrieval accuracy falls substantially.

## 18 Registration fields

Lengths, fixed dimensions, precision, device, and timing repetitions
Team entry: __________________________________________________________
____________________________________________________________________

Primary quality metric, δ, speed target, and interval method
Team entry: __________________________________________________________
____________________________________________________________________

Retrieval generator, decoder, test seeds, and extension triggers
Team entry: __________________________________________________________
____________________________________________________________________

# Team 3 Act and Submission

## 19 Decisions after checking

If K and L disagree, fix computation and normalization before discussing the quality trade-off.

If L differs from S but retrieval meets tolerance, report acceptable alternative behavior under those settings. Do not claim exact equivalence.

If speed improves but retrieval worsens, report the trade-off and inspect interference, input scale, and feature dimension. Increasing r must include its additional cost.

If an optimized exact baseline is faster than the naive comparison suggests, revise the engineering conclusion and preserve the unfavorable control.

After basic checks pass, consider model training or layer replacement. Replacing pretrained attention also changes the model’s representation distribution and may require adaptation; evaluate that separately.

## 20 Submission checklist

Submit corrected equations and calculations, all three implementations, correctness checks, benchmark settings, raw timing logs, environment, retrieval generator, result tables, efficiency-quality curves, and limitations. Preserve OOM cases, failures, and unsupported hypotheses.

## 21 Conclusion template

Under [hardware and precision], [fixed dimensions], and [length range], the specified method had [time ratio and variability] relative to [baseline implementation], with [peak memory]. Retrieval accuracy differed by [estimate and interval], which [meets or fails] the predefined δ. The conclusion applies to [conditions]. Next, test [remaining question].

## 22 References

Team source: Linear_Attention_Research_Team_3.docx, section 12 calculations and section 14 experimental sequence.

Katharopoulos et al. Transformers are RNNs
https://proceedings.mlr.press/v119/katharopoulos20a.html

Dao et al. FlashAttention
https://arxiv.org/abs/2205.14135

These references distinguish kernelized linear methods, standard softmax, and optimized exact attention. Record the actual implementation and version used in each experiment.

# Team 3 Execution and Review Record

## Execution record

## Cycle review

Completed work and supporting evidence
Team entry: __________________________________________________________
____________________________________________________________________

Findings that differed from the hypothesis
Team entry: __________________________________________________________
____________________________________________________________________

Method retained or changed and the reason
Team entry: __________________________________________________________
____________________________________________________________________

Next research question and responsible person
Team entry: __________________________________________________________
____________________________________________________________________


## Tables

### Table 1

| PDCA | Existing work | Required next step |
| --- | --- | --- |
| Plan | Question and controls identified | Define baseline and quality tolerance |
| Do | Worked example explains associativity | Correct it and implement both computations |
| Check | Efficiency and quality metrics proposed | Run scaling and retrieval experiments |
| Act | Longer sequences proposed | Determine measured operating boundaries |


### Table 2

| Computation | First output row | Role |
| --- | --- | --- |
| Original scaled softmax | [4.280169, 5.719831] | Standard baseline |
| Original unscaled softmax | [4.226511, 5.773489] | Equation check only |
| Mapped unscaled softmax | [4.639074, 5.360926] | Matches reported value |
| Normalized linear attention | [4.8, 5.2] | Specified method |


### Table 3

| Path | Computation | Purpose |
| --- | --- | --- |
| S Standard | softmax(QK^T/sqrt(d_k))V | Behavior and performance baseline |
| K Explicit kernel | Form φ(Q)φ(K)^T, normalize, multiply V | Small-scale numerical reference |
| L Reordered linear | Build φ(K)^TV and φ(K)^T1, then query | Method under evaluation |


### Table 4

| Factor | Design | Record |
| --- | --- | --- |
| Sequence length | Increase in predefined groups | Include OOM and unrun points |
| Feature dimension r | Fixed initially; separate sweep later | Include mapping and state costs |
| Input distribution | Predefined scales and conditions | Share inputs across methods |
| Random seeds | Prefer at least five; expand as needed | Preserve seed-level results |
| Baseline implementation | Name naive and optimized separately | Backend and version |


### Table 5

| Length and method | Latency | Peak memory | Output error | Retrieval |
| --- | --- | --- | --- | --- |
| Pending | Pending | Pending | Pending | Pending |
| Pending | Pending | Pending | Pending | Pending |


### Table 6

| Deliverable | Owner | Status and evidence |
| --- | --- | --- |
| Mathematics and implementation review | To complete | To complete |
| Performance and retrieval experiments | To complete | To complete |
| Analysis and revised proposal | To complete | To complete |


### Table 7

| Record | Team entry |
| --- | --- |
| Run ID and code version | To complete |
| Operator and reviewer | To complete |
| Data version and split manifest | To complete |
| Environment and random seeds | To complete |
| Raw results and figure locations | To complete |
| Deviations from the plan and reasons | To complete |

