# Two-Person Execution Plan: Adaptive Attention Sprint

**Project:** Adaptive Attention Computation: Balancing Efficiency and Retrieval in Long-Context Transformers  
**Team Structure:** Sub-Group of 2 Engineers / Researchers  
**Total Duration:** 8.5 – 9.5 Hours  
**Target Hardware:** Kaggle Free Tier (Dual Tesla T4, 16GB VRAM)  
**Primary Deliverables:** Verified Codebase, Benchmark CSVs, and Master 7-Slide Deck  

---

## 1. High-Level Role Partitioning

The original 6-person sprint is consolidated into two focused roles with **Contract-First Decoupling**, ensuring neither engineer blocks the other.

```
┌────────────────────────────────────────────────────────┐
│                      2-PERSON SUB-GROUP                │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│        PERSON 1: RESEARCH     │ │        PERSON 2: TESTING      │
│  Theory, Architecture & Deck  │ │  Efficiency, Retrieval, SOTA  │
├───────────────────────────────┤ ├───────────────────────────────┤
│ • Mathematical bounds & error │ │ • T4 compute & memory harness │
│   certificates (r*(X, δ))     │ │ • Passkey / Needle benchmark  │
│ • Adaptive kernel development │ │ • Gated DeltaNet baseline     │
│ • Qwen2.5-0.5B monkey-patch   │ │ • Data collection & charts    │
│ • 7-slide deck narrative      │ │ • Non-inferiority validation  │
└───────────────────────────────┘ └───────────────────────────────┘
```

| Dimension | Person 1: Research Lead (Theory & Architecture) | Person 2: Testing Lead (Compute & Validation) |
| :--- | :--- | :--- |
| **Merged Original Roles** | Person 1 (Theory) + Person 4 (Model Patch) + Person 6 (Synthesis & Deck) | Person 2 (Compute) + Person 3 (Retrieval) + Person 5 (SOTA Baselines) |
| **Core Mission** | Design the mathematical bounds, build the adaptive attention kernel, patch the model, and author the presentation narrative. | Build the evaluation harnesses, benchmark T4 latency/VRAM, execute the retrieval suite, and validate against SOTA baselines. |
| **Primary Code Artifacts**| `adaptive_kernel.py`, `patch_qwen.py`, Master 7-Slide Deck | `01_T4_Efficiency.ipynb`, `02_Passkey_Suite.ipynb`, `03_DeltaNet_T4.ipynb`, CSVs |

---

## 2. Decoupled Contract-First Interface (Hour 0 Lock)

To allow both engineers to work in parallel without idle time, interfaces are frozen in Hour 0:

### A. Kernel I/O Contract (`adaptive_kernel.py`)
```python
def adaptive_rank_attention(
    Q: torch.Tensor,       # (B, H, N, d_k)
    K: torch.Tensor,       # (B, H, N, d_k)
    V: torch.Tensor,       # (B, H, N, d_v)
    delta: float = 0.05,   # output error tolerance target
    r_candidates: tuple = (64, 256, 1024),
    is_causal: bool = False
) -> torch.Tensor:         # (B, H, N, d_v)
```
- **Mock implementation:** For Hours 1.0–6.0, Person 2 tests pipelines against a mock kernel returning `linear_attention(Q, K, V)`.

### B. Standardized Metric Schema (`benchmark_results.csv`)
Both engineers output standardized records:
`[timestamp, sequence_length_N, method, metric_name, metric_value, peak_vram_mb, status]`

---

## 3. Hour-by-Hour Master Timeline & Deadlines

```
Hour 0.0  ───────────────────► Contract Freeze (Tensor I/O, CSV Schema, Mock Kernel)
Hour 1.0  ───────────────────► Phase 2: Independent Core Implementation
Hour 3.5  ───────────────────► Checkpoint 1: Baselines & Kernel Prototypes Verified
Hour 6.0  ───────────────────► Checkpoint 2: Long Context Sweeps (N=64K) Complete
Hour 7.5  ───────────────────► CODE FREEZE & MERGE: Hand off Adaptive Kernel to Test Harness
Hour 8.5  ───────────────────► FINAL AUDIT: All CSVs & Figures Ingested into Deck
Hour 9.5  ───────────────────► Presentation Dry-Run & Project Sign-Off
```

### Detailed Schedule:

| Time Window | Person 1: Research Lead (Theory & Architecture) | Person 2: Testing Lead (Compute & Validation) | Shared Milestone & Sync |
| :--- | :--- | :--- | :--- |
| **0.0 – 1.0 h** | • Formalize adaptive-rank bound: $\epsilon_r \le \frac{\delta \min_i Z_i}{N(2V_{\max} + \delta)}$<br>• Publish `MockAdaptiveKernel` stub<br>• Outline 7-slide deck narrative structure | • Setup Kaggle T4 harness (dual T4, 16GB)<br>• Lock PyTorch SDPA memory-efficient flags<br>• Implement synthetic Passkey generator ($N = 1\text{K}\dots 64\text{K}$)<br>• Lock unified CSV schema | **SYNC 1 (Hour 1.0):**<br>Interface freeze. Sign-off on mock kernel I/O and evaluation CSV schema. |
| **1.0 – 3.5 h** | • Implement exact output-error estimator in `adaptive_kernel.py`<br>• Build dynamic rank selection logic ($r \in \{64, 256, 1024\}$)<br>• Verify 3-token analytical toy example ($[4.8, 5.2]$)<br>• Setup Qwen2.5-0.5B attention monkey-patch harness | • Benchmark Softmax vs SDPA efficiency ($N = 64\dots 32,768$)<br>• Run Passkey baseline retrieval on Softmax (verify $100\%$ accuracy)<br>• Implement standalone PyTorch Gated DeltaNet layer<br>• Automated plotting script ready | **SYNC 2 (Hour 3.5):**<br>Verify baseline data. Softmax sanity check confirmed. DeltaNet layer running. |
| **3.5 – 6.0 h** | • Implement numerical stability safeguards (FP32 accumulators, $\epsilon = 10^{-6}$)<br>• Test causal masking / prefix-sum implementation<br>• Validate monkey-patched forward pass on `Qwen2.5-0.5B`<br>• Draft Slide 1 (Dilemma), Slide 2 (Hypothesis), Slide 5 (Gap) | • Execute latency & peak VRAM sweep for Linear ($\text{ReLU}+1$) up to $N = 65,536$<br>• Run Passkey retrieval sweep for Linear vs FAVOR+ ($r=64, 256, 1024$)<br>• Benchmark DeltaNet throughput vs Linear Attention on T4 | **SYNC 3 (Hour 6.0):**<br>Review initial scaling and retrieval curves. Confirm linear scaling ($9.7\text{ ms}$) and retrieval collapse ($14\%$). |
| **6.0 – 7.5 h** | • Package finalized `adaptive_kernel.py`<br>• Profile prefill latency on patched Qwen2.5-0.5B ($N=1\text{K}\dots 8\text{K}$)<br>• Draft Slide 6 (Proposed Architecture) & Slide 7 (Roadmap) | • Ingest `adaptive_kernel.py` into testing suite<br>• Run full retrieval and efficiency comparison on Adaptive Kernel<br>• Compute bootstrap 95% CI for non-inferiority ($\Delta$ vs $\delta = 0.05$)<br>• Export all final figures | **MILESTONE (Hour 7.5):**<br>**CODE FREEZE.** Final numbers locked. No further code edits. |
| **7.5 – 8.5 h** | • Ingest generated PNG charts and summary tables into 7-slide deck<br>• Write executive summary report and slide speaker notes | • Cross-check all CSV numbers against raw notebook logs<br>• Verify Kaggle reproduction script runs end-to-end without errors | **SYNC 4 (Hour 8.5):**<br>Master slide deck review. Verify every data point matches empirical CSVs. |
| **8.5 – 9.5 h** | • Lead 15-minute presentation dry-run<br>• Rehearse theory and architecture Q&A defense | • Present empirical methodology and baseline comparisons during dry-run<br>• Package deliverables repository | **FINAL SIGN-OFF (Hour 9.5):**<br>Deliverables locked and submitted. |

---

## 4. Responsibility Matrix (RACI)

- **R (Responsible):** Does the work.
- **A (Accountable):** Signs off on the work.
- **C (Consulted):** Provides inputs.
- **I (Informed):** Kept updated.

| Deliverable / Activity | Research Lead | Testing Lead |
| :--- | :---: | :---: |
| **Mathematical Bounds & Error Certificate** | **R / A** | C |
| **Standalone Adaptive Kernel (`adaptive_kernel.py`)** | **R / A** | C |
| **Kaggle T4 Environment & OOM Guardrails** | C | **R / A** |
| **Efficiency Profiling (Latency & VRAM Sweeps)** | I | **R / A** |
| **Passkey Retrieval Suite & Distractor Testing** | C | **R / A** |
| **Gated DeltaNet Baseline Implementation** | C | **R / A** |
| **Qwen2.5-0.5B Model Monkey-Patching** | **R / A** | C |
| **Statistical Non-Inferiority Audit (Bootstrap CI)** | C | **R / A** |
| **Master 7-Slide Deck & Executive Summary** | **R / A** | C |
| **Final Code & Reproduction Verification** | C | **R / A** |

---

## 5. Kaggle Free Tier (Tesla T4) Operational Guardrails

Both members must adhere to these hardware constraints during development and benchmarking:

1. **Memory-Efficient SDPA:**  
   Tesla T4 (Turing SM 7.5) does not support FlashAttention-2. Use:
   ```python
   torch.backends.cuda.enable_flash_sdp(False)
   torch.backends.cuda.enable_mem_efficient_sdp(True)
   ```
2. **Batch Size Strategy:**  
   Fix $B = 1$ whenever $N \ge 8,192$ to avoid VRAM exhaustion.
3. **Model Footprint:**  
   Model patching is strictly confined to `Qwen/Qwen2.5-0.5B` (~1.0 GB VRAM in FP16), leaving $> 12\text{ GB}$ for intermediate states and KV cache.
4. **Session Timeout Prevention:**  
   For long evaluation sweeps ($N \ge 16,384$), run notebooks via Kaggle *Save & Run All (Commit)* rather than interactive cells.
