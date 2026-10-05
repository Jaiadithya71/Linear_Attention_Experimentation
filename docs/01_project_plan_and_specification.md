# Project Plan & Technical Specification: Adaptive Attention Computation

**Project Title:** Adaptive Attention Computation: Balancing Efficiency and Retrieval in Long-Context Transformers  
**Team Size:** 6 Engineers / Researchers  
**Total Workload:** 8.0 – 9.5 Hours per Person  
**Hardware Environment:** Kaggle Free Tier (Dual Tesla T4, 16GB VRAM)  
**Target Milestone:** Empirical Results & 7-Slide Deck  

---

## Executive Summary & Core Research Pivot

Initial benchmarks proved that basic kernelized linear attention achieves linear scaling:
- **Latency at $N = 65,536$:** **$9.7\text{ ms}$** for Linear Attention vs **$1,602\text{ ms}$** for SDPA (Flash/Memory-Efficient Exact Attention).
- **Retrieval Collapse:** At $N = 1,024$, Linear Attention achieves only **$14\%$ key-value retrieval accuracy** vs **$100\%$ for Softmax**.

> **The Research Pivot:**  
> Standard linear attention compresses history into a fixed-capacity buffer, leading to catastrophic key collisions. This project pivots to **Adaptive Attention Computation**: dynamically allocating feature capacity ($r \in \{64, 256, 1024\}$) based on input difficulty and sequence demands.

---

## 1. Decoupled Sprint Architecture (Zero-Bottleneck Workflow)

To eliminate idle waiting times across all 6 members, work is partitioned using **Contract-First Engineering**. Component interfaces are frozen in Hour 0 so members develop in parallel against mock implementations until a unified 30-minute merge in Hour 7.5.

| Role | Member / Lead | Core Responsibilities | Primary Output | Workload |
| :--- | :--- | :--- | :--- | :--- |
| **Person 1** | **Theory Lead** | Formulate adaptive rank equations; define input difficulty metric; build standalone `AdaptiveRankAttention` kernel. | `adaptive_kernel.py` + LaTeX math specification | 9.0 Hours |
| **Person 2** | **Compute Lead** | Profile execution time, peak VRAM, and empirical scaling exponents on Kaggle T4 across $N = 1\text{K}$ to $64\text{K}$. | `01_T4_Efficiency.ipynb` + latency/VRAM curves | 9.5 Hours |
| **Person 3** | **Retrieval Lead** | Implement synthetic Passkey / Needle-In-A-Haystack suite; test Softmax, Linear, and FAVOR+ across depths. | `02_Passkey_Suite.ipynb` + 2D retrieval heatmaps | 9.0 Hours |
| **Person 4** | **Integration Lead**| Patch Qwen2.5-0.5B attention blocks using mock and adaptive kernels; test generation and prefill latency. | `patch_qwen.py` + end-to-end forward test script | 9.5 Hours |
| **Person 5** | **Baselines Lead** | Implement pure-PyTorch layer for Gated DeltaNet; benchmark throughput and state retention against linear attention. | `03_DeltaNet_T4.ipynb` + comparative metrics table | 8.5 Hours |
| **Person 6** | **Synthesis Lead** | Standardize CSV schema; build automated chart generator; construct master 7-slide deck and final PDF report. | Master 7-Slide Deck + Consolidated PDF Document | 9.5 Hours |

---

## 2. Hour-by-Hour Master Execution Schedule

| Hour Window | Person 1 (Theory) | Person 2 (Compute) | Person 3 (Retrieval) | Person 4 (Model Patch) | Person 5 (SOTA Baselines) | Person 6 (Deck & Lead) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0 – 1.0 h** | Lock tensor I/O contract & rank formula | Build T4 memory tracking harness | Write Passkey token injection harness | Verify Qwen2.5 baseline inference | Extract Gated DeltaNet recurrent math | Lock CSV schema & setup presentation repo |
| **1.0 – 3.0 h** | Derive truncation error bounds | Benchmark Softmax vs SDPA (1K–64K) | Run baseline Softmax retrieval grid | Build `MockAttention` monkey-patch | Write standalone DeltaNet PyTorch layer | Build automated plotting script with mock data |
| **3.0 – 5.5 h** | Implement `AdaptiveRank` kernel | Benchmark ReLU+1 Linear Attention | Run Linear Attention Passkey sweep | Validate patched model forward pass | Profile DeltaNet latency on Kaggle T4 | Draft slide narratives and speaker notes |
| **5.5 – 7.5 h** | Unit-test kernel for stability/NaNs | Fit scaling exponents ($\alpha$) to CSV | Run FAVOR+ ($r=64\text{–}1024$) sweeps | Benchmark prefill latency on Qwen2.5 | Profile state memory vs Linear attention | Ingest real CSVs; auto-render final charts |
| **7.5 – 8.5 h** | Hand kernel to P4; review math writeup | Clean Kaggle code & export plots | Export 2D heatmaps to presentation | Drop in P1's kernel into Qwen2.5 | Finalize baseline comparative table | Assemble master slide deck & executive report |
| **8.5 – 9.5 h** | Rehearse theory Q&A for presentation | Verify Kaggle reproduction scripts | Audit retrieval figures in slide deck | Push patched model codebase | Audit comparative SOTA claims | Lead presentation dry-run & deliverable sign-off |

---

## 3. Kaggle Free Tier (Tesla T4) Guardrails

1. **FlashAttention-2 Workaround:**  
   The Tesla T4 (SM 7.5 Turing architecture) cannot execute FlashAttention-2 (which requires Ampere+ compute capability 8.0+ and FP16/BF16).  
   *Action:* Use PyTorch SDPA with memory-efficient exact kernel:  
   `enable_flash=False, enable_mem_efficient=True`.
2. **Batch Size Constraint:**  
   Fix batch size to $B = 1$ for all tests where sequence length $N \ge 8,192$ to prevent GPU Out-of-Memory (OOM) errors.
3. **Model Footprint:**  
   Base model experiments are restricted to `Qwen/Qwen2.5-0.5B` (~1.0 GB VRAM in FP16), leaving $> 12\text{ GB}$ free for KV caches and long-context tensors.
4. **Session Persistence:**  
   For long evaluation sweeps ($N \ge 16,384$), submit notebooks via Kaggle *Save & Run All (Commit)* to avoid the 60-minute interactive idle timeout.

---

## 4. The 7-Slide Departmental Presentation Structure

| Slide | Title | Core Content & Empirical Evidence | Lead Speaker |
| :--- | :--- | :--- | :--- |
| **Slide 1** | **The Scaling Dilemma** | Softmax quadratic bottleneck ($O(N^2)$) vs linear attention reordering ($O(N)$). | Person 6 |
| **Slide 2** | **Initial Hypothesis** | Evaluating whether linear attention preserves retrieval quality under fair benchmarking. | Person 1 |
| **Slide 3** | **Experimental Design** | Controlled setup: Tesla T4, float32/fp16, $B=1$, $H=4$, $d_k=d_v=64$, sequence lengths $1\text{K}$ to $64\text{K}$. | Person 2 |
| **Slide 4** | **Empirical Findings** | Efficiency won ($9.7\text{ ms}$ vs $1,602\text{ ms}$), but retrieval collapsed ($14\%$ vs $100\%$ at $N=1,024$). | Person 3 |
| **Slide 5** | **The Real Research Gap** | Information bottleneck: Compressing histories into fixed-size buffer causes key collisions. | Person 1 |
| **Slide 6** | **Proposed Architecture** | Adaptive Attention Computation: Dynamic rank allocation ($r \in \{64, 256, 1024\}$) & hybrid routing. | Person 4 |
| **Slide 7** | **Research Roadmap** | 5-phase roadmap: SOTA baselines (DeltaNet) $\to$ Adaptive Kernel $\to$ Time-series testing (TiRex). | Person 5 / 6 |
