# Linear Attention Experimentation & Boundary Analysis

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Tesla%20T4%20(Kaggle%2FColab)-orange.svg)](https://cloud.google.com/gpu)
[![Research Team](https://img.shields.io/badge/Team-Team%203-blue.svg)](#)

Empirical efficiency, output deviation, and key-value retrieval quality analysis comparing **Standard Softmax Attention**, **Flash / Memory-Efficient SDPA**, **Local Block-Sparse Attention**, and **Kernelized Linear Attention** (ReLU+1 and FAVOR+) across sequence lengths up to $N = 65,536$.

---

## 📌 Executive Summary

### 1. The Scaling Dilemma & Empirical Findings
* **Computational Scaling**: Kernelized linear attention ($\text{ReLU}(x)+1$) demonstrates true $O(N)$ scaling on Tesla T4 GPUs—achieving **9.7 ms** at $N=65,536$ compared to **1,602 ms** for PyTorch SDPA, with linear memory consumption (257 MB vs. OOM for naive softmax at $N \ge 32,768$).
* **The Retrieval Collapse**: Under fair synthetic key-value retrieval testing (16 classes, chance = 6.25%), basic linear attention collapsed to **14% accuracy at $N=1,024$** (while softmax maintained 100%), exhibiting a 0.79 relative output error. Even scaling feature dimension $16\times$ ($r=1024$) only recovered 27% accuracy.
* **The Engineering Control**: Flash / memory-efficient SDPA remains exact ($\Delta = 0.00$), requires $O(N)$ transient memory, and does not OOM through $N=65,536$.
* **Research Pivot**: Moving from fixed-rank kernelized attention to **Adaptive Attention Computation** (dynamically allocating feature rank $r \in \{64, 256, 1024\}$ based on input sequence difficulty bounds).

---

## 📂 Repository Organization

```text
Linear_Attention_Experimentation/
├── docs/
│   ├── Adaptive_Attention_Project_Plan_and_Technical_Specification.pdf
│   ├── Linear_Attention_Research_Proposal_Revised.docx
│   ├── Mathematically_Correct_Linear_Attention_Hypothesis.docx
│   └── Team_3_Linear_Attention_PDCA_Workbook.docx
├── notebooks/
│   └── Linear_Attention_Colab_Benchmark.ipynb
├── presentation/
│   └── Team3_Attention_Results.pptx
├── .gitignore
└── README.md
```

### Document Index & Curation Notes

| File | Type | Description |
| :--- | :--- | :--- |
| [`docs/Adaptive_Attention_Project_Plan_and_Technical_Specification.pdf`](docs/Adaptive_Attention_Project_Plan_and_Technical_Specification.pdf) | Specification | 3-page master project plan, decoupled sprint architecture (contract-first engineering across 6 leads), Kaggle T4 guardrails, and 7-slide layout. |
| [`docs/Linear_Attention_Research_Proposal_Revised.docx`](docs/Linear_Attention_Research_Proposal_Revised.docx) | Research Proposal | Pre-registered, revised research proposal (*Kernelized Linear Attention vs. Softmax Attention: An Efficiency–Quality Boundary Analysis*), superseding unrevised early drafts. |
| [`docs/Mathematically_Correct_Linear_Attention_Hypothesis.docx`](docs/Mathematically_Correct_Linear_Attention_Hypothesis.docx) | Theoretical Framework | Error-Controlled Adaptive-Rank Linear Attention mathematical formulation, error bounds ($B_{X, r} \le \delta$), and associative state accumulation. |
| [`docs/Team_3_Linear_Attention_PDCA_Workbook.docx`](docs/Team_3_Linear_Attention_PDCA_Workbook.docx) | Research Workbook | Team 3 Plan-Do-Check-Act (PDCA) experimental protocol, worked examples, controls, and acceptance criteria. |
| [`notebooks/Linear_Attention_Colab_Benchmark.ipynb`](notebooks/Linear_Attention_Colab_Benchmark.ipynb) | Jupyter Notebook | Complete Colab/Kaggle benchmark notebook containing timing harness, memory tracking, scaling slopes, FAVOR+ sweeps, and retrieval evaluations. |
| [`presentation/Team3_Attention_Results.pptx`](presentation/Team3_Attention_Results.pptx) | Slide Deck | 7-slide departmental presentation deck summarizing latency curves, memory footprints, retrieval failure points, and architectural conclusions. |

*(Note: Duplicate/superseded initial proposal draft `Linear_Attention_Research_Team_3.docx` has been omitted to maintain a clean, non-redundant repository).*

---

## 🔬 Experimental Methodology & Baselines

All benchmarks are evaluated with identical input tensors:
* **Dimensions**: $B = 1, H = 4, d_k = d_v = r = 64$
* **Sequence Lengths**: $N \in \{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536\}$
* **Hardware**: NVIDIA Tesla T4 (SM 7.5), float32 precision, PyTorch 2.x

| Method | Formulation | Exact Softmax? | Time Complexity | Memory Complexity |
| :--- | :--- | :---: | :---: | :---: |
| **Softmax (Naive)** | $\text{softmax}(QK^T / \sqrt{d_k})V$ | Yes | $O(N^2 d)$ | $O(N^2)$ |
| **Flash / SDPA** | PyTorch Memory-Efficient SDPA | Yes | $O(N^2 d)$ | $O(N)$ |
| **Sparse Attention** | Local window ($w = 64$) | No (local only) | $O(N w d)$ | $O(N w)$ |
| **Linear Attention** | Reordered $\phi(Q)(\phi(K)^T V)$ with $\phi(x)=\text{ReLU}(x)+1$ | No (kernelized) | $O(N r d_v)$ | $O(r d_v)$ state |

---

## 🚀 Running the Benchmarks

To replicate or extend the benchmarks locally or on Kaggle/Google Colab:

```bash
# Clone the repository
git clone https://github.com/Jaiadithya71/Linear_Attention_Experimentation.git
cd Linear_Attention_Experimentation

# Launch the benchmark notebook
jupyter lab notebooks/Linear_Attention_Colab_Benchmark.ipynb
```

> **Note on Hardware (Tesla T4)**: FlashAttention-2 requires CUDA compute capability $\ge 8.0$ (Ampere+). On Tesla T4 (compute capability 7.5), PyTorch SDPA automatically runs via the memory-efficient C++ backend (`enable_flash=False, enable_mem_efficient=True`).
