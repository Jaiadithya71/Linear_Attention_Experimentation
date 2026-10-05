# Theoretical Presentation Narrative & Speaker Notes

**Deliverable for Person 1 (Theory Lead) & Person 6 (Synthesis Lead)**  
*Target Deck: Team 3 Attention Results Presentation*

---

## Slide 1: Title & Strategic Research Framing

### Speaker Narrative:
"Good morning, everyone. Today Team 3 presents an empirical and theoretical boundary analysis comparing Linear, Softmax, Sparse, and Flash Attention. 

Our investigation originated from a compelling premise in the literature: that linear attention replaces the quadratic $O(N^2)$ memory and compute bottleneck of Transformers with an $O(N)$ recurrence while preserving representation capacity. 

Rather than taking this premise at face value, we subjected it to rigorous boundary testing across sequence lengths from $N=64$ to $65,536$ on standard Tesla T4 hardware. Our central finding is clear: while linear attention delivers an exceptional $164\times$ speedup at sequence length $65,536$, standard positive-kernel linear attention suffers catastrophic key-value retrieval collapse on needle-in-a-haystack tasks—dropping to $14.2\%$ accuracy at $N=1,024$. 

To address this fundamental dilemma, we developed an error-controlled adaptive rank framework that dynamically balances computational throughput with mathematical error guarantees."

---

## Slide 2: Mathematical Foundation & What We Compared

### Speaker Narrative:
"To make this comparison mathematically rigorous, we established a strictly controlled layer-level experimental testbed where all mechanisms received identical Query, Key, and Value inputs under shared dimensions: batch size 1, 4 heads, $d_k=64$, and $d_v=64$.

The fundamental divide lies in how the attention weights are computed:
1. **Standard Softmax Attention:** Computes the full $N \times N$ matrix $A = \text{softmax}(QK^T / \sqrt{d_k})$. This requires $O(N^2 d)$ time and materializes an $O(N^2)$ intermediate matrix that exhausts 16GB GPU memory at $N=32,768$.
2. **Flash / SDPA:** Computes the exact same mathematical softmax output, but fuses the operation into SRAM tiles, reducing memory footprint to $O(N)$ without altering a single bit of output precision.
3. **Local Sparse Attention:** Restricts attention to a local window $w=64$, achieving $O(N w d)$ complexity but discarding all long-range associations.
4. **Kernelized Linear Attention:** Replaces exponential pairwise scoring with an explicit feature map $\phi(x) = \text{ReLU}(x) + 1$. By leveraging the associative property of matrix multiplication, we compute $(QK^T)V$ as $Q(K^TV)$, updating a fixed-size $r \times d_v$ recurrent memory state."

---

## Slide 5: The Output Deviation & Error Certificate Bound

### Speaker Narrative:
"Why does linear attention fail on associative retrieval, and can we mathematically certify its accuracy before running it?

Slide 5 illustrates our core theoretical contribution: the **Sufficient Output Error Certificate**. 

When we approximate the softmax kernel with positive features of rank $r$, the element-wise kernel error is $\epsilon_r$. Through our proof in `theory_spec.md`, we demonstrate that the maximum output error $\|y_i^L - y_i^S\|_\infty$ is strictly bounded by:
$$\|y_i^L - y_i^S\|_\infty \le \frac{2 N \epsilon_r V_{\max}}{\min_i Z_i - N \epsilon_r}$$

To guarantee that the final attention vector stays within a user-defined error tolerance $\delta$, the required kernel error must satisfy:
$$\epsilon_r(X) \le \frac{\delta \min_i Z_i}{N (2 V_{\max} + \delta)}$$

The key insight is that $\min_i Z_i$ is directly governed by attention entropy:
- In diffuse, high-entropy contexts (general language modeling), $Z_i$ is large, error decays rapidly, and a low rank ($r=64$) provides full fidelity.
- In concentrated, low-entropy needle retrieval contexts, $Z_i$ is small, and the feature dimension required to preserve the sharp exponential peak scales with $O(N)$. 

This formalizes why static linear attention collapses on retrieval tasks and provides the analytical engine for our adaptive routing kernel."
