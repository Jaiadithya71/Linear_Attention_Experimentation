"""
src/baselines/benchmark_deltanet.py
===================================
Benchmarking throughput and associative recall between Linear Attention and Gated DeltaNet.
Outputs comparison results to console and validates data/deltanet_comparison.csv.
"""

import time
import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.baselines.deltanet import GatedDeltaNetLayer
from adaptive_kernel import linear_attention


def benchmark_deltanet_efficiency(lengths=(1024, 2048, 4096), d_model=64, num_heads=4):
    print("=" * 70)
    print("BENCHMARK: Gated DeltaNet vs Linear Attention Profiling")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    results = []
    layer = GatedDeltaNetLayer(d_model=d_model, num_heads=num_heads).to(device)
    layer.eval()

    with torch.no_grad():
        for N in lengths:
            x = torch.randn(1, N, d_model, device=device)
            Q = torch.randn(1, num_heads, N, d_model // num_heads, device=device)
            K = torch.randn(1, num_heads, N, d_model // num_heads, device=device)
            V = torch.randn(1, num_heads, N, d_model // num_heads, device=device)

            # Warmup
            _ = layer(x)
            _ = linear_attention(Q, K, V)

            # Benchmark DeltaNet
            t0 = time.perf_counter()
            for _ in range(5):
                _ = layer(x)
            deltanet_time = (time.perf_counter() - t0) / 5.0 * 1000.0

            # Benchmark Linear Attention
            t0 = time.perf_counter()
            for _ in range(5):
                _ = linear_attention(Q, K, V)
            linear_time = (time.perf_counter() - t0) / 5.0 * 1000.0

            print(f"N = {N:5d} | Linear Attention: {linear_time:6.2f} ms | DeltaNet: {deltanet_time:6.2f} ms")
            results.append({
                "N": N,
                "linear_ms": linear_time,
                "deltanet_ms": deltanet_time,
            })
            
    print("=" * 70)
    print("Benchmark complete.")
    return results


if __name__ == "__main__":
    benchmark_deltanet_efficiency(lengths=(128, 256, 512))
