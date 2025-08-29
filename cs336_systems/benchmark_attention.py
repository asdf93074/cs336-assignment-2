from collections import defaultdict
import math
import timeit
import os
import sys
sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cs336-basics"))
)
from itertools import product

import torch
from einops import einsum

from cs336_basics.model import scaled_dot_product_attention 
from cs336_basics.nn_utils import softmax

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = K.shape[-1]

    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        qk = torch.where(mask, qk, float("-inf")) 

    qk_softmax = softmax(qk, dim=-1)
    return einsum(qk_softmax, V, "... queries keys, ... keys d_k -> ... queries d_k")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    compile = sys.argv[-1]
    if compile == "compile":
        print("'compile' passed. Using torch.compile.")
        attention = torch.compile(attention)

    device = torch.device("cuda")

    warmup_steps     = 10
    n                = 100

    batch_size       = 8
    d_model_sizes    = [16, 32, 64, 128]
    seq_length_sizes = [256, 1024, 4096, 8192, 16384]

    measurements = defaultdict(list) 

    for d_model, seq_length in product(d_model_sizes, seq_length_sizes):
        try:
            torch.cuda.empty_cache()

            Q = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
            K = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
            V = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)

            mask    = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))

            print(f"d_model: {d_model}")
            print(f"seq_length: {seq_length}")
            for _ in  range(warmup_steps):
                attention(Q, K, V, mask)

            torch.cuda.synchronize()

            for _ in range(n):
                start = timeit.default_timer()

                a = attention(Q, K, V, mask)

                end = timeit.default_timer()
                torch.cuda.synchronize()
                
                mem = torch.cuda.memory_reserved()

                (a**2).mean().backward()
                end_backw = timeit.default_timer()
                torch.cuda.synchronize()

                measurements[f"{d_model}_{seq_length}"].append([end - start, end_backw - end, mem])

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory on d_model: {d_model} and seq_length: {seq_length}")

    forward_pass_results = defaultdict(float)    
    print("d_model & Sequence Length & Forward Pass Mean (ms) & Backward Pass Mean (ms) & Backward Pass Mem. Usage (MB)")
    for k, measurements in measurements.items():
        f_mean = sum(map(lambda x: x[0], measurements)) / len(measurements)
        b_mean = sum(map(lambda x: x[1], measurements)) / len(measurements)
        mem_mean = sum(map(lambda x: x[2], measurements)) / len(measurements)

        f_mean *= 1000
        b_mean *= 1000

        d_model, seq_length = k.split("_")

        print(f"{d_model} & {seq_length} & {f_mean:.4f} & {b_mean:.4f} & {(mem_mean / (1024**2)):.4f}") 
