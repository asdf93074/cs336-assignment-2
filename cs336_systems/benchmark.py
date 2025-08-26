import argparse
from contextlib import nullcontext
from math import sqrt
import os
import sys

from torch.cuda import nvtx

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cs336-basics"))
)
from dataclasses import asdict, dataclass

import timeit
import torch

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW 

ProfileResults = list[float]


@dataclass
class TransformerParams:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int
    context_length: int
    rope_theta: int

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(
    model: torch.nn.Module,
    vocab_size: int,
    context_length: int,
    n: int,
    warm_up_steps: int = 0,
    forward_only: bool = True,
    mixed_precision: bool = False,
    device: torch.device | None = None,
) -> tuple[ProfileResults, ProfileResults]:
    """Profile a model's forward or forward+backward passes.

    Args:
        model (torch.nn.Module): PyTorch model to profile.
        n (int): Number of steps to profile.
        warm_up_steps (int): Number of steps to wait before profiling begins.
        forward_only (bool): If True, count only forward passes (one forward = one step).
            If False, count both forward and backward passes as one step.
    """
    batch_size = 4

    model = model.to(device)
    optim = AdamW(model.parameters())

    forw_results = []
    backw_results = []

    cm = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()

    for _ in range(warm_up_steps):
        inputs = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
                )
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        cuda_sync()

        with nvtx.range("forward pass warmup"):
            with cm:
                out = model(inputs)
                loss = cs336_basics.nn_utils.cross_entropy(out, targets)

        cuda_sync()

        forw_end = timeit.default_timer()

        with nvtx.range("backward pass warmup"):
            if not forward_only:
                loss.backward()
                optim.step()

        cuda_sync()

    for i in range(n):
        inputs = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
                )
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        cuda_sync()

        start = timeit.default_timer()

        with nvtx.range("forward pass"):
            with cm:
                out = model(inputs)
                loss = cs336_basics.nn_utils.cross_entropy(out, targets)

        cuda_sync()

        forw_end = timeit.default_timer()

        with nvtx.range("backward pass"):
            if not forward_only:
                loss.backward()
                optim.step()


        backw_end = timeit.default_timer()

        if i >= warm_up_steps:
            forw_results.append(forw_end - start)

            if not forward_only:
                backw_results.append(backw_end - forw_end)

    return forw_results, backw_results


def create_model(model_cls: type[torch.nn.Module], params: TransformerParams):
    return model_cls(**asdict(params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--rope-theta", type=int, default=10000)

    parser.add_argument("--warm-up-steps", type=int, default=5)
    parser.add_argument("--profiling-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    args = parser.parse_args()

    if args.forward_only:
        print("Profiling forward pass only.")
    else:
        print("Profiling forward+backward passes.")

    print(f"Warmup steps: {args.warm_up_steps}")
    print(f"Profiling steps: {args.profiling_steps}")

    device = torch.device("cuda")
    model_params = TransformerParams(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
    )
    model = create_model(BasicsTransformerLM, model_params)
    forw_results, backw_results = run(
        model,
        args.vocab_size,
        args.context_length,
        args.profiling_steps,
        args.warm_up_steps,
        args.forward_only,
        args.mixed_precision,
        device=device,
    )

    n = len(forw_results)
    mean = sum(forw_results) / n
    var = sum([(x - mean) ** 2 for x in forw_results]) / n
    std = sqrt(var)

    print("\nForward Pass:")
    print(f"Min: {min(forw_results):.6f}, Max: {max(forw_results):.6f}, Avg: {mean:.6f} Std: {std:.6f}")

    if not args.forward_only:
        n = len(backw_results)
        mean = sum(backw_results) / n
        var = sum([(x - mean) ** 2 for x in backw_results]) / n
        std = sqrt(var)

        print("\nBackward Pass:")
        print(
            f"Min: {min(backw_results):.6f}, Max: {max(backw_results):.6f}, Avg: {mean:.6f} Std: {std:.6f}"
        )
