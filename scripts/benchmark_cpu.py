import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-len", type=int, default=8)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--mode",
    choices=["forward", "forward_backward", "train"],
    default="train",
)
args = parser.parse_args()

if min(args.batch_size, args.seq_len, args.threads) < 1 or args.warmup < 0 or args.steps < 2:
    parser.error("Require positive sizes/threads, warmup >= 0, steps >= 2.")

torch.set_num_threads(args.threads)

torch.manual_seed(0)

model = BasicsTransformerLM(
    vocab_size=32,
    context_length=args.seq_len,
    d_model=32,
    num_layers=1,
    num_heads=4,
    d_ff=64,
    rope_theta=10000.0,
).cpu()
model.train()

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

tokens = torch.randint(0, 32, (args.batch_size, args.seq_len + 1))
x = tokens[:, :-1]
y = tokens[:, 1:]

from time import perf_counter
from statistics import mean, stdev


def run_step():
    if args.mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    if args.mode == "forward":
        return

    loss = cross_entropy(logits, y)
    loss.backward()

    if args.mode == "train":
        optimizer.step()


def measure_step_ms():
    start = perf_counter()  # TODO 1: Read the timer.
    run_step()
    elapsed = perf_counter() - start  # TODO 2: Compute elapsed seconds.
    return elapsed * 1000


warmup_steps = args.warmup
measurement_steps = args.steps

for _ in range(warmup_steps):
    run_step()

times_ms = [measure_step_ms() for _ in range(measurement_steps)]

print(
    f"device=cpu, threads={torch.get_num_threads()}, "
    f"dtype={next(model.parameters()).dtype}, "
    f"B={x.shape[0]}, T={x.shape[1]}"
)
print(f"warmup={warmup_steps}, repeats={measurement_steps}")
print(f"mean step time: {mean(times_ms):.3f} ms")
print(f"std step time: {stdev(times_ms):.3f} ms")
print(f"min step time: {min(times_ms):.3f} ms")
print(f"max step time: {max(times_ms):.3f} ms")

timing_scopes = {
    "forward": "forward_grad_enabled",
    "forward_backward": "zero_grad+forward+loss+backward",
    "train": "zero_grad+forward+loss+backward+optimizer_step",
}

print(f"mode: {args.mode}")

result = {
    "device": "cpu",
    "torch_version": torch.__version__,
    "dtype": str(next(model.parameters()).dtype),
    "threads": torch.get_num_threads(),
    "seed": 0,
    "model_config": model.config,
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "batch_size": args.batch_size,
    "seq_len": args.seq_len,
    "optimizer": "cs336_basics.AdamW",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "warmup": warmup_steps,
    "repeats": measurement_steps,
    "mode": args.mode,
    "timing_scope": timing_scopes[args.mode],
    "optimizer_step_included": args.mode == "train",
    "mean_ms": mean(times_ms),
    "std_ms": stdev(times_ms),
    "min_ms": min(times_ms),
    "max_ms": max(times_ms),
    "tokens_per_second": (args.batch_size * args.seq_len / (mean(times_ms) / 1000)),
    "times_ms": times_ms,
}

if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}")
