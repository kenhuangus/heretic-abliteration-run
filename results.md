# Results

Run: Heretic 1.3.0 abliteration of `Qwen/Qwen3-4B-Instruct-2507` on an Intel
Arc 140V iGPU. Initial refusals: **100/100**. Heretic's rule of thumb:
**KL divergence < 0.5** means capabilities remain intact.

Status: **complete (100 trials total).** Trials 1–50 ran first
(`--n-trials 50 --n-startup-trials 25`); trials 51–100 — all TPE-guided — were
resumed from the same Optuna `JournalStorage` checkpoint via `continue_50.py`,
which returned cleanly ("Optimization finished!"). The committed checkpoint
`checkpoints/Qwen--Qwen3-4B-Instruct-2507.jsonl` is the authoritative record of
all 100 trials.

## Global Pareto front — all 100 trials

The Pareto-optimal refusals-vs-KL trade-offs across the full 100-trial study:

| Trial | Refusals/100 | KL divergence |
|---|---|---|
| 83 | 9 | 0.0630 |
| 87 | 12 | 0.0533 |
| 41 | 53 | 0.0429 |
| 52 | 60 | 0.0420 |
| 97 | 71 | 0.0350 |
| 3 | 86 | 0.0293 |
| 99 | 89 | 0.0105 |
| 27 | 98 | 0.0088 |
| 50 | 99 | 0.0070 |
| 1 | 100 | 0.0019 |

**Headline:** with 100 trials the best result improved to **9/100 refusals
(91% removed) at KL 0.063 (trial 83)** — strictly better than the 50-trial best
(17/100 at KL 0.153). The additional 50 TPE-guided trials roughly **halved both
refusals and KL**. The front spans the full trade-off curve, from aggressive
ablation (trial 83) down to near-zero KL with refusals untouched (trial 1:
100/100 at KL 0.0019).

### For reference — best of the first 50 trials

| Trial | Refusals/100 | KL divergence |
|---|---|---|
| 37 | 17 | 0.1534 |
| 31 | 27 | 0.0830 |

The 50-trial best was 17/100 at KL 0.153 (trial 37); 100 trials beat it on both
axes.

## Batch-size / throughput sweep

Measured on the Arc 140V before the study; Heretic picks the fastest batch size
that fits.

| Batch size | Throughput (tokens/s) |
|---|---|
| 1 | 5 |
| 2 | 12 |
| 4 | 21 |
| 8 | 36 |
| 16 | 77 |
| 32 | 139 |
| 64 | 200 |
| 128 | **231** (chosen) |

Chosen batch size: **128**. Approximately **54 s/trial**; 50 trials ≈ **44 min**.

A single minor XPU→CPU fallback warning was emitted (`torch.linalg.qr` falls
back to CPU during full row-normalization); it does not affect correctness.
