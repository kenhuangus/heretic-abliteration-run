# Results

Run: Heretic 1.3.0 abliteration of `Qwen/Qwen3-4B-Instruct-2507` on an Intel
Arc 140V iGPU. Initial refusals: **100/100**. Heretic's rule of thumb:
**KL divergence < 0.5** means capabilities remain intact.

## Pareto front — first 50 trials

The best refusals-vs-KL trade-offs found in the first 50-trial study
(`heretic --model Qwen/Qwen3-4B-Instruct-2507 --n-trials 50 --n-startup-trials 25`):

| Trial | Refusals/100 | KL divergence |
|---|---|---|
| 37 | 17 | 0.1534 |
| 45 | 20 | 0.1303 |
| 44 | 22 | 0.1578 |
| 31 | 27 | 0.0830 |
| 47 | 29 | 0.0863 |
| 16 | 31 | 0.1131 |
| 35 | 32 | 0.0974 |
| 33 | 34 | 0.1008 |

Refusals dropped from **100/100 to as low as 17/100 (83% removed)** with the
model intact (all KL values well under 0.5).

Trials 51–100 (TPE-guided) were resumed from the same Optuna checkpoint via
`continue_50.py`. **At time of writing run 2 is still in progress** (it was at
trial 63 of 100 when this was committed — see `logs/run2_trials_51_100.log`).

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
