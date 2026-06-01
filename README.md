# heretic-abliteration-run

Reproduction of automated refusal-direction ablation ("abliteration") on
`Qwen/Qwen3-4B-Instruct-2507` using the [Heretic](https://github.com/p-e-w/heretic)
tool, run locally on an **Intel Arc iGPU** (no NVIDIA GPU).

This repo publishes the real driver code, run logs, and the Optuna study
checkpoint from the run so they can be referenced as evidence. It is a
documentation / reproducibility artifact — it does **not** redistribute any
model weights.

---

## Credits

This work is motivated and enabled by others' work, credited here up front:

- **Motivation / inspiration:** the NPR article on AI-safety concerns about
  open-weight models, featuring Iftach (CTO):
  <https://www.npr.org/2026/05/31/nx-s1-5816391/ai-safety-concerns-danger-open-weight-models-risks>
  This run was undertaken to concretely understand, first-hand, how easily
  refusal behavior can be ablated from an open-weight model — the exact
  concern the article raises.
- **Tooling:** the **Heretic** project by **[p-e-w](https://github.com/p-e-w/heretic)**.
  Heretic is the tool that performs the directional ablation and the Optuna
  search. It is **used and credited here for inspiration.** All of the heavy
  lifting (the ablation method, the Optuna objective, the LoRA application) is
  Heretic's; this repo only adds a small non-interactive driver and the logs.
- **Method origins:** the abliteration method derives from the paper
  *"Refusal in Language Models Is Mediated by a Single Direction"*
  ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)), and Heretic also
  cites the "projected abliteration" refinement (grimjim).

> Heretic is licensed under **AGPL-3.0** (its own, separate license — see its
> repository). The files **in this repo** (the driver, logs, README, etc.) are
> the author's own artifacts and are MIT-licensed; see [`LICENSE`](LICENSE) and
> the NOTICE within it.

---

## Environment

- **OS:** Windows 11
- **Python:** 3.13.13
- **GPU:** **No NVIDIA GPU.** Intel **Arc 140V** iGPU, 16 GB shared LPDDR5.
- **Torch:** Heretic's deps initially pulled in `torch 2.12.0+cpu`. We replaced
  it with the **Intel XPU build `torch 2.12.0+xpu`** (which bundles the Intel
  oneAPI runtime). After the swap, `torch.xpu.is_available()` → **True**, and
  Heretic auto-detected `XPU 0: Intel(R) Arc(TM) 140V GPU (16GB)`.

> **Note on "CPU-only mode":** although the initial request mentioned running
> CPU-only, the actual trials ran on the **Arc XPU**. CPU-only abliteration was
> measured to be infeasible (days-long per study) and was deliberately avoided.

## Install

```bash
pip install -U heretic-llm
```

This pulls `heretic-llm 1.3.0` plus its dependency stack:
`transformers 5.9.0`, `accelerate 1.13`, `optuna 4.9`, `peft 0.19`, `lm-eval`,
`bitsandbytes 0.49.2`, `datasets`.

### Dependency bug we hit (and the fix)

Heretic pins `kernels~=0.13`, but `transformers 5.9.0` requires `kernels<0.13`.
pip's resolver landed on **`kernels 0.15.1`**, which crashed at import:

```
ValueError: Either a revision or a version must be specified
```

The traceback originates in `transformers/integrations/hub_kernels.py`, which
constructs a `LayerRepository(...)` **without a revision**; the error is raised
in `kernels/layer/layer.py:76`. With the wrong `kernels` version, the import
chain blows up before Heretic ever runs.

**Fix — pin kernels into the compatible window:**

```bash
pip install "kernels<0.13,>=0.12"   # -> kernels 0.12.3
```

After this, the `transformers` → `kernels` import chain is clean.

## Run

```bash
heretic --model Qwen/Qwen3-4B-Instruct-2507 --n-trials 50 --n-startup-trials 25
```

> **Heretic CLI quirk:** if you pass the model as a *trailing positional*
> argument, Heretic auto-inserts `--model` before the **last** token on the
> command line, which swallows the final flag's value. Always pass the model
> with an **explicit `--model`** (as above) to avoid this.

---

## Model facts

- `Qwen/Qwen3-4B-Instruct-2507`, **36 transformer layers**.
- Abliterable components: `attn.o_proj` (**36 modules**) + `mlp.down_proj` (**36 modules**).
- LoRA adapters target `down_proj` & `o_proj`.
- Loaded in **fp16**: ~**7.5 GB** XPU memory, ~**10.8 GB** system RAM.

## Datasets

| Role | Dataset | Train split | Eval split |
|---|---|---|---|
| harmless ("good") | `mlabonne/harmless_alpaca` | `train[:400]` | `test[:100]` |
| harmful ("bad") | `mlabonne/harmful_behaviors` | `train[:400]` | `test[:100]` |

## Throughput / batch-size sweep

Heretic sweeps batch size before the study to pick the fastest that fits. On
the Arc 140V:

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

Chosen batch size: **128**. ~**54 s/trial**; 50 trials ≈ **44 min**.

One minor XPU→CPU fallback warning was emitted: `torch.linalg.qr` falls back to
CPU during full row-normalization. It is minor and does not affect results.

---

## Method (concise)

Directional ablation works as follows:

1. For each layer, derive a **"refusal direction"** from the difference between
   residual-stream activations on **harmful** vs **harmless** prompts.
2. **Orthogonalize** the direction and **subtract** it from the model's
   computation, applied via a **LoRA adapter** (rather than baking weights).
3. **Optuna** drives a **multivariate-TPE** search that *jointly minimizes two
   objectives*: **KL divergence** from the base model (capability preservation)
   and **refusal count** (safety removal). A trial that lowers refusals but
   blows up KL is dominated.

Per-trial parameters Optuna tunes: `direction_scope` (global / per-layer),
`direction_index`, and per-component `max_weight`, `max_weight_position`,
`min_weight`, `min_weight_distance`. Fixed for this run:
`row_normalization = FULL`, `orthogonalize_direction = True`.

References: *"Refusal in Language Models Is Mediated by a Single Direction"*
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) and the "projected
abliteration" idea (grimjim) that Heretic cites.

---

## Results — first 50 trials

Initial state: **100/100 refusals**. (Heretic's rule of thumb: **KL < 0.5**
means the model's capabilities remain intact.)

Pareto front of the first 50 trials (best trade-offs of refusals vs KL):

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

**Summary:** refusals dropped from **100/100 to as low as 17/100 (83% removed)**
while the model stayed intact (all listed trials have KL well under 0.5).

Trials **51–100** (all TPE-guided, no further startup randomness) were launched
via [`continue_50.py`](continue_50.py) by resuming the same Optuna study
checkpoint. **At time of writing, run 2 is still in progress** (see
[`logs/run2_trials_51_100.log`](logs/run2_trials_51_100.log) — it was at trial
63 of 100 when this README was committed).

See [`results.md`](results.md) for the tables in one place, and
[`logs/`](logs/) for the raw run output.

---

## The background / TTY gotcha + workaround

Heretic's **final trial-picker is interactive** (it uses
`questionary` / `prompt_toolkit`). In a headless/background process there is no
console, so it crashes with:

```
NoConsoleScreenBufferError: No Windows console found
```

This blocks the "run additional trials" path — which is exactly the path you
need to resume a finished study for trials 51–100 unattended.

**Workaround — [`continue_50.py`](continue_50.py):** a small non-interactive
driver that monkeypatches Heretic's two prompt helpers
(`heretic.main.prompt_select` and `heretic.main.prompt_text`) to auto-answer the
menu:

1. *"How would you like to proceed?"* → **continue** (load the finished study)
2. *"Which trial …?"* → **continue** (→ "Run additional trials")
3. *"How many additional trials?"* → **"50"**
4. *"Which trial …?"* (second time, after the +50 run) → **""** (exit cleanly)

Because it resumes the same Optuna **`JournalStorage`** checkpoint, trials
51–100 continue the *same* study (TPE keeps its history) and run fully
unattended. This is the genuinely useful finding: you can resume Heretic's
study headlessly without touching its interactive menu.

---

## What's in this repo

```
README.md                                  this file
results.md                                 Pareto + throughput tables
SETUP.md                                   exact reproducible setup steps
continue_50.py                             the non-interactive resume driver
LICENSE                                    MIT (+ NOTICE re: Heretic's AGPL-3.0)
logs/run1_first50_trials.log               log of the first 50-trial run (UTF-8)
logs/run2_trials_51_100.log                log of trials 51-100 (in progress)
checkpoints/Qwen--Qwen3-4B-Instruct-2507.jsonl   Optuna JournalStorage checkpoint
```

No model weights, no HuggingFace cache, no secrets are included.
