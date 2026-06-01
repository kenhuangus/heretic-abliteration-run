# SETUP — reproducible steps

Exact steps used to reproduce this run on Windows 11, Python 3.13.13, with an
Intel Arc 140V iGPU and **no NVIDIA GPU**.

## 0. Prerequisites

- Python 3.13.x on PATH (`python --version` → 3.13.13).
- An up-to-date `pip` (`python -m pip install -U pip`).
- Intel Arc GPU with current Intel graphics drivers (for the XPU torch build).

## 1. Install Heretic

```bash
pip install -U heretic-llm
```

Resulting key versions: `heretic-llm 1.3.0`, `transformers 5.9.0`,
`accelerate 1.13`, `optuna 4.9`, `peft 0.19`, `lm-eval`, `bitsandbytes 0.49.2`,
`datasets`.

## 2. Swap CPU torch for the Intel XPU build

Heretic's deps pull in `torch 2.12.0+cpu`. CPU-only abliteration is infeasible
(days-long). Replace torch with the Intel XPU build (bundles Intel oneAPI
runtime):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/xpu
```

Verify the XPU is visible:

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
# -> 2.12.0+xpu
# -> True
```

Heretic should then detect `XPU 0: Intel(R) Arc(TM) 140V GPU (16GB)`.

## 3. Fix the `kernels` version conflict

`heretic-llm` pins `kernels~=0.13`, but `transformers 5.9.0` requires
`kernels<0.13`. pip's resolver may install `kernels 0.15.1`, which crashes at
import with:

```
ValueError: Either a revision or a version must be specified
```

(raised in `kernels/layer/layer.py:76`, triggered by
`transformers/integrations/hub_kernels.py` building a `LayerRepository` without
a revision). Pin kernels into the working window:

```bash
pip install "kernels<0.13,>=0.12"   # -> kernels 0.12.3
```

## 4. Run the study (trials 1–50)

```bash
heretic --model Qwen/Qwen3-4B-Instruct-2507 --n-trials 50 --n-startup-trials 25
```

> Always pass the model with an **explicit `--model`**. If you pass it as a
> trailing positional, Heretic inserts `--model` before the *last* CLI token and
> swallows the final flag's value.

Heretic will sweep batch size (it chose **128** on the Arc 140V), load the model
in fp16 (~7.5 GB XPU / ~10.8 GB RAM), and run the Optuna multivariate-TPE study
minimizing (KL divergence, refusal count). It writes an Optuna `JournalStorage`
checkpoint under `checkpoints/`.

## 5. Resume for trials 51–100 (headless)

Heretic's "run additional trials" path is only reachable through its
interactive console menu, which crashes in a background/headless process
(`NoConsoleScreenBufferError: No Windows console found`). Use the included
driver, which monkeypatches the prompt helpers and resumes the same checkpoint:

```bash
python continue_50.py
```

It auto-answers: proceed → continue (load finished study) → "Run additional
trials" → "50" → exit. Output goes to `heretic2.log`.

## Notes

- Do **not** commit the HuggingFace cache (`~/.cache/huggingface`) or any
  `*.safetensors` weights — see `.gitignore`.
- The Optuna checkpoint (~220 KB) is small and is committed here as run evidence.
