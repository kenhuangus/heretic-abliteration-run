"""
Non-interactive driver to run 50 MORE heretic trials (51-100) by resuming the
existing Optuna study checkpoint. Heretic's "run additional trials" path is only
reachable through its interactive console menu, which cannot run in a background
process (no Windows console -> NoConsoleScreenBufferError). We monkeypatch the
two prompt helpers to auto-answer, then call heretic's normal run().

Flow we drive:
  1. startup "How would you like to proceed?" -> "continue" (load finished study)
  2. results menu "Which trial...?"            -> "continue" (run additional)
  3. "How many additional trials?"             -> "50"
  4. results menu "Which trial...?" (2nd time) -> "" (exit cleanly; no export here)
"""
import sys

# Route all output to a clean UTF-8 log BEFORE importing heretic, so rich binds to it.
_log = open(r'C:\Users\kenhu\heretic-run\heretic2.log', 'w', encoding='utf-8', buffering=1)
sys.stdout = _log
sys.stderr = _log

# Heretic reads its config from argv. --model just needs to be present and valid;
# on "continue" the real settings (n_trials=50, batch_size=128, seed, n_startup=25)
# are restored from the checkpoint's stored JSON.
sys.argv = ['heretic', '--model', 'Qwen/Qwen3-4B-Instruct-2507']

import heretic.main as M

_which_trial = {'n': 0}

def fake_prompt_select(message, choices=None, *a, **k):
    m = str(message).lower()
    if 'how would you like to proceed' in m:
        return 'continue'          # show results from the previous (finished) run
    if 'which trial' in m:
        _which_trial['n'] += 1
        if _which_trial['n'] == 1:
            return 'continue'      # -> "Run additional trials"
        return ''                  # 2nd time (after the +50 run): Exit program
    return ''                      # any other menu -> exit

def fake_prompt_text(message, *a, **k):
    return '50'                    # number of additional trials

M.prompt_select = fake_prompt_select
M.prompt_text = fake_prompt_text

M.run()

_log.write('\n===== continue_50.py: run() returned cleanly =====\n')
_log.flush()
