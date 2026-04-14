# MLEvolve (Generic Fork)

Fork of [MLEvolve](https://github.com/InternScience/MLEvolve) — an agentic ML system that automatically builds ML pipelines using Monte Carlo Graph Search (MCGS) with multi-agent collaboration.

The original system is designed for the [MLE-bench](https://github.com/openai/mle-bench) Kaggle competition benchmark. This fork adds a **no-submission mode** so you can point it at any dataset and let the search find the best pipeline, without needing mle-bench, a grading server, or a competition structure.

## What changed

- **`no_submission_mode`** config flag — when `True`, all submission-specific logic is bypassed:
  - No `submission.csv` generation, no grading server, no format validation
  - LLM agents are instructed to do train/val splits and optimize the validation metric only
  - Best solution is saved as `best_solution/solution.py` with its metric
- **macOS support** — CPU affinity calls (`sched_setaffinity`) are skipped on platforms that don't support them
- **Coldstart graceful fallback** — unknown `exp_id` values no longer crash the coldstart module

All original MLE-bench functionality is preserved when `no_submission_mode` is `False` (the default).

## Setup

```bash
pip install --no-deps -r requirements_generic.txt
```

Configure `config/config.yaml` with your LLM API credentials:

```yaml
agent:
  code:
    model: gemini-3-pro-preview
    base_url: "https://generativelanguage.googleapis.com"
    api_key: "your-api-key"
  feedback:
    model: gemini-3-pro-preview
    base_url: "https://generativelanguage.googleapis.com"
    api_key: "your-api-key"
```

## Usage

Point it at any dataset directory containing your CSV files:

```bash
python run.py \
  exp_id=my-task \
  data_dir=/path/to/your/data \
  goal="Predict the target column from the features. Choose an appropriate metric." \
  no_submission_mode=True \
  coldstart.use_coldstart=False \
  use_grading_server=False
```

The `data_dir` should contain your input files (e.g. `train.csv`). They will be available to the generated code under `./input/`.

Results are written to `./runs/<timestamp>/`:
- `best_solution/solution.py` — best pipeline code
- `best_solution/metric.txt` — its validation score
- `top_candidates/` — top-K solutions ranked by metric
- `logs/` — full search tree and execution logs

## Config tuning

Key settings in `config/config.yaml` for quick runs:

| Setting | Default | Description |
|---------|---------|-------------|
| `agent.steps` | 10 | Total search iterations |
| `agent.time_limit` | 7200 | Wall-clock limit (seconds) |
| `agent.initial_drafts` | 2 | Independent initial solutions before search |
| `exec.timeout` | 1800 | Max time per code execution (seconds) |
| `agent.search.parallel_search_num` | 1 | Parallel search branches |
| `agent.use_global_memory` | False | Enable experience-driven memory (needs sentence-transformers) |

## Credits

Built on [MLEvolve](https://github.com/InternScience/MLEvolve) by InternScience. See the original repo for the full paper, MLE-bench results, and citation info.
