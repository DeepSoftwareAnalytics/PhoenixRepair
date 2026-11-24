# PhoenixRepair

PhoenixRepair localizes an issue into one or more complete repair plans. With one plan, the author generates the final patch directly. With several plans, candidate authors share half the generation budget. One same-model summary guides a fresh author using the remaining calls. Localization has its own budget.

## Run PhoenixRepair

From the repository root, prepare a dedicated Python 3.11+ environment on Linux x86-64 with Docker and the task images available.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
python -m phoenix.runtime prepare --output "$PWD/.runtime"
export PHOENIX_RUNTIME="$PWD/.runtime"
```

Supply `PHOENIX_API_KEY` through your environment. Set `API_BASE` for a custom OpenAI-compatible endpoint. Provide a JSON or JSONL task manifest with a string value for each of `instance_id`, `problem_statement`, `base_commit`, `repo_name`, and `image_name`.

```bash
./run.sh --model glm-5.2 --instances /path/to/tasks.jsonl --dry-run
./run.sh --model glm-5.2 --instances /path/to/tasks.jsonl --output runs/glm-repair
```

Each completed task writes `PhoenixRepair.patch` under `<output>/<model>/<instance_id>/`. Use `--resume` with the same inputs to continue an interrupted run. Model request defaults live in `phoenix/settings.py` and per-model run budgets in `configs/models/`.

## Inference profiles

Use the profile name with `--model`. The settings below apply to localization, patch generation and candidate comparison. Output limits are per response.

| Profile | Thinking | Temperature | Top-p | Context tokens | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `deepseek-v3.1` | Off | 0.6 | 0.95 | 131,072 | API default |
| `deepseek-v3.2` | Off | 1.0 | 0.95 | 131,072 | API default |
| `qwen3-coder-plus` | Off | 0.7 | 0.80 | 1,000,000 | 65,536 |
| `glm-4.7` | On | 0.7 | 1.00 | 202,752 | 131,072 |
| `minimax-m2.5` | On (native) | 1.0 | 0.95 | 204,800 | 32,768 |
| `glm-5.2` | Off | 1.0 | 1.00 | 1,048,576 | 131,072 |
| `qwen3.7-plus` | Off | 1.0 | 0.95 | 1,000,000 | 131,072 |
| `qwen3.7-plus-thinking` | On | 1.0 | 0.95 | 1,000,000 | 131,072 |
| `qwen-3.8-max` | Off | 1.0 | 0.95 | 1,000,000 | 131,072 |

Both DeepSeek profiles send `deepseek-chat` and omit the output-length parameter. To reproduce V3.1 (2025-08-21) or V3.2 (2025-12-01), set `API_BASE` to a deployment serving that release. The shared API identifier alone cannot pin a snapshot. Both Qwen-3.7 Plus profiles send `qwen3.7-plus` with the selected thinking flag.
