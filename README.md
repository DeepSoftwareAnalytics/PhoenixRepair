# PhoenixRepair

PhoenixRepair localizes an issue into one or more complete repair plans. With one plan, the author generates the final patch directly. With several plans, candidate authors share half the generation budget. One same-model summary guides a fresh author using the remaining calls. Localization has its own budget.

## Run PhoenixRepair

From the repository root, prepare a dedicated Python 3.11+ environment on Linux x86-64 with Docker and the task images available.

```bash
cd phoenixrepair
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
python -m phoenix.runtime prepare --output "$PWD/.runtime"
export PHOENIX_RUNTIME="$PWD/.runtime"
```

Supply `PHOENIX_API_KEY` through your environment. Set `API_BASE` for a custom OpenAI-compatible endpoint. Provide a JSON or JSONL task manifest with a string value for each of `instance_id`, `problem_statement`, `base_commit`, `repo_name`, and `image_name`.

```bash
phoenixrepair --model glm-5.2 --instances /path/to/tasks.jsonl --dry-run
phoenixrepair --model glm-5.2 --instances /path/to/tasks.jsonl --output runs/glm-repair
```

Each completed task writes `PhoenixRepair.patch` under `<output>/<model>/<instance_id>/`. Use `--resume` with the same inputs to continue an interrupted run. The GLM-5.2 and Qwen3.7-Plus generation profiles disable thinking and use a 32,768-token output limit. Model request defaults live in `phoenixrepair/phoenix/settings.py` and per-model run budgets in `phoenixrepair/configs/models/`.

## Published results for the original method

### Pass@1 on SWE-bench Verified

| Method | LLM | Resolved (%) |
|--------|-----|--------------|
| Mini-SWE-agent | DeepSeek-V3.2 | 61.0% |
| Live-SWE-agent | DeepSeek-V3.2 | 63.8% |
| SWE-Search | DeepSeek-V3.2 | 65.4% |
| Agentless | DeepSeek-V3.2 | 67.0% |
| Trae-agent | DeepSeek-V3.2 | 67.8% |
| SWE-agent | DeepSeek-V3.2 | 69.4% |
| Moatless-tools | Claude-4-Sonnet | 70.8% |
| **PhoenixRepair** | **DeepSeek-V3.2** | **74.4%** (+5.0%) |

### Fault Localization Accuracy

| Method | LLM | File Acc@1 | Module Acc@1 | Function Acc@1 |
|--------|-----|------------|--------------|----------------|
| SWE-agent | DeepSeek-V3.2 | 83.04% | 74.22% | 63.42% |
| **PhoenixRepair** | **DeepSeek-V3.2** | **85.64%** (+2.60%) | **76.82%** (+2.60%) | **66.64%** (+3.22%) |
