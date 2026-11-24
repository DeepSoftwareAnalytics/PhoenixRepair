# PhoenixRepair

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

Each completed task writes `PhoenixRepair.patch` under `<output>/<model>/<instance_id>/`. Use `--resume` with the same inputs to continue an interrupted run.
