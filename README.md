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
./run_our_method.sh --model glm-5.2 --instances /path/to/tasks.jsonl --dry-run
./run_our_method.sh --model glm-5.2 --instances /path/to/tasks.jsonl --output runs/glm-repair
```

## Run a single SWE-agent trajectory

Use the same setup and manifest. SWE-agent starts from the issue and runs one independent trajectory per selected task.

```bash
./run_swe_agent.sh --model glm-5.2 --instances /path/to/tasks.jsonl --output runs/glm-single
```

For either script, add `--instance-id TASK_ID` to select one task, `--dry-run` to inspect the plan, or `--resume` with the original inputs and output directory to continue an interrupted run.

## Evaluate final patches

Set `RUN` to either script's output directory. Use a fresh export directory for each evaluation. Export includes every task in the run manifest; failed tasks receive an empty patch.

**SWE-bench Verified.** Install the official `swebench` Python package in your evaluation environment and start Docker. From the PhoenixRepair repository root:

```bash
RUN="$PWD/runs/glm-single"
python evaluation/export_predictions.py --run-dir "$RUN" --benchmark verified --output "$RUN/eval-verified"
mapfile -t IDS < "$RUN/eval-verified/ids.txt"
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$RUN/eval-verified/predictions.jsonl" \
  --instance_ids "${IDS[@]}" --run_id glm-single-verified --max_workers 4
```

Use a distinct run ID for each evaluation. The harness prints its report path and writes per-instance logs under `logs/run_evaluation/<run_id>/`.

**SWE-bench Pro V2, including HARD-51.** Clone the official [Pro V2 repository](https://github.com/scaleapi/SWE-bench_Pro-os) and follow its Harbor and Modal setup. Set `PRO_V2` to that repository's absolute checkout path. The commands below replay the final patches on clean task images and run the supplied verifiers for the selected task IDs.

```bash
RUN="$PWD/runs/glm-single"
EXPORT="$RUN/eval-pro-v2"
PRO_V2=/path/to/SWE-bench_Pro-os
python evaluation/export_predictions.py --run-dir "$RUN" --benchmark pro-v2 --output "$EXPORT"
python "$PRO_V2/v2/tooling/make_subset.py" "$PRO_V2/v2/tasks" "$EXPORT/ids.txt" "$EXPORT/tasks"
PYTHONPATH="$PRO_V2/v2/tooling" harbor run -p "$EXPORT/tasks" -e modal -n 4 \
  -a patch_replay:PatchReplayAgent -m replay --ak "source_job=$EXPORT/source_job" \
  --jobs-dir "$EXPORT/jobs" --job-name regrade
```

Harbor writes the job result to `$EXPORT/jobs/regrade/result.json` and each trial's score to its verifier output.
