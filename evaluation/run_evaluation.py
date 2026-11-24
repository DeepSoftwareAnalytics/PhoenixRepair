import os
import sys
import json
import argparse
import subprocess
from typing import Any
from pathlib import Path
from datetime import datetime
from utils import BENCHMARK_CONFIG

class BenchmarkEvaluation:
    """
    Simplified class for running evaluations only.
    Handles evaluation using pre-generated patches.
    """

    def __init__(
        self,
        benchmark: str,
        dataset: str = "SWE-bench_Verified",
        run_id: str | None = None,
        max_workers: int = 4,
        instance_ids: list[str] | None = None,
        results_dir: str = "",
        benchmark_harness_path: str = "SWE-bench"
    ):
        """
        Initialize the BenchmarkEvaluation class.

        Args:
            benchmark: Benchmark name.
            dataset: Dataset name.
            benchmark_harness_path: Path to benchmark harness (for evaluation).
            run_id: Unique run identifier (if None, will use timestamp).
            max_workers: Maximum number of parallel workers.
            instance_ids: List of instance IDs to run (optional).
        """
        assert benchmark in BENCHMARK_CONFIG, f"Invalid benchmark name: {benchmark}"
        self.config = BENCHMARK_CONFIG[benchmark]
        self.dataset_name = dataset
        assert self.dataset_name in self.config.valid_datasets, (f"Invalid dataset name: {self.dataset_name}")
        self.benchmark = benchmark
        self.max_workers = max_workers
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"trae-agent-{timestamp}"
        else:
            self.run_id = run_id
        self.results_dir = results_dir
        self.benchmark_harness_path = benchmark_harness_path

    def run_eval(self):
        """
        Run evaluation using the SWE-bench harness.
        """
        print(f"Running evaluation with run_id: {self.run_id}")
        # swebench_harness_path = Path(self.benchmark_harness_path)
        swebench_harness_path = Path(self.benchmark_harness_path).resolve()
        # swebench_python_path = "swebench_venv/bin/python"
        swebench_python_path = sys.executable
        predictions_path = os.path.abspath(os.path.join(self.results_dir, "final_answer1.json"))
        print(f"Looking for predictions.json at: {predictions_path}")
        print(f"File exists: {os.path.exists(predictions_path)}")
        print(f"Working directory: {os.getcwd()}")
        cmd = [
            swebench_python_path,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            f"princeton-nlp/{self.dataset_name}",
            "--predictions_path",
            predictions_path,
            "--run_id",
            self.run_id,
            "--cache_level",
            "instance",
            "--instance_image_tag",
            "latest",
        ]
        process = subprocess.run(cmd, capture_output=True, cwd=swebench_harness_path.as_posix())
        print(process.stdout.decode())
        print(process.stderr.decode())
        result_filename = f"{self.run_id}.json"
        print(f"Evaluation completed and file saved to {result_filename}")

    def get_all_preds(self, instance_ids: list[str] | None = None):
        """
        Collect all generated patches and write predictions.json to results directory.

        Args:
            instance_ids: List of instance IDs to collect (optional).
        """
        preds: list[dict[str, str]] = []
        patch_path = os.path.join(self.results_dir, "answer_final.json")
        print(f"patch_path:{patch_path}")
        with open(patch_path, 'r') as f:
            dict1 = json.load(f)
        for k, v in dict1.items():
            preds.append({"instance_id": k, "model_name_or_path": "PhoenixRepair", "model_patch": v['model_patch']})
        print(f"Collected {len(preds)} predictions")
        with open(os.path.join(self.results_dir, "final_answer1.json"), "w") as f:
            json.dump(preds, f)

def main():
    """
    Main entry point for benchmark evaluation script.
    Parses command-line arguments and runs evaluation only.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--benchmark", type=str, default="SWE-bench", help="Benchmark name.")
    argument_parser.add_argument("--dataset", type=str, default="SWE-bench_Verified", help="Dataset name.")
    argument_parser.add_argument("--run-id", type=str, required=False, help="Run ID for SWE-bench evaluation (deprecated, use --run_id).")
    argument_parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated).")
    argument_parser.add_argument("--run_id", type=str, required=False, help="Run ID for benchmark evaluation. If not provided, will use timestamp.")
    argument_parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel workers.")
    argument_parser.add_argument("--results_dir", type=str, default="")
    argument_parser.add_argument("--benchmark_harness_path", type=str, default="SWE-bench", help="Benchmark name.")
    args = argument_parser.parse_args()
    # Use --run_id if provided, otherwise use --run-id for backward compatibility, otherwise None (will generate timestamp)
    run_id = args.run_id or args.run_id
    evaluation = BenchmarkEvaluation(
        args.benchmark,
        args.dataset,
        run_id,  # This will be None if not provided, triggering timestamp generation
        args.max_workers,
        args.instance_ids,
        args.results_dir,
        args.benchmark_harness_path
    )
    # Evaluation only
    evaluation.get_all_preds(args.instance_ids)
    evaluation.run_eval()

if __name__ == "__main__":
    main()
