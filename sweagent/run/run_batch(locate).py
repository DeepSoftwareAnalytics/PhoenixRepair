"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\ # configure instances
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\     # first 50 instances
    --instances.shuffle=True \\  # shuffle instances (with fixed seed)
    --config config/default.yaml \\
    --agent.model.name gpt-4o  # configure model
[/green]

[cyan][bold]=== LOADING INSTANCES ===[/bold][/cyan]

[cyan][bold]From a file[/bold][/cyan] [green]--instances.type file --instances.path /path/to/file[/green].
[cyan][bold]From huggingface[/bold][/cyan] [green]--instances.type huggingface --instances.dataset_name=SWE_Bench_lite --instances.split=dev[/green].

All instance specifications support the [green]filter[/green], [green]slice[/green], and [green]shuffle[/green] options.
With [green]filter[/green], you can select specific instances, e.g., [green]--instances.filter='instance_id_1|instance_id_2'[/green].
"""
import os
import sys
import time
import copy
import json
import yaml
import random
import hashlib
import asyncio
import getpass
import logging
import traceback
import threading
import requests
from typing import Self
from pathlib import Path
from rich.live import Live
from contextlib import ExitStack
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sweagent import TRAJECTORY_DIR
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.agent.models import GenericAPIModelConfig, LiteLLMModel
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.run_single import RunSingleConfig
from sweagent.tools.tools import ToolConfig
from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (add_file_handler, add_logger_names_to_stream_handlers, get_logger, register_thread_name, remove_file_handler, set_stream_handler_levels)

try:
    from swerex.deployment.hooks.status import SetStatusDeploymentHook
except ImportError:
    SetStatusDeploymentHook = None

def filter_instances_by_ids(instances: list[BatchInstance], target_ids: set[str]) -> list[BatchInstance]:
    filtered_instances = []
    found_ids = set()
    for instance in instances:
        instance_id = instance.problem_statement.id
        original_id = instance_id.split("-sample")[0] if "-sample" in instance_id else instance_id
        if original_id in target_ids:
            filtered_instances.append(instance)
            found_ids.add(original_id)
    missing_ids = target_ids - found_ids
    logger = get_logger("run", emoji="🏃")
    if found_ids:
        logger.info(f"Found the following specified instance IDs: {sorted(found_ids)}")
    if missing_ids:
        logger.warning(f"The following specified instance IDs were not found: {sorted(missing_ids)}")
    logger.info(f"Number of instances after filtering: {len(filtered_instances)} / {len(instances)}")
    return filtered_instances


def deduplicate_patches(sample_paths: list[Path], output_file: Path, logger, min_ratio: float = 0.5) -> dict:
    """
    Deduplicate patch files from multiple samples
    Deduplication rules:
    - If file_path is the same and the longest consecutive matching lines >= min_ratio of the shorter code length, consider as duplicate
    Args:
        sample_paths: List of sample directories
        output_file: Output jsonl file path
        logger: Logger
        min_ratio: Minimum matching ratio (default 0.5, i.e., one-half)
    Returns:
        Deduplication statistics
    """
    all_modifications = []
    unique_modifications = []
    total_count = 0
    duplicate_count = 0
    for sample_path in sample_paths:
        patch_file = sample_path / f"{sample_path.name}.patch"
        if not patch_file.exists():
            logger.debug(f"Patch file does not exist: {patch_file}")
            continue
        try:
            patch_data = json.loads(patch_file.read_text())
            files_to_modify = patch_data.get('files_to_modify', [])
            for file_mod in files_to_modify:
                total_count += 1
                all_modifications.append({'modification': file_mod, 'source': sample_path.name})
        except Exception as e:
            logger.warning(f"Failed to read patch file: {patch_file}, error: {e}")
            continue
    logger.info(f"Collected {total_count} modifications, starting deduplication (requires consecutive matching lines >= {min_ratio*100:.0f}% of shorter code length)")
    for i, item in enumerate(all_modifications):
        mod = item['modification']
        source = item['source']
        file_path = mod.get('file_path', 'N/A')
        is_duplicate = False
        matched_with = None
        for unique_item in unique_modifications:
            is_similar, details = are_modifications_similar(mod, unique_item['modification'], min_ratio)
            if is_similar:
                is_duplicate = True
                duplicate_count += 1
                matched_with = unique_item['source']
                logger.debug(f"Found duplicate: {source} similar to {matched_with}")
                logger.debug(f"  File: {file_path}")
                logger.debug(f"  Code1 lines: {details['lines1_count']}, Code2 lines: {details['lines2_count']}")
                logger.debug(f"  Base length: {details['base_length']} (shorter one)")
                logger.debug(f"  Longest consecutive match: {details['longest_match']} lines")
                logger.debug(f"  Match ratio: {details['match_ratio']*100:.1f}% (threshold: {min_ratio*100:.0f}%)")
                break
        if not is_duplicate:
            unique_modifications.append(item)
            code = mod.get('modifications', [{}])[0].get('current_code', '')
            code_lines = len(get_code_lines(code))
            logger.debug(f"Added unique item #{len(unique_modifications)}: {source}, file: {file_path}, lines: {code_lines}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_modifications:
            json_line = json.dumps({"files_to_modify": [item['modification']], "source_sample": item['source']}, ensure_ascii=False)
            f.write(json_line + '\n')
    logger.info(f"✓ Deduplication complete:")
    logger.info(f"  - Total processed: {total_count} modifications")
    logger.info(f"  - Unique items: {len(unique_modifications)}")
    logger.info(f"  - Duplicate items: {duplicate_count}")
    logger.info(f"  - Deduplication rate: {(duplicate_count/total_count*100):.1f}%" if total_count > 0 else "  - Deduplication rate: N/A")
    logger.info(f"  - Matching threshold: {min_ratio*100:.0f}% of shorter code length")
    return {'total': total_count, 'unique': len(unique_modifications), 'duplicates': duplicate_count, 'dedup_ratio': (duplicate_count/total_count) if total_count > 0 else 0}

def are_modifications_similar(mod1: dict, mod2: dict, min_ratio: float = 0.5) -> tuple[bool, dict]:
    file_path1 = mod1.get('file_path', '')
    file_path2 = mod2.get('file_path', '')
    if file_path1 != file_path2:
        return False, {'reason': 'different_file_path'}
    modifications1 = mod1.get('modifications', [])
    modifications2 = mod2.get('modifications', [])
    if not modifications1 or not modifications2:
        return False, {'reason': 'no_modifications'}
    code1 = modifications1[0].get('current_code', '')
    code2 = modifications2[0].get('current_code', '')
    longest_match, lines1_count = find_longest_consecutive_match(code1, code2)
    _, lines2_count = find_longest_consecutive_match(code2, code1)
    base_length = min(lines1_count, lines2_count)
    if base_length == 0:
        return False, {'reason': 'empty_code', 'longest_match': longest_match, 'lines1_count': lines1_count, 'lines2_count': lines2_count}
    match_ratio = longest_match / base_length
    is_similar = match_ratio >= min_ratio
    details = {'longest_match': longest_match, 'lines1_count': lines1_count, 'lines2_count': lines2_count, 'base_length': base_length, 'match_ratio': match_ratio, 'threshold': min_ratio, 'is_similar': is_similar}
    return is_similar, details

def find_longest_consecutive_match(code1: str, code2: str) -> tuple[int, int]:
    """
    Find the longest consecutive matching lines between two code snippets
    Args:
        code1: First code snippet
        code2: Second code snippet
    Returns:
        (longest consecutive matching lines count, total lines in code1)
    """
    lines1 = get_code_lines(code1)
    lines2 = get_code_lines(code2)
    if not lines1 or not lines2:
        return 0, len(lines1)
    matcher = SequenceMatcher(None, lines1, lines2)
    matches = matcher.get_matching_blocks()
    max_consecutive = 0
    for match in matches:
        if match.size > max_consecutive:
            max_consecutive = match.size
    return max_consecutive, len(lines1)

def get_code_lines(code: str) -> list[str]:
    return [line.strip() for line in code.strip().split('\n') if line.strip()]

class LocationInsightsComparator:
    _thread_local = threading.local()
    def __init__(self, api_key: str = None, base_url: str = "", model_name: str = ""):
        self.logger = get_logger("location-comparator", emoji="🔍")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = 5
        self.retry_delay = 2
    
    @classmethod
    def get_thread_local_instance(cls, api_key: str = None, base_url: str = None, model_name: str = None):
        if not hasattr(cls._thread_local, 'comparator'):
            cls._thread_local.comparator = cls(api_key=api_key, base_url=base_url, model_name=model_name)
        return cls._thread_local.comparator
    
    def _call_api(self, messages: list[dict], temperature: float = 0.7) -> str:
        payload = {'model': self.model_name, 'messages': messages, 'temperature': temperature}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        retries = 0
        while retries < self.max_retries:
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                if 'data' in result and 'response' in result['data']:
                    answer = result['data']['response']['choices'][0]['message']['content']
                elif 'choices' in result:
                    answer = result['choices'][0]['message']['content']
                else:
                    raise ValueError(f"Unexpected API response format: {result.keys()}")
                return answer
            except (ValueError, AttributeError, IndexError, requests.RequestException) as e:
                self.logger.warning(f"API call failed, retrying {retries + 1}/{self.max_retries} (error: {str(e)})")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unknown error occurred during API call: {e}")
                break
        return ""
        
    def compare_location_analyses(self, sample_paths: list[Path], problem_statement: str, num_samples: int) -> str:
        """Compare root cause localization results from multiple samples"""
        self.logger.info(f"Starting comparison of root cause localization results from {num_samples} samples")
        samples_data = []
        for i, sample_path in enumerate(sample_paths, 1):
            traj_file = sample_path / f"{sample_path.name}.traj"
            patch_file = sample_path / f"{sample_path.name}.patch"
            sample_info = {'index': i, 'trajectory': [], 'analysis_content': ''}
            if traj_file.exists():
                try:
                    traj_data = json.loads(traj_file.read_text())
                    sample_info['trajectory'] = traj_data.get('trajectory', [])
                except: 
                    pass
            if patch_file.exists():
                try:
                    sample_info['analysis_content'] = patch_file.read_text()
                except: 
                    pass
            samples_data.append(sample_info)
        prompt = self._build_comparison_prompt(problem_statement, samples_data, num_samples)
        insights = self._call_api([{"role": "user", "content": prompt}])
        return insights if insights.strip() else "Failed to generate location insights"

    def _build_comparison_prompt(self, problem_statement: str, samples: list[dict], num_samples: int) -> str:
        prompt = f"""Analyze {num_samples} root cause localization attempts to identify the TRUE root cause of the issue.

Problem: {problem_statement}

"""
        for s in samples:
            prompt += f"Sample {s['index']}: Analysis = {s['analysis_content']}...\n\n"
        
        prompt += """IMPORTANT: The most frequently identified location is NOT necessarily the root cause. 
Think critically about what truly causes this issue.

Please provide:
1. The TRUE root cause location
   - Distinguish between symptoms vs. actual root cause
   - Consider if the issue originates from a dependency or caller
   - Evaluate which location, if fixed, would resolve the issue at its source
   
2. Detailed explanation of why this is the root cause:
   - What is the fundamental problem in the code logic/design?
   - Why do other identified locations (if any) represent symptoms rather than causes?
   - Trace the causality chain from root cause to observed symptoms
   
3. Actionable insights for fixing:
   - Recommended fix strategy (explain why this approach addresses the root cause)
   - Specific code modification approach with examples
   - Edge cases to consider
   - Testing validation points to confirm the root cause is fixed"""
        return prompt

class MultiSampleComparator:
    _thread_local = threading.local()
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1", model_name: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model_name = model_name
        self.logger = get_logger("comparator", emoji="🔍")
        self.max_retries = 5
        self.retry_delay = 2
        
    @classmethod
    def get_thread_local_instance(cls, api_key: str = None, base_url: str = None, model_name: str = None):
        if not hasattr(cls._thread_local, 'comparator'):
            cls._thread_local.comparator = cls(api_key=api_key, base_url=base_url, model_name=model_name)
            cls._thread_local.thread_id = threading.get_ident()
        return cls._thread_local.comparator
    
    def _call_api(self, messages: list[dict], temperature: float = 0.3, top_p: float = 1.0) -> str:
        payload = {'model': self.model_name, 'messages': messages, 'temperature': temperature, 'top_p': top_p}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        retries = 0
        while retries < self.max_retries:
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                if 'data' in result and 'response' in result['data']:
                    answer = result['data']['response']['choices'][0]['message']['content']
                elif 'choices' in result:
                    answer = result['choices'][0]['message']['content']
                else:
                    raise ValueError(f"Unexpected API response format: {result.keys()}")
                return answer
            except (ValueError, AttributeError, IndexError, requests.RequestException) as e:
                self.logger.warning(f"API call failed, retrying {retries + 1}/{self.max_retries} (error: {str(e)})")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unknown error occurred during API call: {e}")
                break
        return ""

class RunBatchConfig(BaseSettings, cli_implicit_flags=False):
    instances: BatchInstanceSourceConfig = Field(description="Instances to run.")
    agent: AgentConfig = Field(description="Agent options.")
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")
    suffix: str = ""
    raise_exceptions: bool = False
    redo_existing: bool = False
    env_var_path: Path | None = None
    num_workers: int = Field(default=1)
    random_delay_multiplier: float = 0.3
    progress_bar: bool = True
    enable_multi_sampling: bool = False
    num_samples: int = 5
    enable_best_sample_selection: bool = False
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")
    comparison_api_key: str = Field(default="", description="API key for comparison model")
    comparison_api_base: str = Field(default="", description="API base URL for comparison model")
    comparison_model_name: str = Field(
        default="",
        description="Model name for comparison"
    )
    specific_instance_ids: str | None = Field(
        default=None,
        description="Comma-separated list of specific instance IDs to run (e.g., 'django__django-12193,sympy__sympy-14248')"
    )

    def set_default_output_dir(self) -> None:
        if self.output_dir == Path("DEFAULT"):
            user_id = getpass.getuser()
            source_id = self.instances.id
            try:
                model_id = self.agent.model.id
            except AttributeError:
                model_id = "unknown"
            config_file = getattr(self, "_config_files", ["no_config"])[0]
            if config_file != "no_config":
                config_file = Path(config_file).stem
            suffix = f"__{self.suffix}" if self.suffix else ""
            self.output_dir = TRAJECTORY_DIR / user_id / f"{config_file}__{model_id}___{source_id}{suffix}"

    @model_validator(mode="after")
    def evaluate_and_redo_existing(self) -> Self:
        if not isinstance(self.instances, SWEBenchInstances):
            return self
        if self.instances.evaluate and self.redo_existing:
            msg = "Cannot evaluate and redo existing at the same time."
            raise ValueError(msg)
        return self


class _BreakLoop(Exception):
    """Used for internal control flow"""


class RunBatch:
    def __init__(self, instances: list[BatchInstance], agent_config: AgentConfig, *, 
                 output_dir: Path = Path("."), hooks: list[RunHook] | None = None, 
                 raise_exceptions: bool = False, redo_existing: bool = False, 
                 num_workers: int = 1, progress_bar: bool = True, 
                 random_delay_multiplier: float = 0.3):
        if self._model_id in ["human", "human_thought"] and num_workers > 1:
            msg = "Cannot run with human model in parallel"
            raise ValueError(msg)
        self.logger = get_logger("swea-run", emoji="🏃")
        add_file_handler(output_dir / "run_batch.log", id_="progress", filter=lambda name: "swea-run" in name or "config" in name)
        self.instances = instances
        self.agent_config = agent_config
        self.output_dir = output_dir
        self._raise_exceptions = raise_exceptions
        self._chooks = CombinedRunHooks()
        self._redo_existing = redo_existing
        self._num_workers = num_workers
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml")
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier
        self.enable_multi_sampling = False
        self.num_samples = 5
        self._comparator_config = None


    @property
    def _model_id(self) -> str:
        try:
            return self.agent_config.model.id
        except AttributeError:
            return "unknown"

    @classmethod
    def from_config(cls, config: RunBatchConfig) -> Self:
        logger = get_logger("run", emoji="🏃")
        load_environment_variables(config.env_var_path)
        config.set_default_output_dir()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "run_batch.config.yaml").write_text(yaml.dump(config.model_dump_json(), indent=2))
        logger.debug("Loading instances from %s", f"{config.instances!r}")
        instances = config.instances.get_instance_configs()
        logger.info("Loaded %d instances", len(instances))
        if config.specific_instance_ids:
            specific_ids = {id.strip() for id in config.specific_instance_ids.split(',') if id.strip()}
            logger.info(f"Filtering to specific instance IDs: {sorted(specific_ids)}")
            instances = filter_instances_by_ids(instances, specific_ids)
            if not instances:
                msg = f"No instances found after filtering. Please check the specified instance IDs: {sorted(specific_ids)}"
                raise ValueError(msg)
            if not config.enable_multi_sampling:
                pass
            else:
                logger.info(f"Multi-sampling mode: will execute {config.num_samples} samples for each of {len(instances)} tasks")
                if config.num_workers > 1:
                    logger.info(f"Task-level parallelism: {config.num_workers} tasks in parallel, each task's samples executed sequentially")
                else:
                    logger.info(f"Fully sequential mode: both tasks and samples executed sequentially")
        if not instances:
            msg = "No instances to run."
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        actual_num_workers = config.num_workers if config.enable_multi_sampling else 1
        rb = cls(
            instances=instances,
            agent_config=config.agent,
            output_dir=config.output_dir,
            raise_exceptions=config.raise_exceptions,
            redo_existing=config.redo_existing,
            num_workers=actual_num_workers,
            progress_bar=config.progress_bar,
            random_delay_multiplier=config.random_delay_multiplier
        )
        rb.enable_multi_sampling = config.enable_multi_sampling
        rb.num_samples = config.num_samples
        rb._comparator_config = {'api_key': config.comparison_api_key, 'base_url': config.comparison_api_base, 'model_name': config.comparison_model_name}
        if isinstance(config.instances, SWEBenchInstances) and config.instances.evaluate:
            from sweagent.run.hooks.swe_bench_evaluate import SweBenchEvaluate
            rb.add_hook(SweBenchEvaluate(output_dir=config.output_dir, subset=config.instances.subset, split=config.instances.split, continuous_submission_every=1))
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def main_sequential_sampling(self) -> None:
        """Fully sequential mode: all tasks executed serially, samples within each task also executed sequentially"""
        original_ids = set()
        original_instances_map = {}
        for inst in self.instances:
            if '-sample' not in inst.problem_statement.id:
                original_id = inst.problem_statement.id
                original_ids.add(original_id)
                original_instances_map[original_id] = inst
        self.logger.info(f"Fully sequential mode: executing {self.num_samples} samples for each of {len(original_ids)} tasks")
        for original_id in sorted(original_ids):
            try:
                template_instance = original_instances_map[original_id]
                self._run_single_task_sequential_samples(original_id, template_instance)
            except _BreakLoop:
                self.logger.info("Received interrupt signal, stopping subsequent tasks")
                break
            except Exception as e:
                self.logger.error(f"Task {original_id} execution failed: {e}")
                if self._raise_exceptions:
                    raise

    def main_parallel_tasks_with_sequential_samples(self) -> None:
        """Parallel tasks mode: multiple tasks executed in parallel, but samples within each task executed sequentially"""
        original_ids = set()
        original_instances_map = {}
        for inst in self.instances:
            if '-sample' not in inst.problem_statement.id:
                original_id = inst.problem_statement.id
                original_ids.add(original_id)
                original_instances_map[original_id] = inst
        self.logger.info(f"Parallel tasks mode: {len(original_ids)} tasks in parallel ({self._num_workers} threads), {self.num_samples} samples per task executed sequentially")
        add_logger_names_to_stream_handlers()
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)
        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = {}
                for original_id in sorted(original_ids):
                    template_instance = original_instances_map[original_id]
                    future = executor.submit(self._run_single_task_sequential_samples, original_id, template_instance)
                    futures[future] = original_id
                try:
                    for future in as_completed(futures):
                        original_id = futures[future]
                        try:
                            future.result()
                            self.logger.info(f"✅ All samples for task {original_id} completed")
                        except Exception as e:
                            self.logger.error(f"❌ Task {original_id} execution failed: {e}")
                            if self._raise_exceptions:
                                raise
                except (KeyboardInterrupt, _BreakLoop):
                    self.logger.info("Received interrupt signal, waiting for running tasks to complete")
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()

    def _run_single_task_sequential_samples(self, original_id: str, template_instance: BatchInstance) -> None:
        """Execute all samples for a single task, with checkpoint recovery support"""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Starting task: {original_id}")
        self.logger.info(f"{'=' * 80}\n")
        sample5_patch = self.output_dir / f"{original_id}-sample5" / f"{original_id}-sample5.patch"
        if sample5_patch.exists():
            self.logger.info(f"✅ Task {original_id}'s sample5 already has .patch file, skipping entire task")
            return
        start_sample_idx = 1
        for i in range(1, self.num_samples + 1):
            sample_id = f"{original_id}-sample{i}"
            patch_file = self.output_dir / sample_id / f"{sample_id}.patch"
            if not patch_file.exists():
                start_sample_idx = i
                self.logger.info(f"Starting from Sample {i} (first sample missing .patch)")
                break
        else:
            self.logger.info(f"✅ All samples for task {original_id} already have .patch files")
            return
        for sample_idx in range(start_sample_idx, self.num_samples + 1):
            sample_id = f"{original_id}-sample{sample_idx}"
            patch_file = self.output_dir / sample_id / f"{sample_id}.patch"
            if patch_file.exists():
                self.logger.info(f"⏭️ Skipping Sample {sample_idx}/{self.num_samples}: {sample_id} (already has .patch)")
                continue
            self.logger.info(f"▶️ Executing Sample {sample_idx}/{self.num_samples}: {sample_id}")
            is_complete, status = self._check_trajectory_complete(sample_id)
            if is_complete and not self._redo_existing:
                self.logger.info(f"⏭️ Skipping completed sample: {sample_id} (status: {status})")
                continue
            sample_instance = copy.deepcopy(template_instance)
            sample_instance.problem_statement.id = sample_id
            if sample_idx > 1:
                prev_sample_idx = sample_idx - 1
                prev_sample_id = f"{original_id}-sample{prev_sample_idx}"
                reflection = self._generate_reflection(original_id, prev_sample_id, sample_idx)
                if reflection:
                    enhanced_config = self._enhance_config_with_reflection(self.agent_config, reflection, sample_idx, sample_id)
                    self.logger.info(f"Added reflection from sample{prev_sample_idx}")
                else:
                    enhanced_config = self.agent_config
                    self.logger.warning(f"Could not get reflection from sample{prev_sample_idx}, using original config")
            else:
                enhanced_config = self.agent_config
                self.logger.info(f"Sample 1: using original config (no prior reflection)")
            try:
                self._run_single_sample(sample_instance, enhanced_config)
                self.logger.info(f"✅ Sample {sample_idx} execution completed")
            except KeyboardInterrupt:
                raise _BreakLoop
            except Exception as e:
                self.logger.error(f"❌ Sample {sample_idx} execution failed: {e}")
                self.logger.error(traceback.format_exc())
                if self._raise_exceptions:
                    raise
                continue
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting generation of location insights for {original_id}")
        try:
            self._generate_location_insights(original_id, template_instance)
        except Exception as e:
            self.logger.error(f"Failed to generate location insights: {e}")
            self.logger.error(traceback.format_exc())
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting deduplication for all samples of {original_id}")
        try:
            sample_paths = []
            for i in range(1, self.num_samples + 1):
                sample_path = self.output_dir / f"{original_id}-sample{i}"
                if sample_path.exists():
                    sample_paths.append(sample_path)
            last_sample_dir = self.output_dir / f"{original_id}-sample{self.num_samples}"
            last_sample_dir.mkdir(parents=True, exist_ok=True)
            deduplicated_file = last_sample_dir / "deduplicated_patches_new.jsonl"
            stats = deduplicate_patches(sample_paths, deduplicated_file, self.logger)
            self.logger.info(f"✓ Deduplication complete:")
            self.logger.info(f"  - Total processed: {stats['total']} modifications")
            self.logger.info(f"  - Unique items: {stats['unique']}")
            self.logger.info(f"  - Duplicate items: {stats['duplicates']}")
            self.logger.info(f"  - Results saved to: {deduplicated_file}")
        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            self.logger.error(traceback.format_exc())
        self.logger.info(f"\n✅ All samples for task {original_id} completed\n")

    def _generate_reflection(self, original_id: str, prev_sample_id: str, current_sample_idx: int) -> str:
        """Generate reflection from previous sample"""
        prev_sample_dir = self.output_dir / prev_sample_id
        traj_file = prev_sample_dir / f"{prev_sample_id}.traj"
        patch_file = prev_sample_dir / f"{prev_sample_id}.patch"
        if not traj_file.exists():
            self.logger.warning(f"Previous sample's trajectory file does not exist: {traj_file}")
            return ""
        try:
            traj_data = json.loads(traj_file.read_text())
            trajectory = traj_data.get('trajectory', [])
            exit_status = traj_data.get('info', {}).get('exit_status', 'unknown')
            analysis_content = ""
            if patch_file.exists():
                analysis_content = patch_file.read_text()
            reflection = f"""
## Previous Attempt Analysis (Sample {current_sample_idx - 1})

**Execution Summary:**
- Total steps: {len(trajectory)}
- Exit status: {exit_status}
- Analysis file present: {'Yes' if analysis_content else 'No'}

**Previous Analysis Result:**
{analysis_content if analysis_content else '(No analysis file generated)'}

**Key Trajectory Steps:**
"""
            key_steps = []
            self.logger.info(f"Extracting complete trajectory from previous sample ({len(trajectory)} steps)")
            for i, step in enumerate(trajectory, start=1):
                thought = step.get('thought', 'N/A')[:200]
                action = step.get('action', 'N/A')[:200]
                key_steps.append(f"Step {i}: {thought}, Action: {action}...")
            reflection += '\n'.join(key_steps)
            reflection += f"""
**Critical Instructions for Sample {current_sample_idx}:**

1. **Learn from Previous Attempt:**
   - Review what the previous analysis identified
   - Consider if those locations were truly the root cause
   - Think about what might have been overlooked

2. **DO NOT Simply Copy:**
   - Do NOT just repeat the same file locations
   - Do NOT copy the same code snippets without new analysis
   - Challenge the previous conclusions

3. **Explore New Possibilities:**
   - Look at files that were NOT analyzed in the previous attempt
   - Consider alternative root causes
   - Investigate deeper layers of the codebase
   - Think about edge cases or indirect causes

4. **Fresh Perspective Required:**
   - Approach the problem from a different angle
   - Question assumptions from the previous attempt
   - Look for patterns or connections that were missed
   - Consider the broader system architecture

5. **Root Cause Deep Dive:**
   - If previous attempt found symptoms, find the actual cause
   - If previous attempt was surface-level, go deeper
   - If previous attempt focused on one area, explore related areas
   - Trace the issue through multiple layers of abstraction

**Your goal:** Provide a DIFFERENT and potentially MORE ACCURATE analysis than the previous attempt. 
Be creative, thorough, and willing to challenge previous conclusions.
"""
            return reflection
        except Exception as e:
            self.logger.error(f"Failed to generate reflection: {e}")
            return ""

    def _enhance_config_with_reflection(self, original_config: AgentConfig, reflection: str, sample_idx: int, instance_id: str) -> AgentConfig:
        """Enhance agent config with reflection"""
        enhanced_config = original_config.model_copy(deep=True)
        original_template = enhanced_config.templates.instance_template
        enhanced_template = f"""# Iterative Root Cause Analysis - Attempt {sample_idx}

{reflection}

{'='*80}
# Original Task Description
{'='*80}

{original_template}
"""
        enhanced_config.templates.instance_template = enhanced_template
        self.logger.info(f"Added reflection for Sample {sample_idx} (length: {len(reflection)} chars)")
        instance_output_dir = self.output_dir / instance_id
        instance_output_dir.mkdir(parents=True, exist_ok=True)
        enhanced_template_file = instance_output_dir / "enhanced_template.txt"
        try:
            with open(enhanced_template_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_template)
            self.logger.info(f"✓ Enhanced template saved to: {enhanced_template_file}")
        except Exception as e:
            self.logger.error(f"✗ Failed to write enhanced_template: {e}")
        return enhanced_config

    def _run_single_sample(self, instance: BatchInstance, agent_config: AgentConfig) -> None:
        """Execute a single sample"""
        instance_id = instance.problem_statement.id
        self.logger.info(f"Starting execution: {instance_id}")
        register_thread_name(instance_id)
        self._add_instance_log_file_handlers(instance_id, multi_worker=self._num_workers > 1)
        self._progress_manager.on_instance_start(instance_id)
        try:
            result = self._run_single_instance_with_config(instance, agent_config)
            self._progress_manager.on_instance_end(instance_id, exit_status=result.info.get("exit_status", "unknown_exit"))
        except KeyboardInterrupt:
            raise _BreakLoop
        except Exception as e:
            self.logger.error(f"❌ Sample execution failed: {e}")
            self.logger.error(traceback.format_exc())
            self._progress_manager.on_uncaught_exception(instance_id, e)
            if self._raise_exceptions:
                raise
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance_id)

    def _check_trajectory_complete(self, instance_id: str) -> tuple[bool, str]:
        traj_path = self.output_dir / instance_id / f"{instance_id}.traj"
        if not traj_path.exists():
            self.logger.debug(f"Trajectory file does not exist: {traj_path}")
            return False, "not_found"
        try:
            content = traj_path.read_text()
            if not content.strip():
                self.logger.warning(f"Found empty trajectory file: {traj_path}")
                return False, "empty"
            data = json.loads(content)
            exit_status = data.get("info", {}).get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Trajectory file has no valid exit status: {traj_path} (status: {exit_status})")
                return False, "incomplete"
            trajectory = data.get("trajectory", [])
            if not trajectory:
                self.logger.warning(f"Trajectory is empty: {traj_path}")
                return False, "empty_trajectory"
            self.logger.info(f"✓ Trajectory complete: {instance_id} (exit_status: {exit_status}, {len(trajectory)} steps)")
            return True, exit_status
        except json.JSONDecodeError as e:
            self.logger.error(f"Trajectory file JSON format error: {traj_path}: {e}")
            return False, "json_error"
        except Exception as e:
            self.logger.error(f"Failed to check trajectory: {traj_path}: {e}")
            return False, "check_error"

    def should_skip_instance(self, instance: BatchInstance) -> tuple[bool, str]:
        # If redo_existing is set, don't skip
        if self._redo_existing:
            return False, ""
        instance_id = instance.problem_statement.id
        is_complete, status = self._check_trajectory_complete(instance_id)
        if is_complete:
            return True, f"skipped (already_complete: {status})"
        return False, ""

    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()
        if self.enable_multi_sampling:
            if self._num_workers <= 1:
                self.logger.info("Single-threaded mode: all tasks executed serially, samples executed sequentially")
                self.main_sequential_sampling()
            else:
                self.logger.info(f"Multi-threaded mode: {self._num_workers} tasks in parallel, samples within each task executed sequentially")
                self.main_parallel_tasks_with_sequential_samples()
        elif self._num_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()
        output_dirs = []
        for instance in self.instances:
            output_dirs.append(self.output_dir / instance.problem_statement.id)
        merge_predictions(output_dirs, self.output_dir / "preds.json")
        self._chooks.on_end()
    
    def main_sampling_phase(self) -> None:
        sampling_instances = [inst for inst in self.instances if '-sample' in inst.problem_statement.id and int(inst.problem_statement.id.split('-sample')[1]) <= self.num_samples]
        self.logger.info(f"Sampling phase: {len(sampling_instances)} instances to check")
        instances_to_run = []
        skipped_count = 0
        for inst in sampling_instances:
            should_skip, reason = self.should_skip_instance(inst)
            if should_skip:
                self.logger.info(f"⏭️ Skipping completed instance: {inst.problem_statement.id} ({reason})")
                skipped_count += 1
            else:
                instances_to_run.append(inst)
        self.logger.info(f"Sampling phase: skipped {skipped_count} completed instances, executing {len(instances_to_run)} new instances")
        if not instances_to_run:
            self.logger.info("Sampling phase: all instances completed, no execution needed")
            return
        if self._num_workers <= 1:
            self._run_instances_sequential(instances_to_run)
        else:
            self._run_instances_parallel(instances_to_run)

    def main_single_worker(self) -> None:
        with ExitStack() as stack:
            if self._model_id not in ["human", "human_thought"] and self._show_progress_bar:
                stack.enter_context(Live(self._progress_manager.render_group))
            for instance in self.instances:
                try:
                    self.run_instance(instance)
                except _BreakLoop:
                    self.logger.info("Stopping loop over instances")
                    break

    def main_multi_worker(self) -> None:
        add_logger_names_to_stream_handlers()
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)
        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = [executor.submit(self.run_instance, instance) for instance in self.instances]
                try:
                    for future in as_completed(futures):
                        future.result()
                except (KeyboardInterrupt, _BreakLoop):
                    msg = "Received keyboard interrupt, waiting for running instances to finish"
                    self.logger.info(msg)
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()
    
    def _run_instances_sequential(self, instances: list[BatchInstance]) -> None:
        """Execute instance list sequentially"""
        for instance in instances:
            try:
                self.run_instance(instance)
            except _BreakLoop:
                self.logger.info("Stopping loop over instances")
                break
    
    def _run_instances_parallel(self, instances: list[BatchInstance]) -> None:
        """Execute instance list in parallel"""
        add_logger_names_to_stream_handlers()
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)
        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = [executor.submit(self.run_instance, instance) for instance in instances]
                try:
                    for future in as_completed(futures):
                        future.result()
                except (KeyboardInterrupt, _BreakLoop):
                    self.logger.info("Received interrupt signal")
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()

    def run_instance(self, instance: BatchInstance) -> None:
        instance_id = instance.problem_statement.id
        should_skip, skip_reason = self.should_skip_instance(instance)
        if should_skip:
            self.logger.info(f"⏭️ Skipping {instance_id}: {skip_reason}")
            self._progress_manager.on_instance_start(instance_id)
            self._progress_manager.on_instance_end(instance_id, exit_status=skip_reason)
            return
        self.logger.info("Running on instance %s", instance_id)
        register_thread_name(instance_id)
        self._add_instance_log_file_handlers(instance_id, multi_worker=self._num_workers > 1)
        if self._progress_manager.n_completed < self._num_workers:
            time.sleep(random.random() * self._random_delay_multiplier * (self._num_workers - 1))
        self._progress_manager.on_instance_start(instance_id)
        try:
            result = self._run_instance(instance)
        except KeyboardInterrupt:
            raise _BreakLoop
        except (SystemExit, ModelConfigurationError, TotalCostLimitExceededError) as e:
            if self._raise_exceptions:
                raise
            self.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"❌ Failed on {instance_id}: {e}")
            self._progress_manager.on_uncaught_exception(instance_id, e)
            if self._raise_exceptions:
                raise
        else:
            self._progress_manager.on_instance_end(instance_id, exit_status=result.info.get("exit_status", "unknown_exit"))
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance_id)

    def _run_instance(self, instance: BatchInstance) -> AgentRunResult:
        """Standard instance execution"""
        return self._run_single_instance_with_config(instance, self.agent_config)
    
    def _run_single_instance_with_config(self, instance: BatchInstance, agent_config: AgentConfig) -> AgentRunResult:
        """Execute a single instance with specified config"""
        output_dir = Path(self.output_dir) / instance.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)
        thread_local_agent_config = agent_config.model_copy(deep=True)
        thread_local_agent_config.name = instance.problem_statement.id
        agent = get_agent_from_config(thread_local_agent_config)
        thread_local_env_config = instance.env.model_copy(deep=True)
        thread_local_env_config.name = instance.problem_statement.id
        single_run_replay_config = RunSingleConfig(agent=thread_local_agent_config, problem_statement=instance.problem_statement, env=thread_local_env_config)
        (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(yaml.dump(single_run_replay_config.model_dump_json(), indent=2))
        agent.replay_config = single_run_replay_config
        agent.add_hook(SetStatusAgentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        self._progress_manager.update_instance_status(instance.problem_statement.id, "Starting environment")
        env = SWEEnv.from_config(thread_local_env_config)
        env.add_hook(SetStatusEnvironmentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        if SetStatusDeploymentHook is not None:
            env.deployment.add_hook(SetStatusDeploymentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        try:
            env.start()
            self._chooks.on_instance_start(index=0, env=env, problem_statement=instance.problem_statement)
            result = agent.run(problem_statement=instance.problem_statement, env=env, output_dir=output_dir)
        except Exception:
            agent.logger.error(traceback.format_exc())
            raise
        finally:
            env.close()
        save_predictions(self.output_dir, instance.problem_statement.id, result)
        self._chooks.on_instance_completed(result=result)
        return result

    def _generate_location_insights(self, original_id: str, template_instance: BatchInstance) -> None:
        sample_paths = []
        for i in range(1, self.num_samples + 1):
            sample_path = self.output_dir / f"{original_id}-sample{i}"
            sample_paths.append(sample_path)
        last_sample_id = f"{original_id}-sample{self.num_samples}"
        insights_output_dir = self.output_dir / last_sample_id
        insights_output_dir.mkdir(parents=True, exist_ok=True)
        insights_file = insights_output_dir / "location_insights.txt"
        if insights_file.exists() and insights_file.stat().st_size > 0:
            self.logger.info(f"✓ Location insights already exist: {insights_file}")
            return
        problem_statement_text = template_instance.problem_statement.get_problem_statement()
        if hasattr(self, '_comparator_config') and self._comparator_config:
            comparator = LocationInsightsComparator.get_thread_local_instance(api_key=self._comparator_config['api_key'], base_url=self._comparator_config['base_url'], model_name=self._comparator_config['model_name'])
        else:
            comparator = LocationInsightsComparator.get_thread_local_instance()
        self.logger.info(f"Generating location insights for {original_id}")
        insights = comparator.compare_location_analyses(sample_paths, problem_statement_text, self.num_samples)
        with open(insights_file, 'w', encoding='utf-8') as f:
            f.write(f"Location Insights for {original_id}\n")
            f.write(f"Generated from {self.num_samples} sampling attempts\n")
            f.write("=" * 80 + "\n\n")
            f.write(insights)
        self.logger.info(f"✓ Location insights saved to: {insights_file}")
    
    def _enhance_agent_config_with_insights(self, original_config: AgentConfig, insights: str, num_samples: int) -> AgentConfig:
        """Enhance agent config with insights"""
        enhanced_config = original_config.model_copy(deep=True)
        enhanced_config.use_change_history_method = False
        original_instance_template = enhanced_config.templates.instance_template
        enhanced_instance_template = f"""Based on {num_samples} previous solution attempts for this task, we have conducted a comparative analysis to identify the best approach and key learnings:

{insights}

Please solve the task by leveraging these insights from previous attempts. Focus on:
1. Building upon the strengths identified in the best approach
2. Avoiding the common mistakes observed across attempts
3. Addressing the specific areas that need attention

# Original Task Instructions
{original_instance_template}
"""
        enhanced_config.templates.instance_template = enhanced_instance_template
        self.logger.info(f"Enhanced instance template with comparison insights (length: {len(insights)} chars)")
        return enhanced_config

    def _add_instance_log_file_handlers(self, instance_id: str, multi_worker: bool = False) -> None:
        filename_template = f"{instance_id}.{{level}}.log"
        for level in ["trace", "debug", "info"]:
            filter = instance_id if multi_worker else ""
            add_file_handler(self.output_dir / instance_id / filename_template.format(level=level), filter=filter, level=level, id_=f"{instance_id}-{level}")

    def _remove_instance_log_file_handlers(self, instance_id: str) -> None:
        for level in ["trace", "debug", "info"]:
            remove_file_handler(f"{instance_id}-{level}")


def run_from_config(config: RunBatchConfig):
    RunBatch.from_config(config).main()

def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (__doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunBatchConfig))
    run_from_config(BasicCLI(RunBatchConfig, help_text=help_text).get_config(args))

if __name__ == "__main__":
    run_from_cli()