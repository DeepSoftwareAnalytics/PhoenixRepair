"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\
    --instances.shuffle=True \\
    --config config/default.yaml \\
    --agent.model.name gpt-4o
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
import asyncio
import getpass
import logging
import requests
import traceback
import threading
from typing import Self
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.live import Live
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sweagent import TRAJECTORY_DIR
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.agent.models import GenericAPIModelConfig, LiteLLMModel
from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run.run_single import RunSingleConfig
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances

from sweagent.tools.tools import ToolConfig
from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (add_file_handler, add_logger_names_to_stream_handlers, get_logger, register_thread_name, remove_file_handler, set_stream_handler_levels)

try:
    from swerex.deployment.hooks.status import SetStatusDeploymentHook
except ImportError:
    SetStatusDeploymentHook = None

try:
    from learning.summary_new import update_exp as _learning_update_exp
except Exception:
    _learning_update_exp = None




def filter_instances_by_ids(instances: list[BatchInstance], target_ids: set[str]) -> list[BatchInstance]:
    """
    Filter instances to keep only those specified in target_ids
    """
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
        logger.info(f"Found specified instance IDs: {sorted(found_ids)}")
    if missing_ids:
        logger.warning(f"Could not find specified instance IDs: {sorted(missing_ids)}")
    logger.info(f"Filtered instance count: {len(filtered_instances)} / {len(instances)}")
    return filtered_instances

class MultiSampleComparator:
    """Class for comparing multi-sample results and selecting the best trajectory"""
    
    _thread_local = threading.local()
    def __init__(self, api_key: str = None, base_url: str = "", model_name: str = "deepseek-chat"):
        """Initialize comparator"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model_name = model_name
        self.logger = get_logger("comparator", emoji="🔍")
        self.max_retries = 5
        self.retry_delay = 2
        
    @classmethod
    def get_thread_local_instance(cls, api_key: str = None, base_url: str = None, model_name: str = None):
        """Get thread-local comparator instance"""
        if not hasattr(cls._thread_local, 'comparator'):
            cls._thread_local.comparator = cls(api_key=api_key, base_url=base_url, model_name=model_name)
            cls._thread_local.thread_id = threading.get_ident()
        return cls._thread_local.comparator
    
    def _extract_best_location_index(self, insights: str, valid_locations: list[dict]) -> int | None:
        """Extract best location index from insights
        Returns:
            int: valid location index
            None: unable to extract valid index
        """
        import re
        match = re.search(r'BEST_SAMPLE:\s*(\d+)', insights, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if any(loc['index'] == idx for loc in valid_locations):
                self.logger.info(f"Extracted best location from analysis: Location {idx}")
                return idx
            else:
                self.logger.warning(f"Extracted index {idx} not in valid range: {[loc['index'] for loc in valid_locations]}")
        else:
            self.logger.warning("Could not match BEST_SAMPLE format in analysis result")
        return None
    
    def _call_api(self, messages: list[dict], temperature: float = 0.3, top_p: float = 1.0) -> str:
        """Call API to get response"""
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
                self.logger.warning(f"API call failed, retry {retries + 1}/{self.max_retries} (error: {str(e)})")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unknown error in API call: {e}")
                break
        return ""

    def compare_samples(self, sample_paths: list[Path], problem_statement: str, num_samples: int) -> dict:
        self.logger.info(f"Starting comparison of {len(sample_paths)} sample results")
        sample_data = []
        for i, sample_path in enumerate(sample_paths, 1):
            traj_file = sample_path / f"{sample_path.name}.traj"
            if not traj_file.exists():
                self.logger.warning(f"sample {i} trajectory does not exist: {traj_file}")
                continue
            try:
                data = json.loads(traj_file.read_text())
                sample_data.append({
                    'index': i,
                    'path': sample_path,
                    'trajectory': data.get('trajectory', []),
                    'info': data.get('info', {}),
                    'submission': data.get('info', {}).get('submission', ''),
                    'exit_status': data.get('info', {}).get('exit_status', 'unknown')
                })
                self.logger.info(f"Successfully read sample {i}: {len(data.get('trajectory', []))} steps, exit: {data.get('info', {}).get('exit_status', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Failed to read sample {i}: {e}")
                continue
        if not sample_data:
            self.logger.error("No valid sample data")
            return {'insights': 'No valid sample data', 'comparison_failed': True, 'total_locations': len(sample_paths), 'valid_locations': 0, 'best_location_index': None}
        comparison_prompt = self._build_comparison_prompt(problem_statement, sample_data, len(sample_data))
        messages = [{"role": "user", "content": comparison_prompt}]
        insights = ""
        for attempt in range(self.max_retries):
            insights = self._call_api(messages)
            self.logger.info(f"insights:\n{insights}")
            if insights.strip():
                self.logger.info(f"Attempt {attempt + 1} successfully obtained comparison result")
                break
            self.logger.warning(f"Attempt {attempt + 1} did not get valid result")
            time.sleep(1)
        return {'insights': insights, 'comparison_failed': False}

    def compare_location_samples(self, location_paths: list[Path], problem_statement: str) -> dict:
        self.logger.info(f"Starting comparison of {len(location_paths)} location variant results")
        locations_data = []
        for i, location_path in enumerate(location_paths, 1):
            traj_file = location_path / f"{location_path.name}.traj"
            if not traj_file.exists():
                self.logger.warning(f"location-sample {i} trajectory does not exist: {traj_file}")
                continue
            try:
                data = json.loads(traj_file.read_text())
                locations_data.append({
                    'index': i,
                    'path': location_path,
                    'trajectory': data.get('trajectory', []),
                    'info': data.get('info', {}),
                    'submission': data.get('info', {}).get('submission', ''),
                    'exit_status': data.get('info', {}).get('exit_status', 'unknown')
                })
                self.logger.info(f"Successfully read location-sample {i}: {len(data.get('trajectory', []))} steps, exit: {data.get('info', {}).get('exit_status', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Failed to read location-sample {i}: {e}")
                continue
        if not locations_data:
            self.logger.error("No valid location-sample data")
            return {'insights': 'No valid location-sample data', 'comparison_failed': True, 'total_locations': len(location_paths), 'valid_locations': 0, 'best_location_index': None}
        comparison_prompt = self._build_location_comparison_prompt(problem_statement, locations_data, len(locations_data))
        messages = [{"role": "user", "content": comparison_prompt}]
        insights = ""
        best_location_index = None
        max_parse_retries = 6
        for attempt in range(max_parse_retries):
            insights = self._call_api(messages)
            self.logger.info(f"location-sample comparison insights (attempt {attempt + 1}/{max_parse_retries}):\n{insights}")
            if not insights.strip():
                self.logger.warning(f"Attempt {attempt + 1} did not get valid result")
                time.sleep(1)
                continue
            best_location_index = self._extract_best_location_index(insights, locations_data)
            if best_location_index is not None:
                self.logger.info(f"Attempt {attempt + 1} successfully parsed best location: Location {best_location_index}")
                break
            else:
                self.logger.warning(f"Attempt {attempt + 1} could not parse valid best location, regenerating insights")
                messages.append({"role": "assistant", "content": insights})
                messages.append({
                    "role": "user",
                    "content": f"""Your previous response did not contain a clear "BEST_SAMPLE: [number]" specification or the sample number was invalid.
    Please provide your analysis again, and make sure to include a clear line with exactly this format:
    BEST_SAMPLE: [number]
    Where [number] must be one of: {', '.join(str(loc['index']) for loc in locations_data)}
    Please revise your analysis and ensure the BEST_SAMPLE line is present and uses a valid sample number."""})
                time.sleep(1)
        if best_location_index is None:
            self.logger.error(f"After {max_parse_retries} attempts still cannot parse valid best location, using default strategy")
            if not insights.strip():
                insights = "Comparison analysis failed, please solve based on original problem statement"
            for loc in locations_data:
                if loc['submission']:
                    best_location_index = loc['index']
                    break
            if best_location_index is None:
                best_location_index = locations_data[0]['index']
            self.logger.warning(f"Using default strategy selected Location {best_location_index}")
        return {'insights': insights, 'comparison_failed': best_location_index is None and not insights.strip(), 'total_locations': len(location_paths), 'valid_locations': len(locations_data), 'best_location_index': best_location_index}
    
    def _build_location_comparison_prompt(self, problem_statement: str, samples: list[dict], num_samples: int) -> str:
        """Build comparison analysis prompt"""
        prompt = f"""You are an expert code debugging analyst. Compare multiple solution attempts for the same SWE-bench task and identify the best approach.

## Problem Statement
{problem_statement}

## Solution Attempts

"""
        for sample in samples:
            prompt += f"""### Sample {sample['index']}
**Execution Summary:** {len(sample['trajectory'])} steps, Exit: {sample['exit_status']}, Patch: {'Yes' if sample['submission'] else 'No'}

**Trajectory Summary:**
"""
            traj = sample['trajectory']
            for i, step in enumerate(traj, 1):
                prompt += f"Step {i}: {step.get('thought', 'N/A')[:200]}... -> {step.get('action', 'N/A')[:200]}...\n"
            if sample['submission']:
                prompt += f"\n**Patch:**\n```\n{sample['submission'][:1000]}{'...' if len(sample['submission']) > 1000 else ''}\n```\n\n"
            else:
                prompt += "\n**Patch:** No submission\n\n"
        prompt += f"""## Analysis Requirements

Compare all samples and provide:

1. **Best Sample Selection:** Which sample number (1-{num_samples}) has the most correct solution?

2. **Reasoning:** Why is this sample better than others? Consider:
   - Root cause resolution rather than superficial fixes
   - Patch correctness and completeness
   - Problem understanding accuracy
   - Debugging approach effectiveness
   - Edge case handling
   
3. **Key Insights for Next Attempt:** What can we learn from all attempts to guide the next solution? Be specific about:
   - Which approaches worked well
   - Specific files/functions that need attention
   - Testing strategies that should be used

## Output Format

**BEST_SAMPLE:** [number]

**REASONING:**
[Detailed comparison explaining why this sample is best]

**INSIGHTS FOR NEXT ATTEMPT:**
[Specific, actionable guidance for the {num_samples + 1}th attempt based on learnings from all {num_samples} samples]

Keep response under 800 words. Focus on concrete technical differences and actionable insights.
"""
        return prompt

    def _build_comparison_prompt(self, problem_statement: str, samples: list[dict], num_samples: int) -> str:
        prompt = f"""You are an expert code debugging analyst. Compare multiple solution attempts for the same SWE-bench task and identify the best approach.

## Problem Statement
{problem_statement}

## Solution Attempts

"""
        for sample in samples:
            prompt += f"""### Sample {sample['index']}
**Execution Summary:** {len(sample['trajectory'])} steps, Exit: {sample['exit_status']}, Patch: {'Yes' if sample['submission'] else 'No'}

**Trajectory Summary:**
"""
            traj = sample['trajectory']
            for i, step in enumerate(traj, 1):
                prompt += f"Step {i}: {step.get('thought', 'N/A')[:200]}... -> {step.get('action', 'N/A')[:200]}...\n"
            
            if sample['submission']:
                prompt += f"\n**Patch:**\n```\n{sample['submission'][:1000]}{'...' if len(sample['submission']) > 1000 else ''}\n```\n\n"
            else:
                prompt += "\n**Patch:** No submission\n\n"
        
        prompt += f"""## Analysis Requirements

Compare all samples and provide:

1. **Best Sample Selection:** Which sample number (1-{num_samples}) has the most correct solution?

2. **Reasoning:** Why is this sample better than others? Consider:
   - Root cause resolution rather than superficial fixes
   - Patch correctness and completeness
   - Problem understanding accuracy
   - Debugging approach effectiveness
   - Edge case handling

3. **Key Insights for Next Attempt:** What can we learn from all attempts to guide the next solution? Be specific about:
   - Which approaches worked well
   - Common mistakes to avoid
   - Specific files/functions that need attention
   - Testing strategies that should be used

## Output Format

**BEST_SAMPLE:** [number]

**REASONING:**
[Detailed comparison explaining why this sample is best]

**INSIGHTS FOR NEXT ATTEMPT:**
[Specific, actionable guidance for the {num_samples + 1}th attempt based on learnings from all {num_samples} samples]

Keep response under 800 words. Focus on concrete technical differences and actionable insights.
"""
        return prompt
    
    def _extract_best_sample_index(self, insights: str, valid_samples: list[dict]) -> int:
        """Extract best sample index from insights"""
        import re
        match = re.search(r'BEST_SAMPLE:\s*(\d+)', insights, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if any(s['index'] == idx for s in valid_samples):
                self.logger.info(f"Extracted best sample from analysis: Sample {idx}")
                return idx
        self.logger.warning("Could not extract best sample from analysis, using default strategy")
        for sample in valid_samples:
            if sample['submission']:
                return sample['index']
        return valid_samples[0]['index']


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
    comparison_api_key: str = ""
    comparison_api_base: str = ""
    comparison_model_name: str = ""
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")
    
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
    def __init__(self, instances: list[BatchInstance], agent_config: AgentConfig, *, output_dir: Path = Path("."), hooks: list[RunHook] | None = None, raise_exceptions: bool = False, redo_existing: bool = False, num_workers: int = 1, progress_bar: bool = True, random_delay_multiplier: float = 0.3):
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
        self._num_workers = 10
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml")
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier
        self.enable_multi_sampling = False
        self.num_samples = 5
        self.enable_best_sample_selection = False
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
        if SPECIFIC_INSTANCE_IDS:
            instances = filter_instances_by_ids(instances, SPECIFIC_INSTANCE_IDS)
            if not instances:
                msg = f"No instances found after filtering. Please check specified instance IDs: {sorted(SPECIFIC_INSTANCE_IDS)}"
                raise ValueError(msg)
        if config.enable_multi_sampling:
            logger.info(f"Multi-sampling mode enabled, sampling {config.num_samples} times per task")
            all_instances = []
            single_location_tasks = set()
            multi_location_tasks = {}
            if config.enable_best_sample_selection:
                for instance in instances:
                    original_id = instance.problem_statement.id
                    deduplicated_patches_path = Path("")
                    if deduplicated_patches_path.exists():
                        try:
                            location_patches = []
                            with open(deduplicated_patches_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        location_patches.append(json.loads(line))
                            logger.info(f"Found {len(location_patches)} location variants for {original_id}")
                            if len(location_patches) == 1:
                                single_location_tasks.add(original_id)
                                logger.info(f"✓ {original_id} has only 1 location, will create 5 sample instances")
                            else:
                                multi_location_tasks[original_id] = location_patches
                                logger.info(f"✓ {original_id} has {len(location_patches)} locations, will create location instances")
                        except Exception as e:
                            logger.error(f"Failed to read deduplicated_patches.jsonl for {original_id}: {e}")
                    else:
                        logger.warning(f"Could not find deduplicated_patches.jsonl for {original_id}: {deduplicated_patches_path}")
            for instance in instances:
                original_id = instance.problem_statement.id
                if config.enable_best_sample_selection and original_id in single_location_tasks:
                    logger.info(f"Creating 5 sample instances for single-location task {original_id}")
                    for sample_idx in range(1, 6):
                        sample_instance = copy.deepcopy(instance)
                        sample_instance.problem_statement.id = f"{original_id}-sample{sample_idx}"
                        all_instances.append(sample_instance)
                elif config.enable_best_sample_selection and original_id in multi_location_tasks:
                    sample_instance = copy.deepcopy(instance)
                    sample_instance.problem_statement.id = f"{original_id}-sample1"
                    all_instances.append(sample_instance)
        
                    location_patches = multi_location_tasks[original_id]
                    for loc_idx, patch_data in enumerate(location_patches, 1):
                        loc_instance = copy.deepcopy(instance)
                        loc_instance.problem_statement.id = f"{original_id}-location-{loc_idx}"
                        loc_instance.location_patch_data = patch_data
                        all_instances.append(loc_instance)
                else:
                    for sample_idx in range(1, config.num_samples + 1):
                        sample_instance = copy.deepcopy(instance)
                        sample_instance.problem_statement.id = f"{original_id}-sample{sample_idx}"
                        all_instances.append(sample_instance)
            instances = all_instances
            logger.info(f"Created {len(instances)} instances")
            if config.enable_best_sample_selection:
                logger.info(f"- Single-location tasks: {len(single_location_tasks)}")
                logger.info(f"- Multi-location tasks: {len(multi_location_tasks)}")
        if not instances:
            msg = "No instances to run."
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(instances=instances, agent_config=config.agent, output_dir=config.output_dir, raise_exceptions=config.raise_exceptions, redo_existing=config.redo_existing, num_workers=config.num_workers, progress_bar=config.progress_bar, random_delay_multiplier=config.random_delay_multiplier)
        rb.enable_multi_sampling = config.enable_multi_sampling
        rb.num_samples = config.num_samples
        rb.enable_best_sample_selection = config.enable_best_sample_selection
        if config.enable_best_sample_selection:
            rb._comparator_config = {'api_key': config.comparison_api_key, 'base_url': config.comparison_api_base, 'model_name': config.comparison_model_name}
        if isinstance(config.instances, SWEBenchInstances) and config.instances.evaluate:
            from sweagent.run.hooks.swe_bench_evaluate import SweBenchEvaluate
            rb.add_hook(SweBenchEvaluate(output_dir=config.output_dir, subset=config.instances.subset, split=config.instances.split, continuous_submission_every=1))
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

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
        if self.enable_multi_sampling and self.enable_best_sample_selection:
            self.logger.info(f"=== Phase 1: Execute all samples ===")
            self.main_phase_samples()
        elif self._num_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()
        output_dirs = []
        for instance in self.instances:
            output_dirs.append(self.output_dir / instance.problem_statement.id)
        merge_predictions(output_dirs, self.output_dir / "preds.json")
        self._chooks.on_end()

    def main_phase_samples(self) -> None:
        """Phase 1: Execute sample1-5 for single-location tasks and sample1 for multi-location tasks"""
        sample_instances = []
        for inst in self.instances:
            instance_id = inst.problem_statement.id
            if '-location-' in instance_id or '-sample-final' in instance_id:
                continue
            if '-sample' in instance_id:
                original_id = instance_id.split('-sample')[0]
                multiple_locations = any(other_inst.problem_statement.id.startswith(original_id + '-location-') for other_inst in self.instances)
                if multiple_locations:
                    if '-sample1' in instance_id:
                        sample_instances.append(inst)
                        self.logger.debug(f"Adding sample1 for multi-location task: {instance_id}")
                else:
                    sample_instances.append(inst)
                    self.logger.debug(f"Adding sample for single-location task: {instance_id}")
        self.logger.info(f"Phase 1: Checking {len(sample_instances)} sample instances")
        instances_to_run = []
        skipped_count = 0
        for inst in sample_instances:
            should_skip, reason = self.should_skip_instance(inst)
            if should_skip:
                self.logger.info(f"⏭️ Skipping completed instance: {inst.problem_statement.id} ({reason})")
                skipped_count += 1
            else:
                instances_to_run.append(inst)
        self.logger.info(f"Phase 1: Skipped {skipped_count} completed instances, executing {len(instances_to_run)} new instances")
        if not instances_to_run:
            self.logger.info("Phase 1: All instances completed, no need to execute")
            return
        if self._num_workers <= 1:
            self._run_instances_sequential(instances_to_run)
        else:
            self._run_instances_parallel(instances_to_run)

    def main_phase_locations(self) -> None:
        """Phase 2: Execute all location variants in parallel"""
        location_instances = [inst for inst in self.instances if '-location-' in inst.problem_statement.id]
        self.logger.info(f"Phase 2: Checking {len(location_instances)} location variants")
        instances_to_run = []
        skipped_count = 0
        for inst in location_instances:
            should_skip, reason = self.should_skip_instance(inst)
            if should_skip:
                self.logger.info(f"⏭️ Skipping completed instance: {inst.problem_statement.id} ({reason})")
                skipped_count += 1
            else:
                instances_to_run.append(inst)
        self.logger.info(f"Phase 2: Skipped {skipped_count} completed instances, executing {len(instances_to_run)} new instances")
        if not instances_to_run:
            self.logger.info("Phase 2: All location instances completed, no need to execute")
            return
        max_workers_for_locations = min(8, len(instances_to_run))
        if max_workers_for_locations <= 1:
            for instance in instances_to_run:
                self.run_instance_with_location_constraint(instance)
        else:
            add_logger_names_to_stream_handlers()
            set_stream_handler_levels(logging.WARNING)
            self.logger.setLevel(logging.TRACE)
            with Live(self._progress_manager.render_group):
                with ThreadPoolExecutor(max_workers=max_workers_for_locations) as executor:
                    futures = [executor.submit(self.run_instance_with_location_constraint, instance) for instance in instances_to_run]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except (KeyboardInterrupt, _BreakLoop):
                        self.logger.info("Received interrupt signal")
                        executor.shutdown(wait=False, cancel_futures=True)
                    finally:
                        self._progress_manager.print_report()

    def run_instance_with_location_constraint(self, instance: BatchInstance) -> None:
        """Phase 2.1: Execute instance with location constraint"""
        instance_id = instance.problem_statement.id
        should_skip, skip_reason = self.should_skip_instance(instance)
        if should_skip:
            self.logger.info(f"⏭️ Skipping {instance_id}: {skip_reason}")
            self._progress_manager.on_instance_start(instance_id)
            self._progress_manager.on_instance_end(instance_id, exit_status=skip_reason)
            return
        self.logger.info(f"Executing location variant: {instance_id}")
        register_thread_name(instance_id)
        self._add_instance_log_file_handlers(instance_id, multi_worker=True)
        if self._progress_manager.n_completed < 5:
            time.sleep(random.random() * self._random_delay_multiplier * 4)
        self._progress_manager.on_instance_start(instance_id)
        try:
            patch_data = getattr(instance, 'location_patch_data', None)
            if patch_data is None:
                self.logger.error(f"{instance_id} missing location_patch_data")
                raise ValueError(f"Missing location_patch_data for {instance_id}")
            location_insights = self._generate_location_insights_from_patch(patch_data)
            enhanced_agent_config = self._enhance_agent_config_with_location_insights(self.agent_config, location_insights)
            result = self._run_single_instance_with_config(instance, enhanced_agent_config)
            self._progress_manager.on_instance_end(instance_id, exit_status=result.info.get("exit_status", "unknown_exit"))
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
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance_id)

    def _generate_location_insights_from_patch(self, patch_data: dict) -> str:
        """Phase 2.2: Generate location insights from patch data"""
        insights_parts = []
        files_to_modify = patch_data.get('files_to_modify', [])
        insights_parts.append("**Error Code Location Information:**\n\nI will provide you with the location information of the buggy code below. Please make your modifications at the locations I specify for the erroneous code.\n\n")
        for file_info in files_to_modify:
            file_path = file_info.get('file_path', 'unknown')
            modifications = file_info.get('modifications', [])
            insights_parts.append(f"### File: `{file_path}`\n")
            for idx, mod in enumerate(modifications, 1):
                location = mod.get('location', {})
                current_code = mod.get('current_code', '').strip()
                insights_parts.append(f"**Modification {idx}:**")
                insights_parts.append(f"- **Current Code:**")
                insights_parts.append(f"```python")
                insights_parts.append(current_code)
                insights_parts.append("```")
        insights_parts.append("\n**Instructions:**")
        insights_parts.append("- Please make modifications at the code locations mentioned above")
        insights_parts.append("- Do NOT make changes outside these locations!")
        return "\n".join(insights_parts)

    def main_phase_locations_round2(self) -> None:
        """Phase 2.5: Execute second round based on first round location trajectories"""
        location_instances_round1 = [inst for inst in self.instances if '-location-' in inst.problem_statement.id and '(2)' not in inst.problem_statement.id]
        self.logger.info(f"Phase 2.5: Creating second round instances for {len(location_instances_round1)} locations")
        location_round2_instances = []
        for inst in location_instances_round1:
            original_location_id = inst.problem_statement.id
            round2_location_id = original_location_id.replace('-location-', '-location-') + '(2)'
            is_complete, status = self._check_trajectory_complete(round2_location_id)
            if is_complete and not self._redo_existing:
                self.logger.info(f"⏭️ Skipping completed second round location: {round2_location_id} (status: {status})")
                continue
            round2_instance = copy.deepcopy(inst)
            round2_instance.problem_statement.id = round2_location_id
            round2_instance._first_round_location_id = original_location_id
            location_round2_instances.append(round2_instance)
        self.logger.info(f"Phase 2.5: Need to execute {len(location_round2_instances)} second round location instances")
        if not location_round2_instances:
            self.logger.info("Phase 2.5: All second round location instances completed, no need to execute")
            return
        max_workers_for_locations = min(8, len(location_round2_instances))
        if max_workers_for_locations <= 1:
            for instance in location_round2_instances:
                self.run_instance_location_round2(instance)
        else:
            add_logger_names_to_stream_handlers()
            set_stream_handler_levels(logging.WARNING)
            self.logger.setLevel(logging.TRACE)
            with Live(self._progress_manager.render_group):
                with ThreadPoolExecutor(max_workers=max_workers_for_locations) as executor:
                    futures = [executor.submit(self.run_instance_location_round2, instance) for instance in location_round2_instances]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except (KeyboardInterrupt, _BreakLoop):
                        self.logger.info("Received interrupt signal")
                        executor.shutdown(wait=False, cancel_futures=True)
                    finally:
                        self._progress_manager.print_report()

    def run_instance_location_round2(self, instance: BatchInstance) -> None:
        """Execute second round location instance with reflection based on first round trajectory"""
        instance_id = instance.problem_statement.id
        first_round_id = getattr(instance, '_first_round_location_id', None)
        if not first_round_id:
            self.logger.error(f"{instance_id} missing first round location ID")
            return
        self.logger.info(f"Executing second round location: {instance_id} (based on {first_round_id})")
        register_thread_name(instance_id)
        self._add_instance_log_file_handlers(instance_id, multi_worker=True)
        if self._progress_manager.n_completed < 5:
            time.sleep(random.random() * self._random_delay_multiplier * 4)
        self._progress_manager.on_instance_start(instance_id)
        try:
            first_round_traj_path = self.output_dir / first_round_id / f"{first_round_id}.traj"
            if not first_round_traj_path.exists():
                self.logger.error(f"First round trajectory file not found: {first_round_traj_path}")
                raise FileNotFoundError(f"Missing first round trajectory: {first_round_traj_path}")
            first_round_data = json.loads(first_round_traj_path.read_text())
            first_round_trajectory = first_round_data.get('trajectory', [])
            first_round_submission = first_round_data.get('info', {}).get('submission', '')
            first_round_exit_status = first_round_data.get('info', {}).get('exit_status', 'unknown')
            reflection_insights = self._generate_reflection_insights_from_trajectory(first_round_id, first_round_trajectory, first_round_submission, first_round_exit_status)
            patch_data = getattr(instance, 'location_patch_data', None)
            if patch_data is None:
                self.logger.error(f"{instance_id} missing location_patch_data")
                raise ValueError(f"Missing location_patch_data for {instance_id}")
            location_insights = self._generate_location_insights_from_patch(patch_data)
            combined_insights = f"{reflection_insights}\n\n{location_insights}"
            enhanced_agent_config = self._enhance_agent_config_with_location_insights(self.agent_config, combined_insights)
            result = self._run_single_instance_with_config(instance, enhanced_agent_config)
            self._progress_manager.on_instance_end(instance_id, exit_status=result.info.get("exit_status", "unknown_exit"))
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
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance_id)

    def _generate_reflection_insights_from_trajectory(self, location_id: str, trajectory: list, submission: str, exit_status: str) -> str:
        """Generate reflection insights based on first round trajectory"""
        insights_parts = []
        insights_parts.append(f"**Reflection on Previous Attempt ({location_id}):**\n")
        insights_parts.append(f"The previous attempt at this location resulted in: {exit_status}\n")
        if submission:
            insights_parts.append("**Previous Patch Generated:**")
            insights_parts.append("```python")
            insights_parts.append(submission)
            insights_parts.append("```\n")
        else:
            insights_parts.append("**Note:** The previous attempt did not generate a patch.\n")
        insights_parts.append("**Trajectory Summary:**")
        insights_parts.append(f"The previous attempt took {len(trajectory)} steps:\n")
        for i, step in enumerate(trajectory, 1):
            thought = step.get('thought', 'N/A')[:200]
            action = step.get('action', 'N/A')[:200]
            insights_parts.append(f"{i}. Thought: {thought}... → Action: {action}...")
        insights_parts.append("\n**Instructions for This Attempt:**")
        insights_parts.append("- Learn from the previous debugging steps and approach")
        insights_parts.append("- Focus on generating a BETTER patch at the SAME location")
        return "\n".join(insights_parts)

    def analyze_best_locations(self) -> None:
        """Phase 3: Analyze location results for all tasks and generate best location mapping file"""
        original_ids = set()
        for inst in self.instances:
            if '-location-' in inst.problem_statement.id:
                original_id = inst.problem_statement.id.split('-location-')[0]
                original_ids.add(original_id)
        self.logger.info(f"Found {len(original_ids)} tasks requiring best location analysis")
        best_locations = {}
        skipped_count = 0
        for original_id in sorted(original_ids):
            analysis_dir = self.output_dir / f"{original_id}-sample-final"
            analysis_file = analysis_dir / "location_comparison_analysis.json"
            if analysis_file.exists():
                try:
                    analysis_content = analysis_file.read_text(encoding='utf-8')
                    if analysis_content.strip():
                        existing_analysis = json.loads(analysis_content)
                        if existing_analysis and 'best_location_index' in existing_analysis and existing_analysis['best_location_index'] is not None:
                            best_location_index = existing_analysis['best_location_index']
                            best_location_id = f"{original_id}-location-{best_location_index}"
                            best_locations[original_id] = best_location_id
                            self.logger.info(f"⏭️ Skipping already analyzed task: {original_id} (best location: {best_location_id})")
                            skipped_count += 1
                            continue
                except Exception as e:
                    self.logger.warning(f"Failed to read existing analysis file for {original_id}: {e}, will re-analyze")
            self.logger.info(f"Analyzing location results for {original_id}")
            location_paths = []
            for inst in self.instances:
                if inst.problem_statement.id.startswith(original_id + '-location-'):
                    location_path = self.output_dir / inst.problem_statement.id
                    if location_path.exists():
                        location_paths.append(location_path)
            if not location_paths:
                self.logger.warning(f"No location results found for {original_id}")
                continue
            self.logger.info(f"Found {len(location_paths)} location results for {original_id}")
            problem_statement = ""
            for inst in self.instances:
                if inst.problem_statement.id.startswith(original_id) and '-sample' in inst.problem_statement.id:
                    problem_statement = (getattr(inst.problem_statement, 'text', '') or getattr(inst.problem_statement, 'description', '') or str(inst.problem_statement))
                    break
            if self._comparator_config:
                comparator = MultiSampleComparator.get_thread_local_instance(api_key=self._comparator_config['api_key'], base_url=self._comparator_config['base_url'], model_name=self._comparator_config['model_name'])
            else:
                comparator = MultiSampleComparator.get_thread_local_instance()
            comparison_result = comparator.compare_location_samples(location_paths, problem_statement)
            best_location_index = comparison_result.get('best_location_index')
            if best_location_index is not None:
                best_location_id = f"{original_id}-location-{best_location_index}"
                best_locations[original_id] = best_location_id
                self.logger.info(f"✓ Best location for {original_id}: {best_location_id}")
                analysis_dir.mkdir(parents=True, exist_ok=True)
                analysis_data = {
                    'task_id': original_id,
                    'best_location': best_location_id,
                    'best_location_index': best_location_index,
                    'total_locations': comparison_result['total_locations'],
                    'valid_locations': comparison_result['valid_locations'],
                    'insights': comparison_result['insights'],
                    'comparison_failed': comparison_result['comparison_failed']
                }
                analysis_file.write_text(json.dumps(analysis_data, indent=2, ensure_ascii=False), encoding='utf-8')
                self.logger.info(f"Analysis result saved: {analysis_file}")
            else:
                self.logger.warning(f"✗ Could not determine best location for {original_id}")
        self.logger.info(f"Phase 3 analysis complete: skipped {skipped_count} already analyzed tasks, newly analyzed {len(best_locations) - skipped_count} tasks")
        best_locations_file = self.output_dir / "best_locations_mapping.jsonl"
        existing_mappings = {}
        if best_locations_file.exists():
            try:
                with open(best_locations_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            mapping = json.loads(line)
                            existing_mappings.update(mapping)
                self.logger.info(f"Read {len(existing_mappings)} existing mappings from file")
            except Exception as e:
                self.logger.warning(f"Failed to read existing mapping file: {e}")
        new_mappings_count = 0
        for task_id, best_location_id in best_locations.items():
            if task_id not in existing_mappings:
                existing_mappings[task_id] = best_location_id
                new_mappings_count += 1
            else:
                self.logger.info(f"⏭️ Skipping existing mapping: {task_id} -> {existing_mappings[task_id]}")
        with open(best_locations_file, 'w', encoding='utf-8') as f:
            for task_id in sorted(existing_mappings.keys()):
                f.write(json.dumps({task_id: existing_mappings[task_id]}, ensure_ascii=False) + '\n')
        self.logger.info(f"✓ Best location mappings for all tasks saved: {best_locations_file}")
        self.logger.info(f"Total: added {new_mappings_count} new mappings, {len(existing_mappings) - new_mappings_count} existing mappings, {len(existing_mappings)} total mappings")

    def main_phase_final_samples(self) -> None:
        """Phase 3: Compare sample/location results and execute sample-final (modified to include second round locations)"""
        original_ids_from_locations = set()
        original_ids_from_samples = set()
        for inst in self.instances:
            if '-location-' in inst.problem_statement.id:
                temp_id = inst.problem_statement.id.replace('(2)', '')
                original_id = temp_id.split('-location-')[0]
                original_ids_from_locations.add(original_id)
            elif '-sample' in inst.problem_statement.id and '-sample-final' not in inst.problem_statement.id:
                parts = inst.problem_statement.id.split('-sample')
                if len(parts) == 2:
                    original_id = parts[0]
                    original_ids_from_samples.add(original_id)
        single_location_tasks = original_ids_from_samples - original_ids_from_locations
        multi_location_tasks = original_ids_from_locations
        all_original_ids = original_ids_from_samples | original_ids_from_locations
        self.logger.info(f"Phase 3: Generating final sample for {len(all_original_ids)} tasks")
        self.logger.info(f"- Single-location tasks (based on samples): {len(single_location_tasks)}")
        self.logger.info(f"- Multi-location tasks (based on locations + round2): {len(multi_location_tasks)}")
        final_sample_instances = []
        for original_id in sorted(all_original_ids):
            final_sample_id = f"{original_id}-sample-final"
            is_complete, status = self._check_trajectory_complete(final_sample_id)
            if is_complete and not self._redo_existing:
                self.logger.info(f"⏭️ Skipping completed sample-final: {final_sample_id} (status: {status})")
                continue
            template_instance = None
            for inst in self.instances:
                if inst.problem_statement.id.startswith(original_id) and '-sample' in inst.problem_statement.id:
                    template_instance = inst
                    break
            if template_instance is None:
                self.logger.warning(f"Could not find template instance for {original_id}, skipping sample-final creation")
                continue
            final_sample_instance = copy.deepcopy(template_instance)
            final_sample_instance.problem_statement.id = final_sample_id
            if original_id in single_location_tasks:
                final_sample_instance._is_single_location_task = True
            else:
                final_sample_instance._is_single_location_task = False
            final_sample_instances.append(final_sample_instance)
        self.logger.info(f"Phase 3: Need to execute {len(final_sample_instances)} sample-final")
        if not final_sample_instances:
            self.logger.info("Phase 3: All sample-final completed, no need to execute")
            return
        max_workers_for_final = min(8, len(final_sample_instances))
        if max_workers_for_final <= 1:
            for instance in final_sample_instances:
                self.run_instance_with_comparison(instance)
        else:
            add_logger_names_to_stream_handlers()
            set_stream_handler_levels(logging.WARNING)
            self.logger.setLevel(logging.TRACE)
            with Live(self._progress_manager.render_group):
                with ThreadPoolExecutor(max_workers=max_workers_for_final) as executor:
                    futures = [executor.submit(self.run_instance_with_comparison, instance) for instance in final_sample_instances]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except (KeyboardInterrupt, _BreakLoop):
                        self.logger.info("Received interrupt signal")
                        executor.shutdown(wait=False, cancel_futures=True)
                    finally:
                        self._progress_manager.print_report()

    def run_instance_with_comparison(self, instance: BatchInstance) -> None:
        """Compare sample or location results and execute sample-final (modified to include second round locations)"""
        final_sample_id = instance.problem_statement.id
        original_id = final_sample_id.replace('-sample-final', '')
        should_skip, skip_reason = self.should_skip_instance(instance)
        if should_skip:
            self.logger.info(f"⏭️ Skipping {final_sample_id}: {skip_reason}")
            self._progress_manager.on_instance_start(final_sample_id)
            self._progress_manager.on_instance_end(final_sample_id, exit_status=skip_reason)
            return
        is_single_location = getattr(instance, '_is_single_location_task', False)
        if is_single_location:
            self.logger.info(f"Executing sample-final for {original_id} (single-location task)")
        else:
            self.logger.info(f"Executing sample-final for {original_id} (multi-location task, including second round)")
        register_thread_name(final_sample_id)
        self._add_instance_log_file_handlers(final_sample_id, multi_worker=True)
        if self._progress_manager.n_completed < 5:
            time.sleep(random.random() * self._random_delay_multiplier * 4)
        self._progress_manager.on_instance_start(final_sample_id)
        try:
            if is_single_location:
                sample_paths = []
                for sample_idx in range(1, 6):
                    sample_id = f"{original_id}-sample{sample_idx}"
                    sample_path = self.output_dir / sample_id
                    if sample_path.exists():
                        sample_paths.append(sample_path)
                if not sample_paths:
                    self.logger.warning(f"No sample results found for {original_id}, using original problem statement")
                    insights = ""
                else:
                    self.logger.info(f"Found {len(sample_paths)} sample results, starting comparison analysis")
                    if self._comparator_config:
                        comparator = MultiSampleComparator.get_thread_local_instance(api_key=self._comparator_config['api_key'], base_url=self._comparator_config['base_url'], model_name=self._comparator_config['model_name'])
                    else:
                        comparator = MultiSampleComparator.get_thread_local_instance()
                    problem_statement = (getattr(instance.problem_statement, 'text', '') or getattr(instance.problem_statement, 'description', '') or str(instance.problem_statement))
                    num_samples = len(sample_paths)
                    comparison_result = comparator.compare_samples(sample_paths, problem_statement, num_samples)
                    insights = comparison_result.get('insights', '')
                    self.logger.info(f"Generated sample comparison insights (length: {len(insights)})")
            else:
                location_paths = []
                for inst in self.instances:
                    inst_id = inst.problem_statement.id
                    if inst_id.startswith(original_id + '-location-') and '(2)' not in inst_id:
                        location_path = self.output_dir / inst_id
                        if location_path.exists():
                            location_paths.append(location_path)
                    elif inst_id.startswith(original_id + '-location-') and '(2)' in inst_id:
                        location_path = self.output_dir / inst_id
                        if location_path.exists():
                            location_paths.append(location_path)
                if not location_paths:
                    self.logger.warning(f"No location results found for {original_id}, using original problem statement")
                    insights = ""
                else:
                    self.logger.info(f"Found {len(location_paths)} location results (including both rounds), starting comparison analysis")
                    if self._comparator_config:
                        comparator = MultiSampleComparator.get_thread_local_instance(api_key=self._comparator_config['api_key'], base_url=self._comparator_config['base_url'], model_name=self._comparator_config['model_name'])
                    else:
                        comparator = MultiSampleComparator.get_thread_local_instance()
                    problem_statement = (getattr(instance.problem_statement, 'text', '') or getattr(instance.problem_statement, 'description', '') or str(instance.problem_statement))
                    num_locations = len(location_paths)
                    comparison_result = comparator.compare_location_samples(location_paths, problem_statement)
                    insights = comparison_result.get('insights', '')
                    self.logger.info(f"Generated location comparison insights (length: {len(insights)})")
            insights_file = self.output_dir / final_sample_id / "comparison_insights.txt"
            insights_file.parent.mkdir(parents=True, exist_ok=True)
            insights_file.write_text(insights, encoding='utf-8')
            self.logger.info(f"Comparison insights saved: {insights_file}")
            if is_single_location:
                num_attempts = 5
            else:
                num_attempts = len([inst for inst in self.instances if inst.problem_statement.id.startswith(original_id + '-location-')])
            enhanced_agent_config = self._enhance_agent_config_with_insights(self.agent_config, insights, num_samples=num_attempts)
            result = self._run_single_instance_with_config(instance, enhanced_agent_config)
            self._progress_manager.on_instance_end(final_sample_id, exit_status=result.info.get("exit_status", "unknown_exit"))
        except KeyboardInterrupt:
            raise _BreakLoop
        except (SystemExit, ModelConfigurationError, TotalCostLimitExceededError) as e:
            if self._raise_exceptions:
                raise
            self.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"❌ Failed on {final_sample_id}: {e}")
            self._progress_manager.on_uncaught_exception(final_sample_id, e)
            if self._raise_exceptions:
                raise
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(final_sample_id)
    
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
        for instance in instances:
            try:
                self.run_instance(instance)
            except _BreakLoop:
                self.logger.info("Stopping loop over instances")
                break
    
    def _run_instances_parallel(self, instances: list[BatchInstance]) -> None:
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
        """Execute single instance with specified configuration"""
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
    
    def _enhance_agent_config_with_insights(self, original_config: AgentConfig, insights: str, num_samples: int) -> AgentConfig:
        """Enhance agent configuration with insights"""
        enhanced_config = original_config.model_copy(deep=True)
        original_instance_template = enhanced_config.templates.instance_template
        if insights.strip():
            enhanced_instance_template = f"""Based on {num_samples} previous solution attempts at different code locations for this task, we have conducted a comparative analysis to identify the best approach and key learnings:

{insights}

Please solve the task by leveraging these insights from previous attempts. Focus on:
1. Building upon the strengths identified in the best approach
2. Avoiding the common mistakes observed across attempts
3. Addressing the specific areas that need attention

# Original Task Instructions
{original_instance_template}
"""
        else:
            enhanced_instance_template = original_instance_template
        enhanced_config.templates.instance_template = enhanced_instance_template
        self.logger.info(f"Enhanced instance template with comparison insights (length: {len(insights)} chars)")
        return enhanced_config

    def _enhance_agent_config_with_location_insights(self, original_config: AgentConfig, location_insights: str) -> AgentConfig:
        """Enhance agent configuration with location insights"""
        enhanced_config = original_config.model_copy(deep=True)
        original_instance_template = enhanced_config.templates.instance_template
        enhanced_instance_template = f"""
# Original Task Description
{original_instance_template}

{location_insights}
"""
        enhanced_config.templates.instance_template = enhanced_instance_template
        self.logger.info(f"Enhanced instance template with location insights")
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