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
import re
import sys
import time
import copy
import json
import yaml
import random
import litellm
import asyncio
import getpass
import logging
import requests
import traceback
import threading
from typing import Self
from pathlib import Path
from rich.live import Live
from contextlib import ExitStack
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from concurrent.futures import ThreadPoolExecutor, as_completed


from sweagent import TRAJECTORY_DIR
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run.run_single import RunSingleConfig
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.run_regression_test import _run_regression
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances

from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (add_file_handler, add_logger_names_to_stream_handlers, get_logger, register_thread_name, remove_file_handler, set_stream_handler_levels)

try:
    from swerex.deployment.hooks.status import SetStatusDeploymentHook
except ImportError:
    SetStatusDeploymentHook = None

def filter_instances_by_ids(instances: list[BatchInstance], target_ids: set[str]) -> list[BatchInstance]:
    """
    过滤instances，只保留target_ids中指定的instance
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
        logger.info(f"找到以下指定的instance IDs: {sorted(found_ids)}")
    if missing_ids:
        logger.warning(f"未找到以下指定的instance IDs: {sorted(missing_ids)}")
    logger.info(f"过滤后的实例数量: {len(filtered_instances)} / {len(instances)}")
    return filtered_instances

class MultiSampleComparator:
    """对比多次采样结果并选择最佳轨迹的类"""
    _thread_local = threading.local()
    def __init__(self, api_key: str = None, base_url: str = "", model_name: str = ""):
        """初始化对比器"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model_name = model_name
        self.logger = get_logger("comparator", emoji="🔍")
        self.max_retries = 5
        self.retry_delay = 2
        
    @classmethod
    def get_thread_local_instance(cls, api_key: str = None, base_url: str = None, model_name: str = None):
        """获取线程本地的对比器实例"""
        if not hasattr(cls._thread_local, 'comparator'):
            cls._thread_local.comparator = cls(api_key=api_key, base_url=base_url, model_name=model_name)
            cls._thread_local.thread_id = threading.get_ident()
        return cls._thread_local.comparator
    
    def _call_api(self, messages: list[dict], temperature: float = 0.7, top_p: float = 1.0) -> str:
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
                self.logger.warning(f"API调用失败，重试 {retries + 1}/{self.max_retries} (错误: {str(e)})")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"API调用发生未知错误: {e}")
                break
        return ""

    # def _call_api(self, messages: list[dict], temperature: float = 0.7, top_p: float = 1.0) -> str:
    #     response = litellm.completion(model=self.model_name, messages=messages, api_base=self.base_url, api_key=self.api_key, temperature=0.0, top_p=1.0)
    #     content = response.choices[0].message.content
    #     return content

    def compare_samples(self, sample_paths: list[Path], problem_statement: str, num_samples: int) -> dict:
        self.logger.info(f"开始对比 {len(sample_paths)} 个samples的结果")
        sample_data = []
        for i, sample_path in enumerate(sample_paths, 1):
            traj_file = sample_path / f"{sample_path.name}.traj"
            if not traj_file.exists():
                self.logger.warning(f"sample {i} trajectory不存在: {traj_file}")
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
                self.logger.info(f"成功读取 sample {i}: {len(data.get('trajectory', []))} steps, exit: {data.get('info', {}).get('exit_status', 'unknown')}")
            except Exception as e:
                self.logger.error(f"读取 sample {i} 失败: {e}")
                continue
        if not sample_data:
            self.logger.error("没有有效的sample数据")
            return {'insights': '无有效sample数据', 'comparison_failed': True, 'total_locations': len(sample_paths), 'valid_locations': 0, 'best_location_index': None}
        comparison_prompt = self._build_comparison_prompt(problem_statement, sample_data, len(sample_data))
        messages = [{"role": "user", "content": comparison_prompt}]
        insights = ""
        for attempt in range(self.max_retries):
            insights = self._call_api(messages)
            self.logger.info(f"insights:\n{insights}")
            if insights.strip():
                self.logger.info(f"第 {attempt + 1} 次尝试成功获取对比结果")
                break
            self.logger.warning(f"第 {attempt + 1} 次尝试未获取有效结果")
            time.sleep(1)
        return {'insights': insights, 'comparison_failed': False}

    def compare_location_samples(self, location_paths: list[Path], problem_statement: str) -> dict:
        self.logger.info(f"开始对比 {len(location_paths)} 个location变体结果")
        locations_data = []
        for i, location_path in enumerate(location_paths, 1):
            traj_file = location_path / f"{location_path.name}.traj"
            if not traj_file.exists():
                self.logger.warning(f"location-sample {i} trajectory不存在: {traj_file}")
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
                self.logger.info(f"成功读取 location-sample {i}: {len(data.get('trajectory', []))} steps, exit: {data.get('info', {}).get('exit_status', 'unknown')}")
            except Exception as e:
                self.logger.error(f"读取 location-sample {i} 失败: {e}")
                continue
        if not locations_data:
            self.logger.error("没有有效的location-sample数据")
            return {'insights': '无有效location-sample数据', 'comparison_failed': True, 'total_locations': len(location_paths), 'valid_locations': 0, 'best_location_index': None}
        comparison_prompt = self._build_comparison_prompt(problem_statement, locations_data, len(locations_data))
        messages = [{"role": "user", "content": comparison_prompt}]
        insights = ""
        max_parse_retries = 6
        for attempt in range(max_parse_retries):
            insights = self._call_api(messages)
            self.logger.info(f"location-sample对比insights (attempt {attempt + 1}/{max_parse_retries}):\n{insights}")
            if not insights.strip():
                self.logger.warning(f"第 {attempt + 1} 次尝试未获取有效结果")
                time.sleep(2)
                continue
        return {'insights': insights, 'comparison_failed': not insights.strip(), 'total_locations': len(location_paths), 'valid_locations': len(locations_data)}

    def _build_comparison_prompt(self, problem_statement: str, samples: list[dict], num_samples: int, select_top_half: bool = False) -> str:
        if select_top_half:
            import math
            top_n = math.ceil(num_samples / 2)
            prompt = f"""You are an expert code debugging analyst. Compare multiple solution attempts for the same SWE-bench task and select the TOP {top_n} best patches (approximately half).

## Problem Statement
{problem_statement}

## Solution Attempts
**IMPORTANT:** The samples are presented in RANDOM ORDER to avoid position bias. Evaluate each solution independently based on technical merit only.
"""
            for sample in samples:
                prompt += f"""### Sample {sample['index']}
**Execution Summary:** {len(sample['trajectory'])} steps, Exit: {sample['exit_status']}, Patch: {'Yes' if sample['submission'] else 'No'}
"""
                if sample["submission"]:
                    prompt += f"\n**Patch:**\n```\n{sample['submission']}\n```\n\n"
                else:
                    prompt += "\n**Patch:** No submission\n\n"

            prompt += f"""## Analysis Requirements

Compare all samples and provide:

1. **Select Top {top_n} Samples:** Which sample numbers have the most correct solutions?

2. **Reasoning:** Why is this sample better than others? Consider:
   - Patch correctness and completeness
   - Problem understanding accuracy
   - Edge case handling
   
3. **Key Insights for Next Attempt:** What can we learn from all attempts to guide the next solution? Be specific about:
   - Which approaches worked well
   - Specific files/functions that need attention
   - Testing strategies that should be used

## Output Format
**You MUST strictly follow this format:**

**TOP_SAMPLES:** [comma-separated list, e.g., "1,3,5,7"]

**REASONING:**
[Explanation of why these {top_n} samples are best]
"""
        else:
            prompt = f"""You are an expert code debugging analyst. Compare multiple solution attempts for the same SWE-bench task and identify the best generation patch.

## Problem Statement
{problem_statement}

## Solution Attempts
**IMPORTANT:** The samples are presented in RANDOM ORDER to avoid position bias. Evaluate each solution independently based on technical merit only.
"""
            for sample in samples:
                prompt += f"""### Sample {sample['index']}
**Execution Summary:** {len(sample['trajectory'])} steps, Exit: {sample['exit_status']}, Patch: {'Yes' if sample['submission'] else 'No'}
"""
                if sample['submission']:
                    prompt += f"\n**Patch:**\n```\n{sample['submission']}\n```\n\n"
                else:
                    prompt += "\n**Patch:** No submission\n\n"
            prompt += f"""## Analysis Requirements

Compare all samples and provide:

1. **Best Sample Selection:** Which sample number (1-{num_samples}) has the most correct solution?

2. **Reasoning:** Why is this sample better than others? Consider:
   - Patch correctness and completeness
   - Problem understanding accuracy
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

Keep response under 600 words. Focus on concrete technical differences and actionable insights.
"""
        return prompt

    def _extract_best_sample_index(self, insights: str, valid_samples: list[dict]) -> int:
        """从insights中提取最佳样本索引"""
        import re
        pattern = re.compile(r'(?i)\*{0,2}\s*(?:best|bset)[_\s]*sample\s*(?::\s*\*{0,2}|\*{0,2}\s*:)\s*(\d+)\b')
        match = pattern.search(insights)
        if match:
            idx = int(match.group(1))
            if any(s['index'] == idx for s in valid_samples):
                self.logger.info(f"从分析结果中提取到最佳样本: Sample {idx}")
                return idx
        self.logger.warning("未能从 insights 中解析 BEST_SAMPLE / BSET_SAMPLE")
        return -1

    def _extract_top_samples_index(self, insights: str, valid_indices: list[int], expected_count: int) -> list[int]:
        """从insights中提取前N个最佳样本索引"""
        import re
        # match = re.search(r'TOP_SAMPLES:\s*\[?"?([0-9,\s]+)"?\]?', insights, re.IGNORECASE)
        match = re.search(r'\*{0,2}TOP_SAMPLES:\*{0,2}\s*\[?["\']?([0-9,\s]+)["\']?\]?', insights, re.IGNORECASE)
        if match:
            numbers_str = match.group(1)
            indices = []
            for num_str in numbers_str.split(','):
                num_str = num_str.strip()
                if num_str and num_str.isdigit():
                    idx = int(num_str)
                    if idx in valid_indices:
                        indices.append(idx)
            if len(indices) >= 2:
                self.logger.info(f"从insights中提取到 {len(indices)} 个最佳样本: {indices}")
                return indices[:expected_count]
        return []
    
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
    num_samples: int = 3
    enable_location_constrained_generation: bool = False
    comparison_api_key: str = Field(default="", description="API key for comparison model")
    comparison_api_base: str = Field(default="", description="Base URL for comparison API")
    comparison_model_name: str = Field(default="", description="Model name for comparison")
    specific_instance_ids: str = Field(default="", description="Comma-separated list of specific instance IDs to run")
    deduplicated_patches_root: Path = Field(default=Path("DEFAULT"), description=("Root directory that contains locate-stage outputs"))
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
        self._num_workers = 30
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml")
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier
        self.enable_multi_sampling = False
        self.num_samples = 3
        self.enable_location_constrained_generation = False
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
            specific_ids = set(id.strip() for id in config.specific_instance_ids.split(',') if id.strip())
            if specific_ids:
                instances = filter_instances_by_ids(instances, specific_ids)
                if not instances:
                    msg = f"过滤后没有找到任何指定的instances, 请检查指定的instance IDs: {sorted(specific_ids)}"
                    raise ValueError(msg)
        if config.enable_multi_sampling:
            logger.info(f"启用多次采样模式，每个task采样 {config.num_samples} 次")
            all_instances = []
            single_location_tasks = set()
            multi_location_tasks = {}
            if config.enable_location_constrained_generation:
                for instance in instances:
                    original_id = instance.problem_statement.id
                    deduplicated_patches_path = config.deduplicated_patches_root / f"{original_id}-sample5" / "deduplicated_patches_new_new.jsonl"
                    if deduplicated_patches_path.exists():
                        try:
                            location_patches = []
                            with open(deduplicated_patches_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        location_patches.append(json.loads(line))
                            if len(location_patches) == 1:
                                single_location_tasks.add(original_id)
                                logger.info(f"✓ {original_id} 只有1个location，将创建5个sample实例")
                            elif len(location_patches) > 1:
                                multi_location_tasks[original_id] = location_patches
                                logger.info(f"✓ {original_id} 有{len(location_patches)}个location (>1)，将创建location实例")
                        except Exception as e:
                            logger.error(f"读取 {original_id} 的 deduplicated_patches 文件失败: {e}")
                    else:
                        logger.warning(f"未找到 {original_id} 的 deduplicated_patches 文件")
            for instance in instances:
                original_id = instance.problem_statement.id
                if config.enable_location_constrained_generation and original_id in single_location_tasks:
                    logger.info(f"为单location任务 {original_id} 创建{config.num_samples}个sample实例")
                    for sample_idx in range(1, config.num_samples + 1):
                        sample_instance = copy.deepcopy(instance)
                        sample_instance.problem_statement.id = f"{original_id}-sample{sample_idx}"
                        all_instances.append(sample_instance)
                elif config.enable_location_constrained_generation and original_id in multi_location_tasks:
                    logger.info(f"为多location任务 {original_id} 创建1个sample + {len(multi_location_tasks[original_id])}个location实例")
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
            logger.info(f"创建了 {len(instances)} 个实例")
            if config.enable_location_constrained_generation:
                logger.info(f"- 单location任务: {len(single_location_tasks)} 个")
                logger.info(f"- 多location任务: {len(multi_location_tasks)} 个")
        if not instances:
            msg = "No instances to run."
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(instances=instances, agent_config=config.agent, output_dir=config.output_dir, raise_exceptions=config.raise_exceptions, redo_existing=config.redo_existing, num_workers=config.num_workers, progress_bar=config.progress_bar, random_delay_multiplier=config.random_delay_multiplier)
        rb.enable_multi_sampling = config.enable_multi_sampling
        rb.num_samples = config.num_samples
        rb.enable_location_constrained_generation = config.enable_location_constrained_generation
        if config.enable_location_constrained_generation:
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
            self.logger.debug(f"Trajectory文件不存在: {traj_path}")
            return False, "not_found"
        try:
            content = traj_path.read_text()
            if not content.strip():
                self.logger.warning(f"发现空的Trajectory文件: {traj_path}")
                return False, "empty"
            data = json.loads(content)
            exit_status = data.get("info", {}).get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Trajectory文件无有效退出状态: {traj_path} (status: {exit_status})")
                return False, "incomplete"
            trajectory = data.get("trajectory", [])
            if not trajectory:
                self.logger.warning(f"Trajectory为空: {traj_path}")
                return False, "empty_trajectory"
            self.logger.info(f"✓ Trajectory完整: {instance_id} (exit_status: {exit_status}, {len(trajectory)} steps)")
            return True, exit_status
        except json.JSONDecodeError as e:
            self.logger.error(f"Trajectory文件JSON格式错误: {traj_path}: {e}")
            return False, "json_error"
        except Exception as e:
            self.logger.error(f"检查trajectory失败: {traj_path}: {e}")
            return False, "check_error"

    def should_skip_instance(self, instance: BatchInstance) -> tuple[bool, str]:
        if self._redo_existing:
            return False, ""
        instance_id = instance.problem_statement.id
        is_complete, status = self._check_trajectory_complete(instance_id)
        if is_complete:
            return True, f"skipped (already_complete: {status})"
        return False, ""

    def _get_comparator(self):
        if self._comparator_config:
            return MultiSampleComparator.get_thread_local_instance(api_key=self._comparator_config['api_key'], base_url=self._comparator_config['base_url'], model_name=self._comparator_config['model_name'])
        else:
            return MultiSampleComparator.get_thread_local_instance()
        
    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()
        if self.enable_multi_sampling and self.enable_location_constrained_generation:
            self.logger.info(f"=== 阶段1: 执行所有 sample ===")
            self.main_phase_samples()
            self.logger.info(f"=== 阶段2: 并行执行所有 location-sample===")
            self.main_phase_locations()  
            self.logger.info(f"=== 阶段3: 并行生成 final samples ===")
            self.main_phase_final_samples()
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
        """阶段1: 执行单location任务的sample1-num_samples和多location任务的sample1"""
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
                        self.logger.debug(f"添加多location任务的sample1: {instance_id}")
                else:
                    sample_instances.append(inst)
                    self.logger.debug(f"添加单location任务的sample: {instance_id}")
        self.logger.info(f"阶段1：检查 {len(sample_instances)} 个 sample 实例")
        instances_to_run = []
        skipped_count = 0
        for inst in sample_instances:
            should_skip, reason = self.should_skip_instance(inst)
            if should_skip:
                self.logger.info(f"⏭️ 跳过已完成的实例: {inst.problem_statement.id} ({reason})")
                skipped_count += 1
            else:
                instances_to_run.append(inst)
        self.logger.info(f"阶段1：跳过 {skipped_count} 个已完成实例，执行 {len(instances_to_run)} 个新实例")
        if not instances_to_run:
            self.logger.info("阶段1：所有实例已完成，无需执行")
            return
        if self._num_workers <= 1:
            self._run_instances_sequential(instances_to_run)
        else:
            self._run_instances_parallel(instances_to_run)

    def main_phase_locations(self) -> None:
        """阶段2: 多轮淘汰式执行location变体"""
        original_ids_with_locations = {}
        for inst in self.instances:
            if '-location-' in inst.problem_statement.id:
                original_id = inst.problem_statement.id.split('-location-')[0]
                if original_id not in original_ids_with_locations:
                    original_ids_with_locations[original_id] = []
                original_ids_with_locations[original_id].append(inst)
        self.logger.info(f"阶段2：找到 {len(original_ids_with_locations)} 个多location任务")
        max_parallel_tasks = min(10, len(original_ids_with_locations))
        if max_parallel_tasks <= 1:
            for original_id, location_instances in original_ids_with_locations.items():
                self._process_single_task_locations(original_id, location_instances)
        else:
            self.logger.info(f"使用 {max_parallel_tasks} 个worker并行处理多个任务")
            with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
                futures = {executor.submit(self._process_single_task_locations, original_id, location_instances): original_id for original_id, location_instances in original_ids_with_locations.items()}
                completed = 0
                total = len(futures)
                for future in as_completed(futures):
                    original_id = futures[future]
                    try:
                        future.result()
                        completed += 1
                        self.logger.info(f"任务完成 {completed}/{total}: {original_id}")
                    except Exception as e:
                        self.logger.error(f"任务 {original_id} 执行失败: {e}")
                        self.logger.error(traceback.format_exc())

    def _process_single_task_locations(self, original_id: str, location_instances: list) -> None:
        """处理单个任务的所有location"""
        num_locations = len(location_instances)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"处理instance: {original_id}")
        self.logger.info(f"Location数量: {num_locations}")
        self.logger.info(f"{'='*60}")
        self._run_multi_round_location_elimination(original_id, location_instances)

    def _run_multi_round_location_elimination(self, original_id: str, initial_locations: list) -> None:
        """多轮淘汰直到只剩1个location"""
        current_round = 0
        current_locations = initial_locations
        self.logger.info(f"\n开始多轮淘汰流程: {original_id}")
        self.logger.info(f"初始location数量: {len(current_locations)}")
        while len(current_locations) > 1:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Round {current_round}: {len(current_locations)} locations")
            self.logger.info(f"{'='*50}")
            self._execute_locations_batch_parallel(current_locations, round_num=current_round)
            locations_data  = self._collect_location_trajectories(current_locations, round_num=current_round)
            if len(locations_data) < 1:
                self.logger.warning(f"没有有效结果，停止淘汰")
                break
            if len(current_locations) == 2:
                top_indices = self._select_top_half_locations(original_id, locations_data, round_num=current_round, select_count=1)
                if len(top_indices) == 1:
                    # current_locations = self._prepare_next_round_locations(original_id, current_locations, top_indices, round_num=current_round + 1)
                    # current_round += 1
                    # self.logger.info(f"\nRound {current_round}: 最后1个location")
                    # self._execute_locations_batch_parallel(current_locations, round_num=current_round)
                    self.logger.info(f"淘汰完成，剩余1个location")
                    break
                else:
                    self.logger.warning(f"选择失败，停止淘汰")
                    break
            else:
                top_indices = self._select_top_half_locations(original_id, locations_data, round_num=current_round)
                current_locations = self._prepare_next_round_locations(original_id, current_locations, top_indices, round_num=current_round + 1)
                current_round += 1
            if current_round > 10:
                self.logger.error("超过最大轮次10，强制停止")
                break
        self.logger.info(f"\n{original_id} 淘汰流程完成")
        self.logger.info(f"总轮次: {current_round}")
        self.logger.info(f"最终剩余: {len(current_locations)} 个location")

    def _execute_locations_batch(self, locations: list, round_num: int) -> None:
        """并行执行一批locations"""
        instances_to_run = [loc for loc in locations if not self.should_skip_instance(loc)[0]]
        if not instances_to_run:
            return
        max_workers = min(15, len(instances_to_run))
        if max_workers <= 1:
            for loc in instances_to_run:
                self.run_instance_with_location_constraint(loc)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.run_instance_with_location_constraint, loc) for loc in instances_to_run]
                for future in as_completed(futures):
                    future.result()

    def _collect_location_trajectories(self, locations: list, round_num: int) -> list[dict]:
        locations_data = []
        # select_regression_file = ""
        # select_regression_dict = dict()
        # with open(select_regression_file, 'r', encoding='utf-8') as f:
        #     for item in f:
        #         item = json.loads(item)
        #         select_regression_dict[item['instance_id']] = dict()
        #         select_regression_dict[item['instance_id']]['tests_passing_in_original_repo'] = item['tests_passing_in_original_repo']
        for loc in locations:
            instance_id = loc.problem_statement.id
            original_id = re.sub(r'-location-\d+.*$', '', instance_id)
            # loc_item = select_regression_dict[original_id]
            traj_file = self.output_dir / instance_id / f"{instance_id}.traj"
            if not traj_file.exists():
                continue
            try:
                data = json.loads(traj_file.read_text())
                match = re.search(r'-location-(\d+)', instance_id)
                if not match:
                    continue
                loc_idx = int(match.group(1))
                locations_data.append({
                    'index': loc_idx,
                    'instance_id': instance_id,
                    'original_id': original_id,
                    'submission': data.get('info', {}).get('submission', ''),
                    'exit_status': data.get('info', {}).get('exit_status', 'unknown'),
                    'trajectory': data.get('trajectory', []),
                    'round': round_num,
                })
            except Exception as e:
                self.logger.error(f"读取失败 {instance_id}: {e}")
        return locations_data

    def _select_top_half_locations(self, original_id: str, locations_data: list[dict], round_num: int, select_count: int = None) -> list[int]:
        if len(locations_data) < 1:
            return []
        if len(locations_data) == 1:
            return [locations_data[0]['index']]
        problem_statement = ""
        for inst in self.instances:
            if inst.problem_statement.id.startswith(original_id) and '-sample' in inst.problem_statement.id:
                problem_statement = str(inst.problem_statement)
                break
        comparator = self._get_comparator()
        if select_count is not None:
            expected_count = select_count
        else:
            import math
            expected_count = math.ceil(len(locations_data) / 2)
        self.logger.info(f"Round{round_num}: 从{len(locations_data)}个location中选{expected_count}个")
        # use regression test
        # for data in locations_data:
        #     regression_dict = _run_regression(data)
        #     data['regression_result'] = regression_dict
        analysis_dir = self.output_dir / f"{original_id}-analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = analysis_dir / f"round{round_num}_comparison.json"
        if analysis_file.exists() and analysis_file.stat().st_size > 0:
            try:
                with open(analysis_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                selected_indices = data.get("selected_indices", [])
                self.logger.info(f"复用已有 {analysis_file.name}，selected_indices={selected_indices}")
                return selected_indices
            except Exception as e:
                self.logger.warning(f"读取已有 {analysis_file.name} 失败，将重新生成: {e}")
        comparison_prompt = comparator._build_comparison_prompt(problem_statement, locations_data, len(locations_data), select_top_half=(expected_count > 1))
        messages = [{"role": "user", "content": comparison_prompt}]
        valid_indices = [loc['index'] for loc in locations_data]
        max_retries = 8
        insights = ""
        top_indices = []
        for attempt in range(max_retries):
            insights = comparator._call_api(messages)
            self.logger.info(f"Round{round_num} attempt {attempt + 1}/{max_retries}: 获取insights (length: {len(insights)})")
            if not insights.strip():
                self.logger.warning(f"Round{round_num} attempt {attempt + 1}: insights为空，重试...")
                time.sleep(1)
                continue
            if expected_count == 1:
                extracted_idx = comparator._extract_best_sample_index(insights, locations_data)
                if extracted_idx is not None:
                    top_indices = [extracted_idx]
            else:
                top_indices = comparator._extract_top_samples_index(insights, valid_indices, expected_count)
            if len(top_indices) >= expected_count:
                self.logger.info(f"Round{round_num} attempt {attempt + 1}: 成功解析出 {len(top_indices)} 个有效indices: {top_indices}")
                break
            else:
                self.logger.warning(f"Round{round_num} attempt {attempt + 1}: 无法解析有效格式，重试...")
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": insights})
                    format_hint = "BEST_SAMPLE: [number]" if expected_count == 1 else f"TOP_SAMPLES: [number1, number2, ...]"
                    messages.append({
                        "role": "user",
                        "content": f"""Your previous response did not contain a clear "{format_hint}" specification or the format was incorrect.
Valid sample numbers are: {', '.join(map(str, valid_indices))}
You need to select the TOP {expected_count} best sample(s).
Please provide your analysis again, and make sure to include a clear line with exactly this format:
{format_hint}
For example: {"BEST_SAMPLE: 2" if expected_count == 1 else "TOP_SAMPLES: [1, 3, 4]"}
The numbers must be from the valid sample numbers listed above."""})
                    time.sleep(1)
            if not top_indices or len(top_indices) < expected_count:
                self.logger.error(f"Round{round_num}: 经过 {max_retries} 次尝试仍无法解析有效indices，使用fallback策略选择前{expected_count}个")
                top_indices = valid_indices[:expected_count]
                if not insights.strip():
                    insights = (f"Failed to generate valid comparison after {max_retries} attempts. Using fallback selection of top {expected_count} location(s).")
        analysis_dir = self.output_dir / f"{original_id}-analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = analysis_dir / f"round{round_num}_comparison.json"
        analysis_file.write_text(json.dumps({'round': round_num, 'total_locations': len(locations_data), 'expected_count': expected_count, 'selected_indices': top_indices, 'insights': insights, 'attempts': attempt + 1}, indent=2, ensure_ascii=False), encoding='utf-8')
        self.logger.info(f"Round{round_num}: 从{len(locations_data)}个location中选出{len(top_indices)}个: {top_indices}")
        return top_indices

    def _prepare_next_round_locations(self, original_id: str, current_locations: list, top_indices: list[int], round_num: int) -> list:
        """为下一轮准备location实例，并保存前一轮的patch信息"""
        next_round_locations = []
        for idx in top_indices:
            source_instance = None
            for loc in current_locations:
                match = re.search(r'-location-(\d+)', loc.problem_statement.id)
                if match and int(match.group(1)) == idx:
                    source_instance = loc
                    break
            if source_instance is None:
                self.logger.warning(f"未找到 location-{idx} 的source instance")
                continue
            new_instance = copy.deepcopy(source_instance)
            new_instance.problem_statement.id = f"{original_id}-location-{idx}-round{round_num}"
            if hasattr(source_instance, 'location_patch_data'):
                new_instance.location_patch_data = source_instance.location_patch_data
            if round_num == 1:
                source_id = f"{original_id}-location-{idx}"
            else:
                source_id = f"{original_id}-location-{idx}-round{round_num - 1}"
            source_traj_file = self.output_dir / source_id / f"{source_id}.traj"
            if source_traj_file.exists():
                try:
                    traj_data = json.loads(source_traj_file.read_text())
                    previous_patch = traj_data.get('info', {}).get('submission', '')
                    previous_trajectory = traj_data.get('trajectory', [])
                    new_instance.previous_round_info = {
                        'round': round_num - 1 if round_num > 1 else 0,
                        'patch': previous_patch,
                        'trajectory_length': len(previous_trajectory),
                        'exit_status': traj_data.get('info', {}).get('exit_status', 'unknown'),
                        'source_id': source_id
                    }
                    if round_num == 1:
                        self.logger.info(f"✓ location-{idx}: 从原始location继承patch -> round1")
                    else:
                        self.logger.info(f"✓ location-{idx}: 从 round{round_num-1} 继承patch -> round{round_num}")
                except Exception as e:
                    self.logger.error(f"读取 {source_id} 的trajectory失败: {e}")
            else:
                self.logger.warning(f"前一轮trajectory文件不存在: {source_traj_file}")
            next_round_locations.append(new_instance)
        self.logger.info(f"为 round{round_num} 准备了 {len(next_round_locations)} 个location实例")
        return next_round_locations

    def run_instance_with_location_constraint(self, instance: BatchInstance) -> None:
        """阶段2.1:使用 location 约束执行实例"""
        instance_id = instance.problem_statement.id
        should_skip, skip_reason = self.should_skip_instance(instance)
        if should_skip:
            self.logger.info(f"⏭️ Skipping {instance_id}: {skip_reason}")
            self._progress_manager.on_instance_start(instance_id)
            self._progress_manager.on_instance_end(instance_id, exit_status=skip_reason)
            return
        self.logger.info(f"执行 location 变体: {instance_id}")
        register_thread_name(instance_id)
        self._add_instance_log_file_handlers(instance_id, multi_worker=True)
        if self._progress_manager.n_completed < 5:
            time.sleep(random.random() * self._random_delay_multiplier * 4)
        self._progress_manager.on_instance_start(instance_id)
        try:
            patch_data = getattr(instance, 'location_patch_data', None)
            if patch_data is None:
                self.logger.error(f"{instance_id} 缺少 location_patch_data")
                raise ValueError(f"Missing location_patch_data for {instance_id}")
            previous_round_info = getattr(instance, 'previous_round_info', None)
            location_insights = self._generate_location_insights_from_patch(patch_data, previous_round_info)
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

    def _generate_location_insights_from_patch(self, patch_data: dict, previous_round_info: dict = None) -> str:
        """从 patch data 生成 location insights，包含前一轮的反思"""
        insights_parts = []
        if previous_round_info:
            prev_round = previous_round_info.get('round', -1)
            prev_patch = previous_round_info.get('patch', '')
            prev_status = previous_round_info.get('exit_status', 'unknown')
            insights_parts.append("**Previous Attempt Reflection:**\n")
            insights_parts.append(f"In Round {prev_round}, we attempted to fix this location with the following approach:\n")
            insights_parts.append(f"- Exit Status: {prev_status}")
            if prev_patch:
                insights_parts.append(f"\n**Previous Patch:**")
                insights_parts.append("```python")
                insights_parts.append(prev_patch)
                insights_parts.append("```\n")
                insights_parts.append("**Instructions for This Round:**")
                insights_parts.append("- Reflect on the previous patch")
                insights_parts.append("- Improve upon the previous solution")
        files_to_modify = patch_data.get('files_to_modify', [])
        insights_parts.append("**Error Code Location Information:**\n")
        insights_parts.append("I will provide you with the location information of the buggy code below. ")
        insights_parts.append("Please make your modifications at the locations specify for the erroneous code.\n\n")
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
                insights_parts.append("```\n")
        insights_parts.append("\n**Final Instructions:**")
        if previous_round_info:
            insights_parts.append("- Learn from the previous attempt and improve your solution")
        insights_parts.append("- Please make modifications at the code locations mentioned above")
        insights_parts.append("- Do NOT make changes outside these locations")
        return "\n".join(insights_parts)

    def _execute_locations_batch_parallel(self, locations: list, round_num: int) -> None:
        instances_to_run = []
        skipped_count = 0
        for loc in locations:
            should_skip, reason = self.should_skip_instance(loc)
            if should_skip:
                self.logger.info(f"⏭️ 跳过已完成: {loc.problem_statement.id} ({reason})")
                skipped_count += 1
            else:
                instances_to_run.append(loc)
        if not instances_to_run:
            self.logger.info(f"Round {round_num}: 所有location已完成，无需执行")
            return
        self.logger.info(f"Round {round_num}: 跳过 {skipped_count} 个已完成，执行 {len(instances_to_run)} 个新location")
        max_workers = min(15, len(instances_to_run))
        if max_workers <= 1:
            for loc in instances_to_run:
                self.run_instance_with_location_constraint(loc)
        else:
            self.logger.info(f"Round {round_num}: 使用 {max_workers} 个worker并行执行")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.run_instance_with_location_constraint, loc): loc for loc in instances_to_run}
                completed = 0
                total = len(futures)
                for future in as_completed(futures):
                    loc = futures[future]
                    try:
                        future.result()
                        completed += 1
                        self.logger.info(f"Round {round_num}: 完成 {completed}/{total} - {loc.problem_statement.id}")
                    except Exception as e:
                        self.logger.error(f"Round {round_num}: 执行 {loc.problem_statement.id} 失败: {e}")
            self.logger.info(f"Round {round_num}: 所有location执行完成 ({completed}/{total} 成功)")

    def main_phase_final_samples(self) -> None:
        """阶段3: 对比sample/location结果并执行sample-final"""
        original_ids_from_locations = set()
        original_ids_from_samples = set()
        location_counts = {}
        for inst in self.instances:
            if '-location-' in inst.problem_statement.id:
                parts = inst.problem_statement.id.split('-location-')
                original_id = parts[0]
                original_ids_from_locations.add(original_id)
                location_counts[original_id] = location_counts.get(original_id, 0) + 1
            elif '-sample' in inst.problem_statement.id and '-sample-final' not in inst.problem_statement.id:
                parts = inst.problem_statement.id.split('-sample')
                if len(parts) == 2:
                    original_id = parts[0]
                    original_ids_from_samples.add(original_id)
        single_location_tasks = original_ids_from_samples - original_ids_from_locations
        multi_location_tasks = original_ids_from_locations
        all_original_ids = original_ids_from_samples | original_ids_from_locations
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"阶段3：准备为 {len(all_original_ids)} 个task生成final sample")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"- 单location任务: {len(single_location_tasks)} 个")
        self.logger.info(f"- 多location任务: {len(multi_location_tasks)} 个")
        if multi_location_tasks:
            self.logger.info(f"\n为 {len(multi_location_tasks)} 个多location任务分析最终轮次结果")
            self._analyze_final_round_locations(multi_location_tasks)
        final_sample_instances = []
        for original_id in sorted(all_original_ids):
            final_sample_id = f"{original_id}-sample-final"
            is_complete, status = self._check_trajectory_complete(final_sample_id)
            if is_complete and not self._redo_existing:
                self.logger.info(f"⏭️ 跳过已完成的sample-final: {final_sample_id} (status: {status})")
                continue
            template_instance = None
            for inst in self.instances:
                if inst.problem_statement.id.startswith(original_id) and '-sample' in inst.problem_statement.id:
                    template_instance = inst
                    break
            if template_instance is None:
                self.logger.warning(f"未找到 {original_id} 的模板实例，跳过sample-final创建")
                continue
            final_sample_instance = copy.deepcopy(template_instance)
            final_sample_instance.problem_statement.id = final_sample_id
            if original_id in single_location_tasks:
                final_sample_instance._is_single_location_task = True
                self.logger.debug(f"{original_id}: 单location任务")
            else:
                final_sample_instance._is_single_location_task = False
                self.logger.debug(f"{original_id}: 多location任务")
            final_sample_instances.append(final_sample_instance)
        self.logger.info(f"\n需要执行 {len(final_sample_instances)} 个sample-final")
        if not final_sample_instances:
            self.logger.info("所有sample-final已完成，无需执行")
            return
        max_workers_for_final = min(15, len(final_sample_instances))
        if max_workers_for_final <= 1:
            for instance in final_sample_instances:
                self.run_instance_with_comparison(instance)
        else:
            add_logger_names_to_stream_handlers()
            set_stream_handler_levels(logging.WARNING)
            self.logger.setLevel(logging.TRACE)
            with Live(self._progress_manager.render_group):
                with ThreadPoolExecutor(max_workers=max_workers_for_final) as executor:
                    futures = [executor.submit(self.run_instance_with_comparison, instance) 
                            for instance in final_sample_instances]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except (KeyboardInterrupt, _BreakLoop):
                        self.logger.info("接收到中断信号")
                        executor.shutdown(wait=False, cancel_futures=True)
                    finally:
                        self._progress_manager.print_report()

    def _analyze_final_round_locations(self, task_ids: set[str]) -> None:
        """分析最终轮次的location结果，收集top-1位置的所有轮次patch"""
        for original_id in sorted(task_ids):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"分析{original_id}的最终location结果")
            self.logger.info(f"{'='*60}")
            round_pattern = re.compile(rf"{re.escape(original_id)}-location-(\d+)-round(\d+)")
            max_round = -1
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    match = round_pattern.match(item.name)
                    if match:
                        round_num = int(match.group(2))
                        if round_num > max_round:
                            max_round = round_num
            top_location_idx = None
            if max_round == -1:
                analysis_dir = self.output_dir / f"{original_id}-analysis"
                round0_comparison_file = analysis_dir / "round0_comparison.json"
                if round0_comparison_file.exists():
                    try:
                        with open(round0_comparison_file, 'r', encoding='utf-8') as f:
                            round0_data = json.load(f)
                        selected_indices = round0_data.get('selected_indices', [])
                        if selected_indices:
                            top_location_idx = selected_indices[0]
                            self.logger.info(f"从round0选出的top-1 location: {top_location_idx}")
                    except Exception as e:
                        self.logger.error(f"读取round0_comparison.json失败: {e}")
                max_round = 0
            else:
                analysis_dir = self.output_dir / f"{original_id}-analysis"
                max_round_comparison_file = analysis_dir / f"round{max_round}_comparison.json"
                if max_round_comparison_file.exists():
                    try:
                        with open(max_round_comparison_file, 'r', encoding='utf-8') as f:
                            round_data = json.load(f)
                        selected_indices = round_data.get('selected_indices', [])
                        if selected_indices:
                            top_location_idx = selected_indices[0]
                            self.logger.info(f"从round{max_round}选出的top-1 location: {top_location_idx}")
                    except Exception as e:
                        self.logger.error(f"读取round{max_round}_comparison.json失败: {e}")
            if top_location_idx is None:
                self.logger.warning(f"{original_id} 无法确定top-1 location，跳过")
                continue
            all_patches = []
            location_history = []
            location_id_r0 = f"{original_id}-location-{top_location_idx}"
            traj_file_r0 = self.output_dir / location_id_r0 / f"{location_id_r0}.traj"
            if traj_file_r0.exists():
                try:
                    traj_data = json.loads(traj_file_r0.read_text())
                    patch = traj_data.get('info', {}).get('submission', '')
                    if patch:
                        all_patches.append({
                            'round': 0,
                            'patch': patch,
                            'exit_status': traj_data.get('info', {}).get('exit_status', 'unknown'),
                            'trajectory_length': len(traj_data.get('trajectory', []))
                        })
                        location_history.append(f"**Round 0:**\n```python\n{patch}\n```\n")
                        self.logger.info(f"✓ 收集到 location-{top_location_idx} Round 0 的patch")
                except Exception as e:
                    self.logger.error(f"读取Round 0 trajectory失败: {e}")
            for round_num in range(1, max_round + 1):
                location_id_rn = f"{original_id}-location-{top_location_idx}-round{round_num}"
                traj_file_rn = self.output_dir / location_id_rn / f"{location_id_rn}.traj"
                if traj_file_rn.exists():
                    try:
                        traj_data = json.loads(traj_file_rn.read_text())
                        patch = traj_data.get('info', {}).get('submission', '')
                        if patch:
                            all_patches.append({
                                'round': round_num,
                                'patch': patch,
                                'exit_status': traj_data.get('info', {}).get('exit_status', 'unknown'),
                                'trajectory_length': len(traj_data.get('trajectory', []))
                            })
                            location_history.append(f"**Round {round_num}:**\n```python\n{patch}\n```\n")
                            self.logger.info(f"✓ 收集到 location-{top_location_idx} Round {round_num} 的patch")
                    except Exception as e:
                        self.logger.error(f"读取Round {round_num} trajectory失败: {e}")
            if not all_patches:
                self.logger.warning(f"{original_id} 的top-1 location没有有效patch，跳过")
                continue
            self.logger.info(f"共收集到 {len(all_patches)} 个历史patches")
            problem_statement = ""
            for inst in self.instances:
                if inst.problem_statement.id.startswith(original_id) and '-sample' in inst.problem_statement.id:
                    problem_statement = str(inst.problem_statement.text)
                    break
            insights = self._generate_multi_round_insights(problem_statement, top_location_idx, all_patches, location_history)
            analysis_dir = self.output_dir / f"{original_id}-sample-final"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            insights_file = analysis_dir / f"top1_location_{top_location_idx}_all_rounds_insights.txt"
            insights_file.write_text(insights, encoding='utf-8')
            metadata = {
                'task_id': original_id,
                'top_location_idx': top_location_idx,
                'max_round': max_round,
                'num_patches_collected': len(all_patches),
                'patches_summary': [
                    {
                        'round': p['round'],
                        'exit_status': p['exit_status'],
                        'trajectory_length': p['trajectory_length']
                    }
                    for p in all_patches
                ]
            }
            metadata_file = analysis_dir / f"top1_location_{top_location_idx}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
            self.logger.info(f"✓ {original_id} 的top-1 location综合分析已保存")
            self.logger.info(f"  - Location: {top_location_idx}")
            self.logger.info(f"  - Rounds: 0-{max_round}")
            self.logger.info(f"  - Patches: {len(all_patches)}")


    def _generate_multi_round_insights(self, problem_statement: str, location_idx: int, all_patches: list[dict], location_history: list[str]) -> str:
        """基于top-1位置的所有轮次patches生成综合insights"""
        insights_parts = []
        insights_parts.append(f"# Solution Evolution Analysis for Location {location_idx}")
        insights_parts.append(f"This location has been iteratively refined across {len(all_patches)} round(s).")
        insights_parts.append("## Problem Statement")
        insights_parts.append(f"{problem_statement}")
        insights_parts.append("## Evolution History")
        insights_parts.append("The following shows how the solution evolved across multiple rounds:")
        insights_parts.extend(location_history)
        insights_parts.append("\n## Key Insights for Final Solution")
        insights_parts.append("Based on the iterative refinement process:")
        insights_parts.append("**Evolutionary Pattern**: Review how the solution evolved across rounds to understand:")
        insights_parts.append("   - What approaches were tried")
        insights_parts.append("   - How the solution was refined")
        insights_parts.append("**Final Attempt Strategy**:")
        insights_parts.append("   - Build upon the most recent patch as a foundation")
        insights_parts.append("   - Incorporate successful elements from earlier attempts")
        insights_parts.append("   - Address any remaining issues or edge cases")
        insights_parts.append("   - Ensure the fix is complete and robust")
        insights_parts.append("**Instructions for Final Solution:**")
        insights_parts.append("- Learn from the evolution shown above")
        insights_parts.append("- Create a refined, complete solution that addresses all issues")
        return "\n".join(insights_parts)

    def run_instance_with_comparison(self, instance: BatchInstance) -> None:
        """执行sample-final，基于之前的sample或location对比结果"""
        final_sample_id = instance.problem_statement.id
        original_id = final_sample_id.replace('-sample-final', '')
        self.logger.info("sample-final debug: instance=%s, thread=%s", final_sample_id, threading.current_thread().name)
        should_skip, skip_reason = self.should_skip_instance(instance)
        if should_skip:
            self.logger.info(f"⏭️ Skipping {final_sample_id}: {skip_reason}")
            self._progress_manager.on_instance_start(final_sample_id)
            self._progress_manager.on_instance_end(final_sample_id, exit_status=skip_reason)
            return
        is_single_location = getattr(instance, '_is_single_location_task', False)
        if is_single_location:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"执行 {original_id} 的 sample-final（单location任务）")
            self.logger.info(f"{'='*60}")
        else:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"执行 {original_id} 的 sample-final（多location任务）")
            self.logger.info(f"{'='*60}")
        register_thread_name(final_sample_id)
        self._add_instance_log_file_handlers(final_sample_id, multi_worker=True)
        if self._progress_manager.n_completed < 5:
            time.sleep(random.random() * self._random_delay_multiplier * 4)
        self._progress_manager.on_instance_start(final_sample_id)
        try:
            insights = ""
            num_attempts = 0
            if is_single_location:
                sample_paths = []
                for sample_idx in range(1, self.num_samples+1):
                    sample_id = f"{original_id}-sample{sample_idx}"
                    sample_path = self.output_dir / sample_id
                    if sample_path.exists():
                        sample_paths.append(sample_path)
                if sample_paths:
                    num_attempts = len(sample_paths)
                    self.logger.info(f"找到 {len(sample_paths)} 个sample结果，开始对比分析")
                    comparator = self._get_comparator()
                    problem_statement = (getattr(instance.problem_statement, 'text', '') or getattr(instance.problem_statement, 'description', '') or str(instance.problem_statement))
                    comparison_result = comparator.compare_samples(sample_paths, problem_statement, len(sample_paths))
                    insights = comparison_result.get('insights', '')
                    self.logger.info(f"sample对比完成，insights长度: {len(insights)}")
                else:
                    self.logger.warning(f"未找到 {original_id} 的任何sample结果")
            else:
                analysis_dir = self.output_dir / final_sample_id
                top1_insights_files = list(analysis_dir.glob("top1_location_*_all_rounds_insights.txt"))
                if top1_insights_files:
                    insights_file = top1_insights_files[0]
                    insights = insights_file.read_text(encoding='utf-8')
                    metadata_file = insights_file.parent / insights_file.name.replace('_insights.txt', '_metadata.json')
                    if metadata_file.exists():
                        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                        num_attempts = metadata.get('num_patches_collected', 0)
                        top_location = metadata.get('top_location_idx', 'unknown')
                        max_round = metadata.get('max_round', 0)
                        self.logger.info(f"使用top-1 location {top_location} 的综合insights, ({num_attempts} patches from rounds 0-{max_round})")
                    else:
                        num_attempts = 1
                    self.logger.info(f"成功加载top-1位置综合insights，长度: {len(insights)}")
                else:
                    self.logger.warning(f"{original_id} 未找到top-1 location的综合insights")
            insights_file = self.output_dir / final_sample_id / "comparison_insights.txt"
            insights_file.parent.mkdir(parents=True, exist_ok=True)
            insights_file.write_text(insights, encoding='utf-8')
            self.logger.info(f"最终对比insights已保存: {insights_file}")
            if num_attempts == 0:
                num_attempts = 5
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
                    self.logger.info("接收到中断信号")
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
        """标准实例执行"""
        return self._run_single_instance_with_config(instance, self.agent_config)
    
    def _run_single_instance_with_config(self, instance: BatchInstance, agent_config: AgentConfig) -> AgentRunResult:
        """使用指定配置执行单个实例"""
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
        """使用insights增强agent配置"""
        enhanced_config = original_config.model_copy(deep=True)
        # enhanced_config.use_change_history_method = False
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
        """使用location insights增强agent配置"""
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