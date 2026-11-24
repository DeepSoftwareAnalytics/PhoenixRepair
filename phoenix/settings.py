"""Shared, explicit profiles for PhoenixRepair."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
FIELDS = ("instance_id", "problem_statement", "base_commit", "repo_name", "image_name")
METHOD_NAME = "PhoenixRepair"
LEGACY_METHOD = "our_method"
REPAIR_METHODS = (METHOD_NAME, LEGACY_METHOD)

CONTEXT_FIELDS = ("max_input_tokens", "context_compaction_threshold", "context_compaction_target")


def profile(api_model, temperature, top_p, context, output, thinking, *, extra=None,
            timeout=240, tool_choice="auto", task_timeout=7200, release=None):
    return dict(api_model=api_model, temperature=temperature, top_p=top_p,
                max_input_tokens=context, max_tokens=output, thinking=thinking,
                extra_body=extra if extra is not None else {"enable_thinking": thinking},
                timeout=timeout, tool_choice=tool_choice, task_timeout=task_timeout, release=release)


PROFILES = {
    "deepseek-v3.1": profile("deepseek-chat", 0.6, 0.95, 131072, None, False, extra={}, release="2025-08-21"),
    "deepseek-v3.2": profile("deepseek-chat", 1.0, 0.95, 131072, None, False, extra={}, release="2025-12-01"),
    "qwen3-coder-plus": profile("qwen3-coder-plus", 0.7, 0.8, 1000000, 65536, False, extra={}),
    "glm-4.7": profile("glm-4.7", 0.7, 1.0, 202752, 131072, True, extra={"thinking": {"type": "enabled"}}),
    "minimax-m2.5": profile("MiniMax-M2.5", 1.0, 0.95, 204800, 32768, True, extra={}),
    "glm-5.2": profile("glm-5.2", 1.0, 1.0, 1048576, 131072, False, tool_choice="required"),
    "qwen3.7-plus": profile("qwen3.7-plus", 1.0, 0.95, 1000000, 131072, False, timeout=600),
    "qwen3.7-plus-thinking": profile("qwen3.7-plus", 1.0, 0.95, 1000000, 131072, True, timeout=600),
    "qwen-3.8-max": profile("qwen-3.8-max", 1.0, 0.95, 1000000, 131072, False, timeout=600),
    # Additional supported models outside the reported experiments.
    "kimi-k2.7-code": profile("kimi-k2.7-code", 0.0, 1.0, 131072, 32768, True, task_timeout=3000,
                              extra={"enable_thinking": True, "preserve_thinking": True}),
    "deepseek-v4.1-flash": profile("deepseek-v4.1-flash", 0.0, 1.0, 131072, 32768, False, tool_choice="required"),
}


def inference_fields(model):
    spec = PROFILES[model]
    fields = {k: spec[k] for k in ("temperature", "top_p")}
    if spec["max_tokens"] is not None:
        fields["max_tokens"] = spec["max_tokens"]
    return dict(model=spec["api_model"], **fields, **spec["extra_body"])


def context_defaults(limit):
    """Scale the existing compression watermarks with the configured context window."""
    return dict(max_input_tokens=limit, context_compaction_threshold=limit * 120000 // 131072,
                context_compaction_target=limit * 110000 // 131072)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def component(value, name):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty single path component")
    return value


def load_instances(path: Path, selected=(), limit=0):
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("The manifest must contain a nonempty list of tasks")
    clean, seen = [], set()
    for row in rows:
        task = {key: row[key] for key in FIELDS}
        if any(not isinstance(v, str) or not v.strip() for v in task.values()):
            raise ValueError("All five task fields must be nonempty strings")
        component(task["instance_id"], "instance_id")
        component(task["repo_name"], "repo_name")
        if task["instance_id"] in seen:
            raise ValueError("Duplicate instance_id: " + task["instance_id"])
        seen.add(task["instance_id"])
        clean.append(task)
    if set(selected) - seen:
        raise ValueError("Requested task IDs are absent from the manifest")
    clean = [x for x in clean if not selected or x["instance_id"] in selected]
    return clean[:limit] if limit else clean


def model_config(model, api_base, *, shared_budget_mode=False, context_settings=None):
    profile = PROFILES[model]
    data = {
        "name": "openai/" + profile["api_model"], "api_base": api_base, "api_key": "$PHOENIX_API_KEY",
        "temperature": profile["temperature"], "top_p": profile["top_p"], "per_instance_cost_limit": 0,
        "total_cost_limit": 0, "per_instance_call_limit": 0, **context_defaults(profile["max_input_tokens"]),
        "context_keep_first_rounds": 5, "context_keep_last_rounds": 5,
        "retry": {"retries": 8 if shared_budget_mode else 20, "min_wait": 10, "max_wait": 120},
        "completion_kwargs": {
            "extra_body": copy.deepcopy(profile["extra_body"]),
            "timeout": profile["timeout"],
            "tool_choice": profile["tool_choice"], "num_retries": 0,
        },
    }
    if profile["max_tokens"] is not None:
        data["completion_kwargs"]["max_tokens"] = profile["max_tokens"]
    if profile["thinking"]:
        data["preserve_thinking"] = True
    if shared_budget_mode:
        data["request_min_interval"] = (context_settings or {}).get("api_request_interval", 2)
    data.update({k: context_settings[k] for k in CONTEXT_FIELDS
                 if context_settings and k in context_settings})
    if context_settings and context_settings.get("no_truncation"):
        data.update(reject_truncation=True, context_compaction_threshold=0)
    return data


def load_settings(mode, config_path, overrides):
    if not config_path.exists() and config_path.name == LEGACY_METHOD + ".yaml":
        replacement = config_path.resolve().with_name(METHOD_NAME + ".yaml")
        if replacement.is_relative_to(ROOT / "configs") and replacement.is_file():
            config_path = replacement
    data = yaml.safe_load(config_path.read_text()) or {}
    allowed = {"model", "instances", "output", "runtime", "api_base", "concurrency", "max_steps",
               "task_timeout", "locations", "stage", "docker_host", "container_cpus", "container_memory",
               "localization_max_steps", "localization_timeout", "adaptive_io", "io_device",
               "environment_start_concurrency", "local_command_concurrency", "pin_cpus",
               "submission_recovery_timeout", "no_truncation", "exclude_rate_limit_from_budget", "api_request_interval", *CONTEXT_FIELDS}
    if set(data) - allowed:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(set(data) - allowed)))
    data.update({k: v for k, v in overrides.items() if v is not None})
    data.setdefault("model", "glm-5.2")
    if data["model"] not in PROFILES:
        raise ValueError("Unknown model profile: " + data["model"])
    for key, env in [("instances", "INSTANCES"), ("runtime", "PHOENIX_RUNTIME"), ("docker_host", "DOCKER_HOST"), ("api_base", "API_BASE")]:
        if not data.get(key) and os.environ.get(env):
            data[key] = os.environ[env]
    data.setdefault("api_base", API_BASE)
    data["api_base"] = data["api_base"] or API_BASE
    data.setdefault("output", "runs/" + mode)
    data.setdefault("concurrency", 4)
    data.setdefault("max_steps", 200)
    data.setdefault("localization_max_steps", 150)
    data.setdefault("stage", "all")
    data.setdefault("container_cpus", 2)
    data.setdefault("container_memory", "6g")
    data.setdefault("adaptive_io", False)
    data.setdefault("io_device", None)
    data.setdefault("pin_cpus", False)
    if "api_request_interval" in data and (type(data["api_request_interval"]) not in (int, float) or data["api_request_interval"] < 0):
        raise ValueError("api_request_interval must be nonnegative")
    data.setdefault("exclude_rate_limit_from_budget", False)
    if type(data["exclude_rate_limit_from_budget"]) is not bool:
        raise ValueError("exclude_rate_limit_from_budget must be a boolean")
    data.setdefault("no_truncation", False)
    if type(data["no_truncation"]) is not bool:
        raise ValueError("no_truncation must be a boolean")
    if data["no_truncation"]:
        if data.get("context_compaction_threshold", 0) != 0:
            raise ValueError("no_truncation requires context_compaction_threshold=0")
        data["context_compaction_threshold"] = 0
    data.setdefault("environment_start_concurrency", 2)
    data.setdefault("local_command_concurrency", 4)
    data.setdefault("submission_recovery_timeout", 600)
    data["task_timeout"] = data.get("task_timeout") or PROFILES[data["model"]]["task_timeout"]
    limit = data.get("max_input_tokens", PROFILES[data["model"]]["max_input_tokens"])
    if type(limit) is not int or limit < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    for k, v in context_defaults(limit).items():
        data.setdefault(k, v)
    context = {k: data[k] for k in CONTEXT_FIELDS}
    for key, value in context.items():
        minimum = 0 if key == "context_compaction_threshold" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(key + " must be an integer >= " + str(minimum))
    if context["context_compaction_threshold"]:
        if not (context["context_compaction_target"] <= context["context_compaction_threshold"]
                < context["max_input_tokens"]):
            raise ValueError("Require compaction target <= threshold < max_input_tokens")
    if not 1 <= data["max_steps"] <= 200:
        raise ValueError("max_steps must be between 1 and 200")
    if not 1 <= data["localization_max_steps"] <= 200:
        raise ValueError("localization_max_steps must be between 1 and 200")
    if data.get("localization_timeout") is not None and data["localization_timeout"] <= 0:
        raise ValueError("localization_timeout must be positive when set")
    if not isinstance(data["concurrency"], int) or data["concurrency"] < 1:
        raise ValueError("concurrency must be positive")
    if data["task_timeout"] <= 0:
        raise ValueError("task_timeout must be positive")
    if data["adaptive_io"] and not data["io_device"]:
        raise ValueError("adaptive_io requires io_device for the experiment disk")
    for key in ('adaptive_io', 'pin_cpus'):
        if type(data[key]) is not bool:
            raise ValueError(key + ' must be a boolean')
    if data['adaptive_io']:
        component(data['io_device'], 'io_device')
        if not (Path('/sys/class/block') / data['io_device']).exists():
            raise ValueError('io_device must name an existing Linux block device')
    for key in ("environment_start_concurrency", "local_command_concurrency", "submission_recovery_timeout"):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if data["stage"] not in ("all", "localization", "patch"):
        raise ValueError("stage must be all, localization, or patch")
    if mode not in REPAIR_METHODS and (data.get("locations") or data["stage"] != "all"):
        raise ValueError("locations/stage only apply to PhoenixRepair")
    for key in ("instances", "output", "runtime", "locations"):
        if data.get(key):
            data[key] = str(Path(data[key]).expanduser().resolve())
    data["method"] = mode
    return data


def localization_config(settings):
    """Match the frozen complete-strategy locations used by the scored PhoenixRepair run.

    Localization predates the model-specific patch profiles. Applying the patch
    profile here changes the experiment, especially GLM's output allowance.
    """
    data = copy.deepcopy(yaml.safe_load((ROOT / "config/locate.yaml").read_text()))
    if settings.get("method") == LEGACY_METHOD:
        templates = data["agent"]["templates"]
        templates["system_template"] = templates["system_template"].removeprefix("You are PhoenixRepair.\n")
    data["agent"]["max_steps"] = settings["localization_max_steps"]
    model = settings["model"]
    limit = settings.get("max_input_tokens", PROFILES[model]["max_input_tokens"])
    data["localization_runtime"] = {
        "inference": inference_fields(model),
        "api_timeout": 240 if model == "kimi-k2.7-code" else 600,
        "tool_choice": "auto", "task_timeout": settings.get("localization_timeout"),
        "action_repeat_limit": 10, "max_compacted_tokens": limit * 110000 // 131072,
        "context_compaction_threshold": limit * 100000 // 131072,
        "context_compaction_target": limit * 90000 // 131072,
    }
    if settings.get("no_truncation"):
        data["localization_runtime"].update(no_truncation=True, context_compaction_threshold=0,
                                            max_compacted_tokens=limit)
    return data
