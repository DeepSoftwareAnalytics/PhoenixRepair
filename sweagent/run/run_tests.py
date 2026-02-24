import os
import re
import copy
import json
import docker
import platform
import resource
import traceback
from tqdm import tqdm
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from swebench.harness.constants import LATEST
from swebench.harness.docker_build import build_env_images
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.run_evaluation import get_dataset_from_preds, run_instance
from swebench.harness.test_spec.test_spec import (TestSpec, make_env_script_list, make_repo_script_list)
from swebench.harness.constants import (FAIL_TO_PASS, KEY_INSTANCE_ID, MAP_REPO_VERSION_TO_SPECS, PASS_TO_PASS, USE_X86, SWEbenchInstance)

OPEN_FILE_LIMIT = 4096

NOOP_PATCH = """diff --git a/this_is_invisible.py b/this_is_invisible.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/this_is_invisible.py
@@ -0,0 +1 @@
+# This is a commented out line
"""

NOOP_PATCH_2 = """diff --git a/this_is_invisible_2.py b/this_is_invisible_2.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/this_is_invisible_2.py
@@ -0,0 +1 @@
+# This is a commented out line
"""

def remove_ansi_sequences(input_string):
    ansi_escape_pattern = r"\x1b\[\d+m"
    clean_string = re.sub(ansi_escape_pattern, "", input_string)
    return clean_string

def txt_file_contains_string(path_to_txt, expected_output, other_patterns=[]):
    """
    Check if the given text file contains the specified string.
    :param path_to_txt: Path to the text file.
    :param expected_output: The string to search for in the text file.
    :return: True if the string is found in the text file, otherwise False.
    """
    try:
        with open(path_to_txt, "r", encoding="utf-8") as file:
            content = file.read()
            filtered_content = remove_ansi_sequences(content)
            for pattern in other_patterns:
                if pattern in filtered_content:
                    return False
            return expected_output in filtered_content
    except FileNotFoundError:
        pass
    except IOError:
        print(f"An error occurred while reading the file at {path_to_txt}.")
    return False

def create_instance_test_dict(jsonl_file_path):
    instance_test_dict = {}
    with open(jsonl_file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            instance_id = json_obj["instance_id"]
            test_patch = json_obj["test_patch"]
            instance_test_dict[instance_id] = test_patch
    return instance_test_dict


def extract_resolved_info(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {}
    result = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            report_path = os.path.join(subdir_path, "report.json")
            if os.path.isfile(report_path):
                try:
                    with open(report_path, "r", encoding="utf-8") as report_file:
                        data = json.load(report_file)
                        resolved_value = data.get(subdir, {}).get("resolved", False)
                        result[subdir] = resolved_value
                except (json.JSONDecodeError, KeyError):
                    result[subdir] = False
    return result

def make_regression_spec(instance: SWEbenchInstance) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance["version"]
    base_commit = instance["base_commit"]
    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]
    pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
    fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name)
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_regression_script_list(instance, specs, env_name, repo_directory, base_commit)
    if platform.machine() in {"aarch64", "arm64"}:
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"
    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=specs.get("language", "python"),
        docker_specs=specs,
        namespace=specs.get("namespace", "swebench"),
    )


def make_regression_script_list(instance, specs, env_name, repo_directory, base_commit):
    HEREDOC_DELIMITER = "EOF_114329324912"
    fake_apply_test_patch_command = (f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{NOOP_PATCH_2}\n{HEREDOC_DELIMITER}")
    test_command = " ".join([MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"], *get_test_directives(instance)])
    eval_commands = ["source /opt/miniconda3/bin/activate", f"conda activate {env_name}", f"cd {repo_directory}"]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [fake_apply_test_patch_command, test_command]
    return eval_commands


def rearrange_patches(test_specs):
    """
    rearrange the patches such that slower instance_ids are evaluated first
    this way pipelining will be faster.
    """
    slow_instance_ids = ["sympy__sympy-11870"]
    slow_specs = [test_spec for test_spec in test_specs if test_spec.instance_id in slow_instance_ids]
    if len(slow_specs) != 0:
        print(f"rearrange patches such that {[x.instance_id for x in slow_specs]} are evaluated first")
        rearranged_test_specs = slow_specs
        for test_spec in test_specs:
            if test_spec.instance_id not in slow_instance_ids:
                rearranged_test_specs.append(test_spec)
        return rearranged_test_specs
    else:
        return test_specs


def run_tests(location_data:list, instance_ids: list, model_patches: list, max_workers: int, run_id: str, timeout: int, apply_model_patch=True, dataset_name="princeton-nlp/SWE-bench_Verified"):
    assert len(instance_ids) == len(model_patches), "There must be the same number of instance_ids as model patches"
    resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILE_LIMIT, OPEN_FILE_LIMIT))
    print(f"Using run_id: {run_id}")
    split = "test"
    client = docker.from_env()
    force_rebuild = False
    predictions = {}
    for idx, one_instance_id in enumerate(instance_ids):
        if not apply_model_patch:
            patch_to_apply = NOOP_PATCH
        else:
            patch_to_apply = model_patches[idx]
        predictions[one_instance_id] = {"model_name_or_path": "test", "model_patch": patch_to_apply, "instance_id": one_instance_id}
    instances = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id, False, True)
    print(f"Running {len(instances)} unevaluated instances...")
    instance_test_dict = {}
    for line in location_data:
        instance_id = line["original_id"]
        test = line["tests_passing_in_original_repo"]
        instance_test_dict[instance_id] = test
    no_f2p_instances = []
    for instance in instances:
        revised_instance = instance
        revised_instance["FAIL_TO_PASS"] = "[]"
        revised_instance["PASS_TO_PASS"] = instance_test_dict[instance["instance_id"]]
        no_f2p_instances.append(revised_instance)
    build_env_images(client, no_f2p_instances, force_rebuild=force_rebuild, max_workers=max_workers, namespace="swebench", instance_image_tag=LATEST, env_image_tag=LATEST)
    test_specs = list(map(make_regression_spec, no_f2p_instances))
    test_specs = rearrange_patches(test_specs)
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {tag for i in client.images.list(all=True) for tag in i.tags if tag in instance_image_ids}
    print(f"Found {len(existing_images)} existing instance images. Will reuse them.")
    resolved_dict = extract_resolved_info(os.path.join("logs", "run_evaluation", run_id, "test"))
    ids = [test_spec.instance_id for test_spec in test_specs if test_spec.instance_id not in list(resolved_dict.keys())]
    for index, patch in enumerate(model_patches):
        if patch == "":
            resolved_dict[instance_ids[index]] = False
    with tqdm(total=len(ids), smoothing=0, colour="MAGENTA") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_instance, test_spec, predictions[test_spec.instance_id], False, force_rebuild, client, run_id, timeout): None for test_spec in test_specs if test_spec.instance_id in ids}
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error getting future result: {e}")
                    continue
                if result and isinstance(result, tuple) and len(result) == 2:
                    instance_id = result[0]
                    resolved = result[1][instance_id]["resolved"]
                    resolved_dict[instance_id] = resolved
                elif result and isinstance(result, dict):
                    for instance_id in result:
                        resolved_dict[instance_id] = False
                else:
                    print(f"Warning: Unexpected result type: {type(result)}")
    print("All instances run.")
    return resolved_dict
