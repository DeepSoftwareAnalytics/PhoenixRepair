import os
import uuid
import json
import json
import time
import requests
import jsonlines
from datasets import load_dataset
from sweagent.run.run_tests import run_tests
from sweagent.run.run_tests import make_regression_spec
from swebench.harness.run_evaluation import get_dataset_from_preds
from swebench.harness.grading import get_eval_tests_report, get_logs_eval
from swebench.harness.constants import (FAIL_TO_PASS, KEY_INSTANCE_ID, PASS_TO_PASS, TestStatus)


MODEL_MAX_TOKENS = 100000

class LogExractAgent:
    """Example API caller class compatible with the code above"""
    def __init__(self, api_key, base_url, model_name, max_retries=3, retry_delay=2):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
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
                print(f"API call failed, retry {retries + 1}/{self.max_retries} (error: {str(e)})")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Unknown error in API call: {e}")
                break
        return ""

def rewrite_report(test_spec, input_folder_path, regression_tests, api_caller=None):
    """
    Use LLM to analyze test logs and count the execution results of regression tests
    Args:
        test_spec: Test specification object
        input_folder_path: Input folder path
        regression_tests: List of regression tests
        api_caller: API caller object (must have _call_api method)
    
    Returns:
        dict: Dictionary containing success, failure, and skipped fields
    """
    instance_id = test_spec.instance_id
    log_path = f"{input_folder_path}/test/{instance_id}/test_output.txt"
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return {"success": [], "failure": [], "skipped": []}
    except Exception as e:
        print(f"Error reading log file: {e}")
        return {"success": [], "failure": [], "skipped": []}
    log_lines = log_content.split('\n')
    log_part = '\n'.join(log_lines[80:])
    if not log_part.strip():
        print(f"Log file is empty or last 1/3 is empty: {log_path}")
        return {"success": [], "failure": [], "skipped": []}
    regression_tests_str = '\n'.join(regression_tests)
    prompt = f"""You are a test log analysis expert. I need you to analyze the following test logs and count the execution results of regression tests.

**Regression Test List (total {len(regression_tests)} tests):**
```
{regression_tests_str}
```

**Test Log:**
```
{log_part}
```

**Task Requirements:**
Please carefully analyze the log and determine the execution status for each test in the regression test list above:
1. **success**: Test passed/succeeded (PASSED, OK, SUCCESS, etc.)
2. **failure**: Test failed (FAILED, ERROR, ASSERTION ERROR, etc.)
3. **skipped**: Test was skipped (SKIPPED, XFAIL, etc.)

**Output Format Requirements:**
Please strictly follow the JSON format below without adding any other explanatory text:
```json
{{
    "success": ["test_name1", "test_name2"],
    "failure": ["test_name3", "test_name4"],
    "skipped": ["test_name5"]
}}
```

Notes:
- Only count tests that appear in the regression test list
- If a test is not found in the log, do not add it to any category
- Ensure test names match exactly with those in the regression test list
"""
    messages = [{"role": "system", "content": "You are a professional test log analysis assistant, skilled at extracting and classifying test results from test outputs."}, {"role": "user", "content": prompt}]
    if api_caller is None:
        print("Warning: api_caller not provided, cannot call LLM for analysis")
        return {"success": [], "failure": [], "skipped": []}
    try:
        response = api_caller._call_api(messages, temperature=0.1, top_p=0.95)
        if not response:
            print(f"LLM returned empty response: {instance_id}")
            return {"success": [], "failure": [], "skipped": []}
        response = response.strip()
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0].strip()
        elif '```' in response:
            response = response.split('```')[1].split('```')[0].strip()
        result = json.loads(response)
        if not all(key in result for key in ["success", "failure", "skipped"]):
            print(f"LLM returned incorrect format: {result}")
            return {"success": [], "failure": [], "skipped": []}
        all_tests = set(result["success"] + result["failure"] + result["skipped"])
        regression_tests_set = set(regression_tests)
        invalid_tests = all_tests - regression_tests_set
        if invalid_tests:
            print(f"Warning: LLM returned tests not in regression test list: {invalid_tests}")
        print(f"Instance {instance_id} analysis completed: "
              f"success={len(result['success'])}, "
              f"failure={len(result['failure'])}, "
              f"skipped={len(result['skipped'])}")
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON returned by LLM: {e}")
        print(f"LLM raw response: {response[:500]}...")
        return {"success": [], "failure": [], "skipped": []}
    except Exception as e:
        print(f"Error analyzing test log: {e}")
        return {"success": [], "failure": [], "skipped": []}


def save_passing_tests(output_jsonl_path, input_folder_path, dataset):
    ds = load_dataset(dataset)
    with jsonlines.open(output_jsonl_path, mode="w") as writer:
        for entry in ds["test"]:
            instance_id = entry["instance_id"]
            log_path = f"{input_folder_path}/test/{instance_id}/test_output.txt"
            try:
                eval_sm, _ = get_logs_eval(log_path)
            except FileNotFoundError:
                print(f"File not found: {log_path}")
                continue
            successful_test = []
            for test_name, status in eval_sm.items():
                if status in [TestStatus.PASSED.value]:
                    successful_test.append(test_name)
            if successful_test == []:
                print(f"{instance_id} didn't get any passing tests")
            result_entry = {"instance_id": instance_id, "tests_passing_in_original_repo": successful_test}
            writer.write(result_entry)


def run_regression_for_each_instance(location_data, run_id):
    instance_ids = [location_data["original_id"]]
    patches = [location_data["submission"]]
    # print(f"Running tests for instance: {instance_id} with patch: {patch}")
    result = run_tests(
            location_data=[location_data],
            instance_ids=instance_ids,
            model_patches=patches,
            max_workers=1,
            run_id=run_id,
            timeout=3000,
            apply_model_patch=True,
            dataset_name="princeton-nlp/SWE-bench_Verified"
        )
    return result

def check_if_all_instances_pass(instance_to_plausible):
    all_passed = True
    not_passing_instances = []
    for key, value in instance_to_plausible.items():
        if not value:
            all_passed = False
            not_passing_instances.append(key)
    if all_passed:
        print("All the chosen regression tests pass in the base repository")
    else:
        print(f"One or more of the regression tests for instances {not_passing_instances} do not pass in the original repository")
        print(len(not_passing_instances))

def _run_regression(data, llm_config):
    instance_test_dict = {}
    instance_id = data["original_id"]
    test = data["tests_passing_in_original_repo"]
    instance_test_dict[instance_id] = test
    instance_ids = [data["original_id"]]
    patches = [data["submission"]]
    predictions = {}
    for idx, one_instance_id in enumerate(instance_ids):
        predictions[one_instance_id] = {"model_name_or_path": "test", "model_patch": patches[idx], "instance_id": one_instance_id}
    run_id = f"run_{instance_id}_{uuid.uuid4().hex}"
    instances = get_dataset_from_preds("princeton-nlp/SWE-bench_Verified", "test", instance_ids, predictions, run_id, False, True)
    no_f2p_instances = []
    for instance in instances:
        revised_instance = instance.copy()
        revised_instance["FAIL_TO_PASS"] = "[]"
        revised_instance["PASS_TO_PASS"] = instance_test_dict.get(instance["instance_id"], "[]")
        no_f2p_instances.append(revised_instance)
    test_specs = list(map(make_regression_spec, no_f2p_instances))
    test_spec_dict = {spec.instance_id: spec for spec in test_specs}
    run_regression_for_each_instance(data, run_id)
    api_caller = LogExractAgent(api_key=llm_config["api_key"], base_url=llm_config["api_base"], model_name=llm_config["model_name"])
    regression_dict = {}
    instance_id = data["original_id"]
    if os.path.isfile(f"logs/run_evaluation/{run_id}/test/{instance_id}/report.json"):
        test_spec = test_spec_dict.get(instance_id)
        if test_spec:
            regression_dict[instance_id] = rewrite_report(test_spec, f"logs/run_evaluation/{run_id}", instance_test_dict[instance_id], api_caller=api_caller)
        else:
            regression_dict[instance_id] = {"success": [], "failure": [], "skipped": []}
    else:
        regression_dict[instance_id] = {"success": [], "failure": [], "skipped": []}
    print(f"regression_dict:\n{regression_dict}")
    return regression_dict
