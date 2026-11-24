"""Same-model comparison of PhoenixRepair candidate patches."""
import hashlib
from pathlib import Path
import os

import requests

from phoenix.prompts import PromptBuilder
from phoenix.repair_guidance import SUMMARY_GUIDANCE
from phoenix.settings import LEGACY_METHOD, PROFILES, inference_fields, write_json
from phoenix.run_state import bind, read, digest

def summarize(task, candidates, settings, directory, key):
    directory.mkdir(parents=True, exist_ok=True)
    inputs = []
    for index, row in enumerate(candidates, 1):
        patch = Path(row["patch"])
        preview = patch.read_text()
        if not preview.strip():
            raise ValueError("Summary requires every candidate patch to be nonempty")
        metadata = dict(original_chars=len(preview), prompt_chars=len(preview), compacted=False)
        inputs.append(dict(index=index, trajectory=[None] * row["steps"], submission=preview,
                           exit_status="submitted", patch_sha256=digest(patch), preview=metadata))
    prompt = PromptBuilder()._build_comparison_prompt(task["problem_statement"], inputs, len(inputs))
    if settings["method"] == LEGACY_METHOD:
        prompt = prompt.removeprefix("You are PhoenixRepair.\n")
    prompt += SUMMARY_GUIDANCE
    (directory / "prompt.txt").write_text(prompt)
    write_json(directory / "inputs.json", [{k: v for k, v in row.items() if k != "submission"} for row in inputs])
    payload = dict(inference_fields(settings["model"]), messages=[{"role": "user", "content": prompt}])
    timeout = (30, PROFILES[settings["model"]]["timeout"])
    bind(directory / "identity.json", {"payload": payload, "api_base": settings["api_base"],
                                      "shared_budget": True})
    write_json(directory / "request.json", payload)
    state = read(directory / "state.json") or {
        "attempts": 0, "format_attempt": 0, "transport_attempt": 0, "phase": "ready",
        "usage": dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        "usage_applied": [], "unknown_requests": []}
    receipt = read(directory / "result.json")
    trajectory_path = directory.parent / "comparison.traj" if directory.name == "comparison" else None

    def publish_trajectory(response):
        message = {"role": "assistant", **response["choices"][0]["message"]}
        write_json(trajectory_path, {"model": payload["model"], "history": [*payload["messages"], message],
                                     "usage": response.get("usage", {})})

    if receipt.get("status") == "completed":
        if digest(directory / "summary.txt") != receipt["summary_sha256"]:
            raise ValueError("Completed summary changed; refusing to regenerate or continue")
        if trajectory_path is not None and not trajectory_path.exists():
            publish_trajectory(read(directory / f"response-{receipt['api_attempts']}.json"))
        return (directory / "summary.txt").read_text()
    if state["phase"] == "failed":
        raise RuntimeError("Summary exhausted its original bounded attempts; no budget reset on resume")
    exclude_rate_limits = settings.get("exclude_rate_limit_from_budget", False)
    state.setdefault("rate_limit_retries", 0)
    session = requests.Session()
    session.trust_env = False
    def persist():
        write_json(directory / "state.json", state)
        write_json(directory / "usage.json", {"api_attempts": state["attempts"], **state["usage"],
                                            "rate_limit_retries": state["rate_limit_retries"],
                                            "budget_calls": state["attempts"] - state["rate_limit_retries"],
                                            "unknown_requests": state["unknown_requests"]})

    def transport_failure():
        state["phase"] = "failed"
        persist()

    try:
        while state["format_attempt"] < 1 and state["phase"] != "failed":
            response_path = directory / f"response-{state['attempts']}.json"
            if state["phase"] == "inflight":
                # A response persisted before a crash is consumed exactly once.
                # An unknown provider outcome remains charged, never refunded.
                if response_path.exists():
                    state["phase"] = "response"
                else:
                    state["unknown_requests"].append(state["attempts"])
                    transport_failure()
                    continue
            if state["phase"] == "ready":
                from sweagent.agent.request_pacing import wait_for_request
                account = settings.get("account_id") or os.environ.get("PHOENIX_ACCOUNT_ID") or "env-" + hashlib.sha256(key.encode()).hexdigest()[:16]
                wait_for_request(account, "openai/" + settings["model"], settings.get("api_request_interval", 2))
                state["attempts"] += 1
                state["transport_attempt"] += 1
                state["phase"] = "inflight"
                persist()  # Charge before transmission, including DNS/timeouts.
                response_path = directory / f"response-{state['attempts']}.json"
                try:
                    result = session.post(settings["api_base"].rstrip("/") + "/chat/completions",
                        headers={"Authorization": "Bearer " + key}, json=payload, timeout=timeout)
                    if exclude_rate_limits and result.status_code == 429:
                        from sweagent.agent.request_pacing import cooldown
                        state["rate_limit_retries"] += 1
                        state["transport_attempt"] -= 1
                        state["phase"] = "ready" if state["rate_limit_retries"] < 8 else "failed"
                        write_json(directory / f"rate-limit-{state['attempts']}.json",
                                   {"status": 429, "budget_charged": False})
                        persist()
                        cooldown(account, settings["model"], seconds=90)
                        continue
                    if not result.ok:
                        raise RuntimeError(f"Summary API HTTP {result.status_code}")
                    response = result.json()
                    if "data" in response and "response" in response["data"]:
                        response = response["data"]["response"]
                    write_json(response_path, response)
                    state["phase"] = "response"
                    persist()
                except (requests.RequestException, ValueError, RuntimeError, KeyError, IndexError, TypeError, AttributeError) as exc:
                    write_json(directory / f"transport-error-{state['attempts']}.json",
                               {"error_type": type(exc).__name__, "attempt": state["attempts"]})
                    transport_failure()
                    continue
            try:
                response = read(response_path)
                if state["attempts"] not in state["usage_applied"]:
                    for k in state["usage"]:
                        state["usage"][k] += response.get("usage", {}).get(k, 0) or 0
                    state["usage_applied"].append(state["attempts"])
                    persist()
                choice = response["choices"][0]
                answer = choice["message"].get("content") or ""
                if not isinstance(answer, str):
                    raise ValueError("Summary content must be text")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                transport_failure()
                continue
            if answer.strip():
                if trajectory_path is not None:
                    publish_trajectory(response)
                (directory / "summary.txt").write_text(answer)
                write_json(directory / "result.json", {"status": "completed",
                    "summary_sha256": digest(directory / "summary.txt"), "model": settings["model"],
                    "api_attempts": state["attempts"], "budget_reset": False})
                state["phase"] = "completed"
                persist()
                return answer
            (directory / f"invalid-summary-{state['attempts']}.txt").write_text(answer)
            state["format_attempt"] += 1
            state["transport_attempt"] = 0
            state["phase"] = "ready"
            persist()
        state["phase"] = "failed"
        raise RuntimeError("Summary request/format failed after bounded attempts; final generation is blocked")
    finally:
        persist()
        session.close()
