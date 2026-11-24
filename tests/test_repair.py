import copy
import json
from pathlib import Path

import pytest

from phoenix.pipeline import load_location, run_task, task_jobs
from phoenix.settings import load_instances, write_json


class Backend:
    def __init__(self):
        self.calls = []

    def generate(self, task, stage, **kwargs):
        self.calls.append(kwargs)
        stage.mkdir(parents=True, exist_ok=True)
        patch = stage / "final.patch"
        patch.write_text("diff --git a/app.py b/app.py\n+value = 2\n")
        return dict(status="completed", patch=str(patch), steps=3, model_calls=3, exit_status="submitted")


def test_multicandidate_shared_budget_and_fresh_final(task, candidate, settings):
    second = copy.deepcopy(candidate)
    second["id"] = "C2"
    backend = Backend()
    observed = []

    def summary(task, rows, cfg, stage, key, **kw):
        assert len(rows) == 2 and kw["shared_budget"]
        observed.append(cfg["model"])
        return "Both strategies suggest changing the constant."

    result = run_task(task, settings, backend, "synthetic-key", [candidate, second], summary_fn=summary)
    assert result["status"] == "completed"
    assert [x["allowance"] for x in backend.calls] == [50, 50, 193]
    assert backend.calls[-1]["candidate"] is None
    assert backend.calls[-1]["summary"]
    assert result["model_calls"] == 10  # Three authors x 3, plus one summary.
    assert observed == [settings["model"]]


def test_budget_rejects_impossible_candidate_count():
    with pytest.raises(ValueError):
        task_jobs(max_calls=40)
    with pytest.raises(ValueError):
        task_jobs(candidates=101)


def test_manifest_never_passes_reference_fields(tmp_path, task):
    file = tmp_path / "tasks.json"
    write_json(file, [{**task, "patch": "REFERENCE", "test_patch": "HIDDEN", "resolved": True}])
    assert load_instances(file) == [task]
    write_json(file, [task, task])
    with pytest.raises(ValueError, match="Duplicate"):
        load_instances(file)
    write_json(file, [{**task, "repo_name": "../other"}])
    with pytest.raises(ValueError):
        load_instances(file)


def test_frozen_location_identity_and_complete_coverage(tmp_path, task, candidate):
    model = "glm-5.2"
    folder = tmp_path / model / task["instance_id"]
    write_json(tmp_path / "instances.json", [task])
    write_json(folder / "report.json", {"status": "localized", "model": model})
    artifact = {"candidates": [candidate]}
    write_json(folder / "location_candidates.json", artifact)
    assert load_location(tmp_path, task, model) == artifact
    with pytest.raises(ValueError, match="identity"):
        load_location(tmp_path, {**task, "problem_statement": "Different request"}, model)
    candidate["requirement_coverage"] = []
    write_json(folder / "location_candidates.json", artifact)
    with pytest.raises(ValueError, match="Incomplete"):
        load_location(tmp_path, task, model)
    write_json(folder / "report.json", {"status": "localized", "model": "qwen3.7-plus"})
    with pytest.raises(ValueError, match="own localization"):
        load_location(tmp_path, task, model)


@pytest.mark.parametrize("model,api_model,temp,top_p,context,output,thinking", [
    ("deepseek-v3.1", "deepseek-chat", 0.6, 0.95, 131072, None, False),
    ("deepseek-v3.2", "deepseek-chat", 1.0, 0.95, 131072, None, False),
    ("qwen3-coder-plus", "qwen3-coder-plus", 0.7, 0.80, 1000000, 65536, False),
    ("glm-4.7", "glm-4.7", 0.7, 1.00, 202752, 131072, True),
    ("minimax-m2.5", "MiniMax-M2.5", 1.0, 0.95, 204800, 32768, True),
    ("glm-5.2", "glm-5.2", 1.0, 1.0, 1048576, 131072, False),
    ("qwen3.7-plus", "qwen3.7-plus", 1.0, 0.95, 1000000, 131072, False),
    ("qwen3.7-plus-thinking", "qwen3.7-plus", 1.0, 0.95, 1000000, 131072, True),
    ("qwen-3.8-max", "qwen-3.8-max", 1.0, 0.95, 1000000, 131072, False),
])
@pytest.mark.parametrize("candidate_count", [1, 2])
def test_complete_localization_to_final_patch(monkeypatch, tmp_path, task, candidate, settings, model, candidate_count,
                                             api_model, temp, top_p, context, output, thinking):
    """Exercise real localization tools and handoff; only API/Docker are faked."""
    from phoenix import pipeline, localization, summary
    from phoenix.settings import ROOT, load_settings, model_config
    settings = load_settings("PhoenixRepair", ROOT / "configs/models" / model / "PhoenixRepair.yaml", {
        "output": settings["output"], "runtime": settings["runtime"], "concurrency": 1,
    })
    assert settings["max_input_tokens"] == context
    gen = model_config(model, settings["api_base"], shared_budget_mode=True, context_settings=settings)
    assert gen["name"] == "openai/" + api_model
    assert (gen["temperature"], gen["top_p"], gen["max_input_tokens"]) == (temp, top_p, context)
    assert gen.get("preserve_thinking", False) == thinking

    def check_request(payload):
        assert (payload["model"], payload["temperature"], payload["top_p"]) == (api_model, temp, top_p)
        if output is None:
            assert "max_tokens" not in payload
        else:
            assert payload["max_tokens"] == output
        if model in {"glm-5.2", "qwen3.7-plus", "qwen3.7-plus-thinking", "qwen-3.8-max"}:
            assert payload["enable_thinking"] is thinking
        elif model == "glm-4.7":
            assert payload["thinking"] == {"type": "enabled"}
        else:
            assert "enable_thinking" not in payload

    check_request(dict(model=api_model, temperature=gen["temperature"], top_p=gen["top_p"],
                       **gen["completion_kwargs"], **gen["completion_kwargs"]["extra_body"]))
    source = tmp_path / "source"
    (source / "repo").mkdir(parents=True)
    (source / "repo/app.py").write_text("value = 1\n")
    actions = [("location_read", {"request": dict(operation="read", path="app.py")})]
    hypotheses = [dict(id=f"H{i}", idea=f"Complete implementation approach {i}.", grounding=["app.py defines the only value."])
                  for i in range(1, candidate_count + 1)]
    actions.append(("location_checkpoint", {"request": dict(notes="Read app.py; implement the required value.",
        requirements=candidate["requirements"], hypotheses=hypotheses, active_hypothesis="H1",
        single_hypothesis_reason="Only one state-free constant exists in this fixture.")}))
    for i in range(1, candidate_count + 1):
        if i > 1:
            actions.append(("location_checkpoint", {"request": dict(notes="Explore the next independent approach.",
                                                                     active_hypothesis=f"H{i}")}))
        plan = copy.deepcopy(candidate)
        plan.update(id=f"C{i}", hypothesis_id=f"H{i}")
        actions.append(("location_candidate", {"request": dict(candidate=plan)}))
    actions.append(("analysis_complete", {}))
    pending = iter(actions)
    events = []
    def api(session, endpoint, key, payload, directory, **kwargs):
        assert payload["messages"][0]["content"].startswith("You are PhoenixRepair.")
        check_request(payload)
        assert kwargs["timeout"] == 600
        name, arguments = next(pending)
        events.append(name)
        return dict(choices=[dict(message=dict(role="assistant", content="", tool_calls=[dict(
            id=f"call-{len(events)}", type="function", function=dict(name=name, arguments=json.dumps(arguments)))]))])
    monkeypatch.setattr(localization, "call_api", api)
    monkeypatch.setattr(localization, "export_source", lambda *a, **kw: source)
    class FakeSession:
        def post(self, url, **kwargs):
            check_request(kwargs["json"])
            assert kwargs["json"]["messages"][0]["content"].startswith("You are PhoenixRepair.")
            events.append("summary")
            class Response:
                ok = True
                def json(self):
                    return dict(choices=[dict(message=dict(content="Implement the required constant and verify callers."))])
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(summary.requests, "Session", FakeSession)
    backend = Backend()
    original = backend.generate
    def generate(*args, **kwargs):
        assert "analysis_complete" in events
        events.append("generate")
        return original(*args, **kwargs)
    backend.generate = generate
    monkeypatch.setattr(pipeline, "DockerBackend", lambda *args: backend)
    reports = pipeline.run([task], settings, "synthetic-key")
    assert len(reports) == 1 and reports[0]["status"] == "completed", reports
    assert reports[0]["method"] == "PhoenixRepair"
    assert reports[0]["output_stage"] == "PhoenixRepair" and reports[0]["review_calls"] == reports[0]["repair_calls"] == 0
    assert events[:len(actions)] == [a[0] for a in actions]
    assert events[len(actions):] == (["generate"] if candidate_count == 1 else ["generate", "generate", "summary", "generate"])
    output = Path(settings["output"]) / model / task["instance_id"]
    assert (output / "PhoenixRepair.patch").read_bytes() == (output / "final.patch").read_bytes()
    assert (output / "location-1/final.patch").is_file()
    if candidate_count == 1:
        assert not (output / "final").exists()
    else:
        assert (output / "location-2/final.patch").is_file()
        assert (output / "final/final.patch").read_bytes() == (output / "final.patch").read_bytes()
        comparison = output / "final/comparison"
        trace = json.loads((output / "final/comparison.traj").read_text())
        request = json.loads((comparison / "request.json").read_text())
        assert trace["history"][:-1] == request["messages"]
        assert trace["history"][-1] == {"role": "assistant", "content": (comparison / "summary.txt").read_text()}
    assert not (output / "candidate-1").exists()



def test_single_trajectory_runs_without_location_or_summary(monkeypatch, task, settings):
    from phoenix import pipeline
    from phoenix.settings import SINGLE_METHOD
    settings.update(method=SINGLE_METHOD, max_steps=10)
    backend = Backend()
    monkeypatch.setattr(pipeline, "DockerBackend", lambda *args: backend)
    report, = pipeline.run([task], settings, "fixture")
    assert report["status"] == "completed"
    assert backend.calls == [dict(allowance=10, final_stage=True, candidate=None, summary=None, num_candidates=0)]
    assert [stage["patch"] for stage in report["stages"]] == [
        str(Path(report["patch"]).parent / "sample-1/final.patch")]
    assert Path(report["patch"]).is_file()
    assert not (Path(settings["output"]) / "localization").exists()


def test_publish_author_trajectory_preserves_last_author_on_submission_replay(tmp_path):
    from phoenix.backend import publish_author_trajectory
    stage = tmp_path / "location-1"
    attempt = stage / "attempts/001"
    official = attempt / "generate/author/author.traj"
    checkpoint = attempt / "checkpoint.json"
    write_json(checkpoint, {"history": [{"content": "checkpoint"}]})
    publish_author_trajectory(stage, attempt, {"trajectory": checkpoint}, 1)
    assert (stage / "author.traj").read_bytes() == checkpoint.read_bytes()
    write_json(official, {"history": [{"content": "completed"}]})
    publish_author_trajectory(stage, attempt, {"trajectory": checkpoint}, 0)
    publish_author_trajectory(stage, stage / "attempts/002", {"trajectory": None}, 0)
    assert (stage / "author.traj").read_bytes() == official.read_bytes()


def test_resume_uses_bound_stage_paths(task, candidate, settings):
    from phoenix.run_state import bind
    directory = Path(settings["output"]) / settings["model"] / task["instance_id"]
    second = {**candidate, "id": "C2"}
    candidates = [candidate, second]
    jobs = task_jobs(2)
    for job, stage in zip(jobs, ["candidate-1", "candidate-2", "summary", "final"]):
        job["stage"] = stage
    bind(directory / "task-identity.json", dict(task=task, jobs=jobs, candidates=candidates,
                                               model=settings["model"], method=settings["method"]))
    def compare(task, results, settings, stage, key, *, shared_budget):
        assert stage == directory / "summary"
        return "Use the second strategy."
    result = run_task(task, settings, Backend(), "fixture", candidates, summary_fn=compare)
    assert result["status"] == "completed"
    assert json.loads((directory / "plan.json").read_text()) == jobs
    assert not (directory / "location-1").exists()
