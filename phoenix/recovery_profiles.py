"""Explicit historical model/method recovery profiles; never switch models."""

from phoenix.settings import PROFILES, REPAIR_METHODS

# Generation sampling parameters remain in settings.PROFILES. These values
# describe transport/lifecycle recovery, not another repair or review prompt.
def recovery_profile(settings):
    model = settings["model"]
    spec = PROFILES[model]
    result = dict(model=model, method=settings["method"], api_timeout_seconds=spec["timeout"],
                  wall_time_seconds=settings.get("task_timeout", spec["task_timeout"]), thinking=spec["thinking"],
                  summary_timeout_seconds=120, summary_transport_attempts=5, summary_format_rounds=3,
                  generation_supported=True, same_model_only=True, budget_reset=False, review_calls=0,
                  environment_start_retries=3 if settings["method"] in REPAIR_METHODS else 1,
                  terminal_session_retries=1, submission_replay_attempts=2)
    if spec["thinking"]:
        result["preserve_thinking"] = True
    if settings["method"] in REPAIR_METHODS:
        result.update(summary_timeout_seconds=(30, result["api_timeout_seconds"]),
                      summary_transport_attempts=1, summary_format_rounds=1,
                      summary_request_interval_seconds=2,
                      author_budget="shared task API-call budget; no calls reserved for post-PhoenixRepair review")
    else:
        result["author_budget"] = "independent trajectory step cap; preserve already spent steps on recovery"
    return result
