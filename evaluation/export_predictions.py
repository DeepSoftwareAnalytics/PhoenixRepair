"""Export final run patches for the SWE-bench Verified or Pro V2 harness."""
import argparse
import json
import tempfile
from pathlib import Path


def export(run, destination, benchmark):
    plan = json.loads((run / "plan.json").read_text())
    progress = json.loads((run / "progress.json").read_text())
    ids = plan["instance_ids"]
    reports = {row["instance_id"]: row for row in progress["tasks"]}
    if progress["status"] != "completed" or set(ids) != set(reports) or len(ids) != len(reports):
        raise ValueError("The run needs a terminal report for every selected instance")
    settings = plan["settings"]
    model = settings["model"]
    for value in [model, *ids]:
        if not value or value in (".", "..") or Path(value).name != value:
            raise ValueError(f"Invalid run path component: {value!r}")
    if settings["stage"] == "localization":
        raise ValueError("Localization-only runs have no generated predictions")
    if destination.exists():
        raise ValueError(f"Choose a new export directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        out = Path(tmp) / "export"
        out.mkdir()
        predictions = []
        for iid in ids:
            status = reports[iid]["status"]
            if status not in ("completed", "failed", "infrastructure_failed"):
                raise ValueError(f"{iid}: cannot evaluate task with status {status}")
            patch = (run / model / iid / "final.patch").read_bytes() if status == "completed" else b""
            if benchmark == "verified":
                predictions.append(dict(instance_id=iid, model_name_or_path=f"{settings['method']}-{model}",
                                        model_patch=patch.decode("utf-8")))
            else:
                if not iid.startswith("instance_"):
                    raise ValueError(f"Expected a Pro V2 instance ID: {iid}")
                trial = out / "source_job" / iid
                (trial / "agent").mkdir(parents=True)
                (trial / "result.json").write_text(json.dumps({"task_name": iid}))
                (trial / "agent/model.patch").write_bytes(patch)
        if benchmark == "verified":
            (out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        (out / "ids.txt").write_text("\n".join(ids) + "\n")
        out.rename(destination)
    return len(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, choices=("verified", "pro-v2"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = export(args.run_dir, args.output, args.benchmark)
    print(f"Exported {count} predictions to {args.output}")
