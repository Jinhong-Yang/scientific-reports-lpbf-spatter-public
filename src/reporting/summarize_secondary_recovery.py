"""Read completed W17 outputs once; export counts only, never change outcomes.

This local audit needs excluded prediction payloads. The public manuscript
builder consumes its small aggregate receipt and does not read those payloads.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_records(records: list[dict]) -> dict:
    annotated = [r for r in records if "prediction_sanitization" in r]
    if annotated and len(annotated) != len(records):
        raise ValueError("Partially annotated run")
    retained = sum(len(r["boxes"]) for r in records)
    discarded = sum(r["prediction_sanitization"]["invalid_detections_discarded"] for r in annotated)
    candidates = (sum(r["prediction_sanitization"]["candidate_detections"] for r in annotated)
                  if annotated else retained)
    if candidates != retained + discarded:
        raise ValueError("Candidate/retained/discarded counts do not reconcile")
    return {"images": len(records), "adapter_annotated_images": len(annotated),
            "candidate_detections": candidates, "retained_detections": retained,
            "invalid_detections_discarded": discarded,
            "images_with_discards": sum(r["prediction_sanitization"]["invalid_detections_discarded"] > 0
                                       for r in annotated)}


def main() -> None:
    receipt = json.loads((ROOT / "evidence/test_campaign/W17_TEST_COMPLETENESS.json").read_text())
    if receipt["status"] != "COMPLETE" or receipt["completed_evaluations"] != 208:
        raise RuntimeError("W17 must be terminal and complete")
    metrics_path = ROOT / "evidence/test_campaign/test_metrics.parquet"
    if digest(metrics_path) != receipt["metrics_manifest_sha256"]:
        raise RuntimeError("W17 metrics hash mismatch")
    metrics = pd.read_parquet(metrics_path)
    secondary = metrics[(metrics.source_block == "W12_core") & (metrics.family == "retinanet")]
    if len(secondary) != 20 or secondary.groupby("arm").size().to_dict() != dict.fromkeys(["N1", "NB", "NP", "NR"], 5):
        raise RuntimeError("Incomplete secondary roster")
    runs = []
    for row in secondary.sort_values(["arm", "detector_seed"]).itertuples():
        path = ROOT / "runs/new_study/W17_test_campaign/W12_core" / row.run_id / "predictions.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            counts = summarize_records(json.load(stream))
        if counts["images"] != 1024:
            raise RuntimeError("Incomplete image roster")
        runs.append({"run_id": row.run_id, "arm": row.arm, "detector_seed": int(row.detector_seed),
                     "prediction_sha256": digest(path), **counts})
    failures = sorted((ROOT / "runs/new_study/W17_test_campaign").rglob("failure_*.json"))
    value = {"schema_version": 1, "status": "PASS", "scope": "all_20_completed_secondary_retinanet_cells",
             "amendment": "004 (reuses Amendment 003 adapter)", "runs": runs,
             "failure_record_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in failures},
             "metrics_manifest_sha256": digest(metrics_path),
             "prediction_payloads_exported": False, "inference_rerun": False,
             "totals": {key: sum(r[key] for r in runs) for key in (
                 "images", "adapter_annotated_images", "candidate_detections", "retained_detections",
                 "invalid_detections_discarded", "images_with_discards")}}
    target = ROOT / "evidence/reporting/W17_SECONDARY_RECOVERY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "totals": value["totals"], "failure_records": len(failures)}))


if __name__ == "__main__":
    main()
