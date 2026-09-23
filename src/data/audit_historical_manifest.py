#!/usr/bin/env python3
"""Audit the recovered LPBF study-v1 manifest without modifying source data.

This is an N2 audit utility applied to H0 historical inputs.  It verifies the
manifest at its declared image-level grain and writes only derived inventory
and quality records into the new Scientific Reports recovery workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "specimen",
    "view",
    "frame_stem",
    "temporal_sample_index",
    "image_path",
    "label_path",
    "property_path",
    "image_sha256",
    "label_sha256",
    "material",
    "laser_power_w",
    "scan_speed_mm_s",
    "stratum",
    "bbox_x_px",
    "bbox_y_px",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px2",
    "image_width_px",
    "image_height_px",
}

ALLOWED_SPLITS = {"train", "validation", "test"}
ALLOWED_VIEWS = {"Off_Axis_Images", "On_Axis_Images"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-root",
        type=Path,
        required=True,
        help="Recovered metal_spatter_pinn snapshot root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New workspace directory for derived manifests and report.",
    )
    parser.add_argument(
        "--skip-content-hashes",
        action="store_true",
        help="Skip full file SHA-256 verification (not recommended for W03).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def normalize_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def main() -> int:
    args = parse_args()
    historical_root = args.historical_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = historical_root / "data" / "study_v1" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Historical manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path, dtype={"frame_stem": "string"})
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # Preserve source order and add explicit lineage fields for the new study.
    derived = frame.copy()
    derived.insert(0, "evidence_layer", "H0_HISTORICAL")
    derived["group_id"] = derived["specimen"].astype(str) + "|" + derived["view"].astype(str)
    derived["archive_group"] = derived["specimen"].astype(str)
    derived["manufacturing_build"] = pd.NA
    derived["manufacturing_build_evidence"] = "UNCONFIRMED_FROM_AVAILABLE_METADATA"
    derived["label_source"] = "original"
    derived["label_unit"] = "unresolved_original_bbox"
    derived["label_semantics_status"] = "REQUIRES_W04_HUMAN_RUBRIC"

    null_counts = {column: int(frame[column].isna().sum()) for column in frame.columns}
    blank_counts = {
        column: int(frame[column].map(is_blank).sum())
        for column in frame.select_dtypes(include=["object", "string"]).columns
    }
    exact_duplicate_rows = int(frame.duplicated().sum())
    duplicate_sample_ids = int(frame.duplicated(subset=["sample_id"], keep=False).sum())
    duplicate_grain_rows = int(
        frame.duplicated(subset=["specimen", "view", "frame_stem"], keep=False).sum()
    )

    split_counts = frame.groupby("split", dropna=False).size().astype(int).to_dict()
    specimen_split_counts = frame.groupby("specimen")["split"].nunique()
    specimen_split_overlap = specimen_split_counts[specimen_split_counts > 1].index.astype(str).tolist()
    group_split_counts = derived.groupby("group_id")["split"].nunique()
    group_split_overlap = group_split_counts[group_split_counts > 1].index.astype(str).tolist()

    invalid_split_rows = int((~frame["split"].isin(ALLOWED_SPLITS)).sum())
    invalid_view_rows = int((~frame["view"].isin(ALLOWED_VIEWS)).sum())
    numeric_columns = [
        "bbox_x_px",
        "bbox_y_px",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_px2",
        "image_width_px",
        "image_height_px",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid_numeric_rows = int(numeric.isna().any(axis=1).sum())
    invalid_bbox_mask = (
        (numeric["bbox_x_px"] < 0)
        | (numeric["bbox_y_px"] < 0)
        | (numeric["bbox_width_px"] <= 0)
        | (numeric["bbox_height_px"] <= 0)
        | (numeric["bbox_x_px"] + numeric["bbox_width_px"] > numeric["image_width_px"])
        | (numeric["bbox_y_px"] + numeric["bbox_height_px"] > numeric["image_height_px"])
        | (
            (numeric["bbox_width_px"] * numeric["bbox_height_px"] - numeric["bbox_area_px2"]).abs()
            > 1e-9
        )
    )
    invalid_bbox_rows = int(invalid_bbox_mask.sum())

    file_findings: list[dict[str, Any]] = []
    image_dimensions = Counter()
    image_hash_mismatches = 0
    label_hash_mismatches = 0
    missing_images = 0
    missing_labels = 0
    missing_properties = 0
    unreadable_images = 0
    perceptual_hashes: list[int | None] = []

    for index, row in frame.iterrows():
        image_path = normalize_path(historical_root, str(row["image_path"]))
        label_path = normalize_path(historical_root, str(row["label_path"]))
        property_path = normalize_path(historical_root, str(row["property_path"]))

        if not image_path.is_file():
            missing_images += 1
            perceptual_hashes.append(None)
            if len(file_findings) < 50:
                file_findings.append({"row": int(index), "kind": "missing_image", "sample_id": row["sample_id"]})
        else:
            try:
                with Image.open(image_path) as image:
                    image_dimensions[(int(image.width), int(image.height), str(image.mode))] += 1
                perceptual_hashes.append(dhash64(image_path))
            except Exception as exc:  # pragma: no cover - evidence path
                unreadable_images += 1
                perceptual_hashes.append(None)
                if len(file_findings) < 50:
                    file_findings.append(
                        {"row": int(index), "kind": "unreadable_image", "sample_id": row["sample_id"], "error": str(exc)}
                    )
            if not args.skip_content_hashes and sha256_file(image_path).lower() != str(row["image_sha256"]).lower():
                image_hash_mismatches += 1
                if len(file_findings) < 50:
                    file_findings.append(
                        {"row": int(index), "kind": "image_hash_mismatch", "sample_id": row["sample_id"]}
                    )

        if not label_path.is_file():
            missing_labels += 1
            if len(file_findings) < 50:
                file_findings.append({"row": int(index), "kind": "missing_label", "sample_id": row["sample_id"]})
        elif not args.skip_content_hashes and sha256_file(label_path).lower() != str(row["label_sha256"]).lower():
            label_hash_mismatches += 1
            if len(file_findings) < 50:
                file_findings.append(
                    {"row": int(index), "kind": "label_hash_mismatch", "sample_id": row["sample_id"]}
                )

        if not property_path.is_file():
            missing_properties += 1
            if len(file_findings) < 50:
                file_findings.append({"row": int(index), "kind": "missing_property", "sample_id": row["sample_id"]})

    derived["dhash64"] = [None if value is None else f"{value:016x}" for value in perceptual_hashes]

    exact_phash_groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(perceptual_hashes):
        if value is not None:
            exact_phash_groups[f"{value:016x}"].append(index)
    exact_phash_duplicate_groups = [rows for rows in exact_phash_groups.values() if len(rows) > 1]
    exact_phash_duplicate_rows = sum(len(rows) for rows in exact_phash_duplicate_groups)
    exact_phash_cross_split_groups = 0
    exact_phash_cross_specimen_groups = 0
    for rows in exact_phash_duplicate_groups:
        subset = frame.iloc[rows]
        exact_phash_cross_split_groups += int(subset["split"].nunique() > 1)
        exact_phash_cross_specimen_groups += int(subset["specimen"].nunique() > 1)

    image_hash_counts = frame["image_sha256"].value_counts()
    label_hash_counts = frame["label_sha256"].value_counts()
    duplicate_image_hash_groups = int((image_hash_counts > 1).sum())
    duplicate_image_hash_rows = int(image_hash_counts[image_hash_counts > 1].sum())
    duplicate_label_hash_groups = int((label_hash_counts > 1).sum())
    duplicate_label_hash_rows = int(label_hash_counts[label_hash_counts > 1].sum())

    group_rows = []
    for group_id, group in derived.groupby("group_id", sort=True):
        indices = pd.to_numeric(group["temporal_sample_index"], errors="coerce")
        group_rows.append(
            {
                "group_id": group_id,
                "specimen": str(group["specimen"].iloc[0]),
                "view": str(group["view"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "row_count": int(len(group)),
                "unique_frame_count": int(group["frame_stem"].nunique()),
                "min_temporal_sample_index": None if indices.isna().all() else int(indices.min()),
                "max_temporal_sample_index": None if indices.isna().all() else int(indices.max()),
                "stratum": str(group["stratum"].iloc[0]),
                "archive_group": str(group["specimen"].iloc[0]),
                "manufacturing_build": "",
                "manufacturing_build_evidence": "UNCONFIRMED_FROM_AVAILABLE_METADATA",
                "independence_level": "specimen_view_group_only",
            }
        )

    # Profile actual source label keys without inferring object semantics.
    label_key_counts = Counter()
    label_parse_failures = 0
    for label_value in frame["label_path"]:
        label_path = normalize_path(historical_root, str(label_value))
        if not label_path.is_file():
            continue
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            label_key_counts.update(payload.keys())
        except Exception:
            label_parse_failures += 1

    stratum_split_presence = (
        frame.groupby(["stratum", "split"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
    )
    all_strata_in_all_splits = all(all(int(counts.get(split, 0)) > 0 for split in ALLOWED_SPLITS) for counts in stratum_split_presence.values())

    checks = {
        "manifest_row_count_is_5632": len(frame) == 5632,
        "specimen_count_is_176": frame["specimen"].nunique() == 176,
        "view_count_is_2": frame["view"].nunique() == 2,
        "stratum_count_is_32": frame["stratum"].nunique() == 32,
        "all_strata_present_in_all_splits": all_strata_in_all_splits,
        "sample_id_unique": duplicate_sample_ids == 0,
        "specimen_view_frame_unique": duplicate_grain_rows == 0,
        "no_exact_duplicate_rows": exact_duplicate_rows == 0,
        "no_specimen_split_overlap": len(specimen_split_overlap) == 0,
        "no_group_split_overlap": len(group_split_overlap) == 0,
        "valid_split_values": invalid_split_rows == 0,
        "valid_view_values": invalid_view_rows == 0,
        "numeric_fields_parse": invalid_numeric_rows == 0,
        "bounding_boxes_in_bounds": invalid_bbox_rows == 0,
        "all_images_present": missing_images == 0,
        "all_labels_present": missing_labels == 0,
        "all_properties_present": missing_properties == 0,
        "all_images_readable": unreadable_images == 0,
        "image_hashes_match_manifest": args.skip_content_hashes or image_hash_mismatches == 0,
        "label_hashes_match_manifest": args.skip_content_hashes or label_hash_mismatches == 0,
        "no_exact_image_hash_duplicates": duplicate_image_hash_rows == 0,
        "all_labels_parse_as_json": label_parse_failures == 0,
    }

    critical_failures = [
        name
        for name, passed in checks.items()
        if not passed
        and name
        not in {
            "no_exact_image_hash_duplicates",
        }
    ]
    status = "PASS_WITH_LIMITATION" if not critical_failures else "FAILED"

    derived_csv_path = output_dir / "image_manifest.csv"
    derived_parquet_path = output_dir / "image_manifest.parquet"
    group_manifest_path = output_dir / "group_manifest.csv"
    label_schema_path = output_dir / "label_schema.json"
    report_path = output_dir / "data_integrity_report.json"

    derived.to_csv(derived_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    parquet_status: dict[str, Any]
    try:
        derived.to_parquet(derived_parquet_path, index=False)
        parquet_status = {
            "status": "WRITTEN",
            "path": str(derived_parquet_path),
            "sha256": sha256_file(derived_parquet_path),
        }
    except Exception as exc:
        parquet_status = {"status": "MISSING_ENGINE", "error": str(exc)}

    pd.DataFrame(group_rows).to_csv(group_manifest_path, index=False)
    label_schema = {
        "schema_version": 1,
        "source": "AI-Hub 71476 accessible validation-archive subset; local copied JSON annotations",
        "label_source": "original",
        "label_unit": "unresolved_original_bbox",
        "semantic_status": "REQUIRES_W04_HUMAN_RUBRIC",
        "coordinate_fields": {
            "x": "anotation.bbox.x",
            "y": "anotation.bbox.y",
            "width": "anotation.bbox.width",
            "height": "anotation.bbox.height",
        },
        "observed_key_presence": dict(sorted(label_key_counts.items())),
        "rows_profiled": int(len(frame) - label_parse_failures - missing_labels),
        "parse_failures": int(label_parse_failures),
        "human_validation": False,
        "warning": "No particle/cluster/streak/spark-region semantic claim is made before W04.",
    }
    label_schema_path.write_text(json.dumps(label_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "intended_use": "W03 data-lineage and split-integrity gate for a new Scientific Reports study",
        "grain": "one selected image annotation per sample_id",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_root": str(historical_root),
        "content_hash_verification": "SKIPPED" if args.skip_content_hashes else "FULL",
        "profile": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "specimens": int(frame["specimen"].nunique()),
            "groups_specimen_view": int(derived["group_id"].nunique()),
            "views": int(frame["view"].nunique()),
            "strata": int(frame["stratum"].nunique()),
            "split_counts": split_counts,
            "null_counts": null_counts,
            "blank_counts": blank_counts,
            "image_dimensions_modes": {
                f"{width}x{height}|{mode}": int(count)
                for (width, height, mode), count in sorted(image_dimensions.items())
            },
        },
        "checks": checks,
        "critical_failures": critical_failures,
        "duplicates": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "duplicate_sample_id_rows": duplicate_sample_ids,
            "duplicate_specimen_view_frame_rows": duplicate_grain_rows,
            "duplicate_image_sha256_groups": duplicate_image_hash_groups,
            "duplicate_image_sha256_rows": duplicate_image_hash_rows,
            "duplicate_label_sha256_groups": duplicate_label_hash_groups,
            "duplicate_label_sha256_rows": duplicate_label_hash_rows,
            "exact_dhash64_groups": int(len(exact_phash_duplicate_groups)),
            "exact_dhash64_rows": int(exact_phash_duplicate_rows),
            "exact_dhash64_cross_split_groups": int(exact_phash_cross_split_groups),
            "exact_dhash64_cross_specimen_groups": int(exact_phash_cross_specimen_groups),
            "interpretation": "dHash equality is a screening signal, not proof of byte-identical or semantically duplicated frames.",
        },
        "integrity": {
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "missing_properties": missing_properties,
            "unreadable_images": unreadable_images,
            "image_hash_mismatches": image_hash_mismatches,
            "label_hash_mismatches": label_hash_mismatches,
            "label_json_parse_failures": label_parse_failures,
            "invalid_bbox_rows": invalid_bbox_rows,
            "specimen_split_overlap": specimen_split_overlap,
            "group_split_overlap": group_split_overlap,
            "bounded_file_findings": file_findings,
        },
        "limitations": [
            "The 5,632-image corpus is a selected accessible subset of complete validation archives, not the full AI-Hub dataset.",
            "TS.z05, TS.z06, and TS.z07 remain unavailable according to the recovered study record.",
            "Specimen names and archive metadata do not establish independent manufacturing-build identity.",
            "dHash screens visual similarity but does not replace source-level frame lineage or human review.",
            "Original label object semantics and visible correctness remain unresolved until W04 human rubric/audit.",
            "The historical test split has already been accessed and is not a pristine external validation cohort.",
        ],
        "outputs": {
            "image_manifest_csv": {
                "path": str(derived_csv_path),
                "sha256": sha256_file(derived_csv_path),
            },
            "image_manifest_parquet": parquet_status,
            "group_manifest_csv": {
                "path": str(group_manifest_path),
                "sha256": sha256_file(group_manifest_path),
            },
            "label_schema_json": {
                "path": str(label_schema_path),
                "sha256": sha256_file(label_schema_path),
            },
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path), "critical_failures": critical_failures}, indent=2))
    return 0 if status != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
