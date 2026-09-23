#!/usr/bin/env python3
"""Build the executable W03 data-quality companion notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "W03_data_quality_audit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        md(
            """
# W03 — LPBF historical-corpus data-quality audit

This notebook is the runnable companion to `data_integrity_report.json`. It
reviews derived audit outputs only; it does not modify the H0 historical corpus.
"""
        ),
        md(
            """
## tl;dr

- Validate manifest grain, split isolation, files, hashes, and bounding boxes.
- Show the scale of each split and the observed duplicate-screening signals.
- Keep label semantics and manufacturing-build independence explicitly
  unresolved until human and provenance gates are completed.
"""
        ),
        md(
            """
## Context & Methods

The audit runs at one manifest row per `sample_id`. Specimen and specimen–view
groups are the available independence units. Every declared image and label hash
was recomputed. Exact dHash equality is used only as a visual-similarity screen;
it is not interpreted as proof of duplicate acquisition. Static corpus
timeliness is not a quality dimension here; provenance and test-access history
are reported instead.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
REPORT_PATH = ROOT / "data" / "manifests" / "data_integrity_report.json"
IMAGE_MANIFEST_PATH = ROOT / "data" / "manifests" / "image_manifest.csv"
GROUP_MANIFEST_PATH = ROOT / "data" / "manifests" / "group_manifest.csv"

report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
images = pd.read_csv(IMAGE_MANIFEST_PATH)
groups = pd.read_csv(GROUP_MANIFEST_PATH)
report["status"], images.shape, groups.shape
"""
        ),
        md("## Data"),
        code(
            """
split_order = ["train", "validation", "test"]
split_table = (
    images.groupby("split")
    .agg(images=("sample_id", "size"), specimens=("specimen", "nunique"), groups=("group_id", "nunique"))
    .reindex(split_order)
)
display(split_table)

lineage_table = pd.DataFrame({
    "measure": ["rows", "specimens", "specimen-view groups", "views", "strata"],
    "value": [
        report["profile"]["rows"], report["profile"]["specimens"],
        report["profile"]["groups_specimen_view"], report["profile"]["views"],
        report["profile"]["strata"],
    ],
})
display(lineage_table)
"""
        ),
        md("## Results"),
        code(
            """
check_table = pd.DataFrame(
    [{"check": name, "passed": bool(value)} for name, value in report["checks"].items()]
)
display(check_table)
assert report["critical_failures"] == [], report["critical_failures"]
assert check_table["passed"].all(), check_table.loc[~check_table["passed"]]
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
split_table["images"].plot(kind="bar", ax=axes[0], color=["#3264a8", "#53a567", "#d38b2f"])
axes[0].set_title("Images by frozen historical split")
axes[0].set_xlabel("")
axes[0].set_ylabel("Image rows")
axes[0].tick_params(axis="x", rotation=0)

dup = report["duplicates"]
dup_plot = pd.Series({
    "image SHA rows": dup["duplicate_image_sha256_rows"],
    "label SHA rows": dup["duplicate_label_sha256_rows"],
    "exact dHash rows": dup["exact_dhash64_rows"],
})
dup_plot.plot(kind="bar", ax=axes[1], color="#8b6bb3")
axes[1].set_title("Duplicate-screening signals")
axes[1].set_xlabel("")
axes[1].set_ylabel("Rows in flagged groups")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """
integrity = report["integrity"]
integrity_table = pd.DataFrame(
    [{"measure": key, "count": value} for key, value in integrity.items() if isinstance(value, int)]
)
display(integrity_table)

limitations = pd.DataFrame({"documented limitation": report["limitations"]})
display(limitations)
"""
        ),
        md(
            """
## Takeaways

The automated checks establish file custody, manifest integrity, and
specimen-grouped split isolation. They do not validate the physical meaning or
visible correctness of inherited boxes, establish manufacturing-build
independence, restore missing source archives, or make the accessed historical
test split external. Those are explicit downstream gates.

Sources reviewed in this notebook: `data_integrity_report.json`,
`image_manifest.csv`, and `group_manifest.csv` generated from the copied
`data/study_v1/manifest.csv`.
"""
        ),
    ]
    nbf.write(notebook, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
