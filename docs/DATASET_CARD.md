# Dataset card — recovered LPBF study-v1 corpus

Status: W03 `PASS_WITH_LIMITATION`. Evidence layer: `H0_HISTORICAL`; the audit
outputs are derived N2 records about H0 inputs, not new experimental outcomes.

## Intended use

This corpus supports bounded reproduction of the historical optical-residual
image-generation and spatter-box detection study, followed by a newly frozen
Scientific Reports experiment protocol. It must not be used to claim thermal
field calibration, manufacturing-build independence, pristine external
validation, or human-validated synthetic labels without additional evidence.

## Source and custody

- Source record: AI-Hub 71476 metal 3D-printing spatter imagery.
- Locally recovered raw archive root:
  `D:\dataset\Metal_3D_Printing_Spart_Image` (14 archive files, approximately
  1.056 TiB), linked through `data/raw_link/AIHub71476_archives`.
- Selected working corpus: copied beneath
  `historical/metal_spatter_pinn/data/study_v1`.
- The recovered project states that official training volumes `TS.z05`,
  `TS.z06`, and `TS.z07` were unavailable; the selected corpus comes from
  complete validation archives.
- Manifest receipt:
  `2909ef4ef845431710a95e9f8c874b8bc01a9168c35286cfc188783b5850e290`.

## Grain and population

- Grain: one selected image/annotation pair per unique `sample_id`.
- Rows: 5,632.
- Specimens: 176.
- Available grouping unit: 352 specimen–view groups.
- Views: `Off_Axis_Images` and `On_Axis_Images`.
- Process-condition strata: 32.
- Frozen historical split: 3,584 train, 1,024 validation, 1,024 test images.

`specimen` is supported by file and manifest lineage. A separate
`manufacturing_build` identity is not supported by the available metadata and
is left blank with status `UNCONFIRMED_FROM_AVAILABLE_METADATA`.

## Labels

Each source JSON exposes a single bounding-box geometry through the historical
`anotation.bbox` object. The coordinate records can be checked mechanically,
but the intended object unit and visible correctness have not been validated
by independent human review. Derived records therefore use:

- `label_source = original`
- `label_unit = unresolved_original_bbox`
- `label_semantics_status = REQUIRES_W04_HUMAN_RUBRIC`

No particle, cluster, streak, spark-region, or equivalent semantic class is
inferred from geometry alone.

## Split and access policy

The historical split is specimen-grouped; W03 independently checks both
specimen and specimen–view group overlap. The 32-specimen / 1,024-image test
partition has already been used by historical evaluations. It remains useful
for disclosed reproduction, but it is neither pristine nor external.

## Quality controls

The W03 audit checks required columns, row/grain uniqueness, nulls and blanks,
split/view domains, numeric parsing, box bounds and area, file presence,
image readability, full image and label SHA-256 values, exact byte-hash
duplicates, exact 64-bit dHash groups, split leakage, and label JSON parsing.

Exact dHash equality is a screening signal only. It may reflect similar frames,
uniform regions, or transformations and is not treated as proof of byte-level
duplication. Identical label-JSON hashes can occur when multiple images share
the same geometry/metadata; they require context, not automatic deletion.

### Observed W03 result

- All 22 automated integrity checks passed and there were no critical
  failures.
- Missing images, labels, and property records: 0 each.
- Unreadable images, image/label hash mismatches, JSON parse failures, and
  invalid boxes: 0 each.
- Exact byte-identical image groups: 0.
- Specimen and specimen–view cross-split overlaps: 0.
- Thirty-six repeated label-JSON hashes cover 72 rows. Every group stays within
  one specimen and one split; this is retained as a lineage observation rather
  than a deletion rule.
- Exact 64-bit dHash equality flags 2,422 rows in 320 groups, including 195
  cross-split groups. The largest groups are sparse off-axis patterns and have
  distinct SHA-256 values. This coarse signal is not adequate to declare image
  duplication; candidates remain available for W04 visual review.

## Known limitations

- This is a selected accessible subset, not the full AI-Hub dataset.
- Missing historical archive volumes cannot be silently reconstructed.
- Manufacturing-build independence is unknown.
- Original box semantics and correctness require W04 human calibration.
- The historical test split is already accessed.
- Static-corpus timeliness is not scored; provenance and version receipts are
  the relevant freshness controls.

## Derived artifacts

- `data/manifests/image_manifest.parquet`: row-level manifest with lineage and
  audit fields.
- `data/manifests/group_manifest.csv`: specimen–view group inventory.
- `data/manifests/label_schema.json`: observed geometry schema and semantic
  limitation.
- `data/manifests/data_integrity_report.json`: machine-readable checks,
  findings, and output hashes.
- `notebooks/W03_data_quality_audit.ipynb`: executable companion analysis.
- `notebooks/W03_data_quality_audit.html`: rendered, visually checked output.
