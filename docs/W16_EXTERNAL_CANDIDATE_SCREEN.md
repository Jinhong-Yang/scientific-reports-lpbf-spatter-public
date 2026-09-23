# W16 external-cohort candidate screen

Status: `NO_VERIFIED_SAME_TASK_EXTERNAL_COHORT`  
Checked: 2026-09-14 KST  
Raw candidate data downloaded: No

## Intended use and acceptance rule

The target is a genuinely independent cohort on which the frozen N2 detector
can be evaluated for the same foreground unit as H0. A candidate must establish
the acquisition/build unit, imaging modality and view, field of view and scale,
machine-readable labels, target semantics, and the absence of development use.
The phrase “LPBF spatter” or the presence of images alone is insufficient.

## Finding

None of seven checked official candidates currently qualifies as
`SAME_TASK_EXTERNAL`. This is a high-severity limitation for an external
validity claim, but it does not invalidate a correctly scoped grouped-internal
study. The detailed row-level classification is in
`docs/DATASET_COMPATIBILITY.csv`.

Two candidates are useful new-task replications:

- The NIST AM-Bench thermography data contain 1,800 FPS thermal videos from
  eight parts and motivate spatter segmentation, but the challenge description
  states that frame anomaly labels are absent. New target labels would be
  required.
- The 2026 NIST landed-spatter release supplies microscope images, CNN labels,
  segmentations, and particle measurements. It is strong independent morphology
  evidence, but its ex-situ landed-particle endpoint is not an in-flight optical
  bounding-box endpoint.

Two candidates may support representation or mechanism analyses only:

- The Heriot-Watt archive provides CC BY high-speed videos and a 4.2 GB raw
  image archive corresponding to annotated frames in the paper, but the portal
  does not establish machine-readable boxes matching H0.
- RAISE-LPBF provides on-axis 20 kFPS 316L video and an official loader, but its
  benchmark endpoint is reconstruction of laser parameters rather than
  spatter-box detection.

Three layer-imaging datasets are incompatible with the current primary task:
ORNL Peregrine 2021, ORNL Peregrine 2025, and Aalto's manually annotated
powder-bed/optical-tomography defects. Their pixel/XML annotations concern
layer anomalies or defects, not an established in-flight foreground box.

## Decision impact

Do not download terabyte- or gigabyte-scale candidates merely to satisfy the
word “external.” After W04 fixes the label unit, the Heriot-Watt archive is the
best first file-level compatibility audit because its modality is closest, but
it remains representation-only until its raw schema and annotation mapping are
verified. NIST landed particles can be a separately declared morphology task.

If the user cannot provide a genuinely independent same-task cohort, W08 should
explicitly approve route B: grouped internal evaluation with the manuscript
scope `internal protocol-frozen replication and controlled sensitivity`.
Nothing in this screen converts the already accessed historical test split into
external confirmation.

## Official sources checked

- Heriot-Watt University dataset portal: https://researchportal.hw.ac.uk/en/datasets/supporting-data-for-the-interplay-between-vapour-liquid-and-solid/
- RAISE-LPBF official repository: https://github.com/Flanders-Make-vzw/RAISE_LPBF_Laser_benchmark
- NIST/INFORMS AM-Bench thermography description: https://informs-qsr.wordpress.ncsu.edu/contributing/qsr-data-challenge/
- ORNL Peregrine 2021: https://doi.ccs.ornl.gov/dataset/e2decf63-021c-563c-8729-ffe02769176c
- ORNL Peregrine 2025: https://doi.ccs.ornl.gov/dataset/96b7da99-07e1-562d-88d5-d7e079acfef7
- Aalto University dataset record: https://research.aalto.fi/en/datasets/annotated-image-dataset-for-defects-detection-in-laser-powder-bed/
- NIST landed-spatter dataset: https://data.nist.gov/od/id/mds2-4277
