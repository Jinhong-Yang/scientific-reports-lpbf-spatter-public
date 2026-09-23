# W15 resolution, exposure, and budget sensitivity

Status: `FROZEN BEFORE W15 OUTCOMES`

The source real images are native 300×300 pixels. Generator outputs are 64×64
and are bilinearly upsampled to the detector's 300×300 tensor shape. Therefore,
all detector inputs have the same tensor dimensions, but real and synthetic
inputs do not have the same native information resolution. Those statements are
kept separate throughout the manuscript.

After W12 completes, W15 reuses the fixed final Faster R-CNN checkpoints for
N1, NR, NB, and NP. The native validation metric is read from its immutable W12
receipt. A single additional validation-only audit downsamples each native real
image to 64×64 and then upsamples it to 300×300 before inference. The difference
between this `common64` path and `native300` is a resolution-path sensitivity,
not a new training trajectory and not a held-out result.

The 1,024-pixel branch is not run because the available source is native 300×300,
the branch would add no verified spatial information, and it is outside the
frozen detector trajectory budget. Budget reporting separately tabulates
optimizer updates, real rows, additional-real rows, synthetic rows, actual
sample exposure, measured GPU time, and input pathways. Equal total updates
must never be described as equal real-image exposure.

W15 uses validation data only and cannot replace the locked primary endpoint.
Independent human review remains excluded, so resolution findings do not
validate the physical semantics of the inherited box target.
