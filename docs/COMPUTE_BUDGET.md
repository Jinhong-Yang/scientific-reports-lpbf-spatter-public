# W06 measured compute budget and projection

Status: `W08_APPROVED_PRE_TEST`. The bounded measurements and projections below
supported the frozen scope-adjusted 244-trajectory matrix, bfloat16 precision,
fixed 8,960-update detector checkpoint, compact-diffusion budget, and local
130 GPU-hour ceiling. This approval does not convert projections into measured
full-run costs.

## Measured evidence

The primary Faster R-CNN mechanics were measured on the local NVIDIA GeForce
RTX 5080 (16,303 MiB reported device memory; driver 591.86) at native 300×300
input, micro/effective batch 4, SGD, and bfloat16 autocast. The run used 64
training and 64 validation images selected as one per stratum–view bucket. It
read no test image and retained no scientific checkpoint.

| Quantity | Actual measured value |
|---|---:|
| Optimizer updates | 200 |
| Wall time including five 64-image validation passes | 34.810 s |
| Mean step time after five warm-up updates | 0.13697 s |
| Median step time, all updates | 0.14282 s |
| 95th-percentile step time, all updates | 0.15422 s |
| Peak CUDA allocated memory | 1,392.46 MiB |
| Peak CUDA reserved memory | 2,268.00 MiB |
| First-20 / last-20 mean training loss | 0.42282 / 0.05030 |

The initially specified fp16 AMP path stopped at update 1 because the loss was
finite (1.13638) but the unscaled gradient norm was infinite at initial scaler
scale 65,536. That stopped attempt is preserved in
`evidence/pilot/failure_amp_fp16_update1.json`. The otherwise identical bf16
diagnostic completed all 200 updates with finite loss and gradients. This is a
precision-path finding, not a favorable-outcome selection.

The validation COCO AP sequence (updates 0/50/100/150/200) was
0.01465/0.30156/0.39246/0.38043/0.38657. Because W04 label semantics are not
resolved and this is one short reused-subset trajectory, those values are only
a mechanics/stability signal. They cannot select a scientific checkpoint or
support a performance claim.

## Explicit extrapolations

The following values are projections, not measured full runs. They use the
post-warm-up bfloat16 step mean, assume 8,960 updates, and linearly scale the
mean 64-image validation duration to four 1,024-image validation passes.

| Projection | Estimated value |
|---|---:|
| 8,960 training updates | 1,227.22 s |
| Four 1,024-image validation passes | 88.16 s |
| One primary trajectory | 1,315.38 s = 0.3654 GPU-h |
| Proposed 80-trajectory primary full-data block | 29.23 GPU-h |
| Same block with 15% scheduling/IO contingency | 33.62 GPU-h |

For scale only, applying the same profile to all 264 detector trajectories in
the work-order example gives 96.46 GPU-h, or 110.93 GPU-h with 15% contingency.
That number is not an approved study budget: RetinaNet, low-data refits,
grouped validation, label interaction, generator fits, HPO, high resolutions,
external evaluation, checkpoint IO, and failures have different or unmeasured
costs. It must not be quoted as the total experiment cost.

## Compact diffusion technical pilot

The proposed same-corpus `ND_compact_conditional_ddpm_v0` was separately
measured at 64×64, batch 16, bfloat16 autocast, 3,626,049 parameters, cosine
1,000-step diffusion schedule, and epsilon prediction. It used 64 training and
64 validation rows with no external pretraining and no test access.

| Quantity | Actual measured value |
|---|---:|
| Optimizer updates | 200 |
| Loop time including four milestone validation passes | 4.149 s |
| Mean step time after five warm-up updates | 0.01824 s |
| 95th-percentile step time | 0.02487 s |
| Peak CUDA allocated / reserved memory | 503.12 / 608.00 MiB |
| First-20 / last-20 mean epsilon MSE | 0.28563 / 0.01412 |

At the measured step rate, 10,000 updates would be about 182.36 seconds and
100,000 updates about 0.5065 GPU-hour before validation, sampling, checkpoints,
or quality metrics. These are scale scenarios only; no full-fit update count or
sampling budget is selected. A separate 10-step random-weight DDIM fixture
validated deterministic finite reverse-process mechanics, but not trained-model
quality or the proposed 50-step cost. FID/KID, memorization, label validation,
detector utility, and 128/256/300-resolution cost remain unmeasured.

## Approved budget gate

No paid resource is authorized. W08 approved the local RTX 5080, bfloat16,
fixed detector budget, 100,000-update compact-diffusion full fits, 50-step DDIM,
the scope-adjusted matrix, and a 130 GPU-hour cumulative stop. The historical
0.36-hour value was not used as the new
runtime; the numerically similar single-trajectory projection above comes from
the new 200-update measurement and its stated scaling assumptions.

Runtime enforcement sums the measured W09 and W10 generator campaigns, W11
diffusion fits, W12 detector training and validation, W15 resolution
evaluations, W14 generator and detector work, and W16 generator and detector
work. Before each new W12, W14, or W16 training trajectory, the runner reserves
a 0.75 GPU-hour safety margin and writes a `PAUSED_COMPUTE_CEILING_GUARD`
receipt instead of starting work that could cross the approved ceiling.
The W17 runner applies the same margin before unlocking the held-out payload
and again before each fixed-checkpoint evaluation; its elapsed GPU time is
measured between explicit CUDA synchronization points and added to the final
project total.
