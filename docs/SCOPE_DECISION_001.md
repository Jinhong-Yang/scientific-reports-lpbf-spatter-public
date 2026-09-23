# Scope decision 001 — independent label review excluded

Decision time: `2026-09-14T20:12:42+09:00`  
Decision source: user instruction in the active Codex task  
Applies before: N2 protocol lock and any N2 test-outcome access

## Decision

The independent-rater process is excluded. The W04 64-item blinded bundle is
retained but will not be distributed or represented as completed human label
validation.

## Mandatory consequences

1. Existing boxes are used only as the operational
   `inherited_bbox_region` target. Their physical object unit remains
   unverified.
2. No rater agreement, human error rate, corrected-label validity, or
   human-ground-truth claim may be reported.
3. The corrected-versus-inherited label-policy experiment and its 20 detector
   trajectories are removed from the approved matrix.
4. The study may estimate utility for reproducing the inherited detection
   target, but may not generalize that target to individual particles, clusters,
   streaks, or spark regions without separate evidence.
5. Route B is used: internal protocol-frozen replication and controlled
   sensitivity. No external-validation claim is permitted.
6. W04 is recorded as `EXCLUDED_BY_USER_SCOPE`, not `PASS`; the unused bundle
   remains available if the scope is later amended before test evaluation.

This is a pre-outcome scope decision, not a favorable-result exclusion. Any
later attempt to restore human-corrected labels, the label-policy interaction,
or external validation requires a dated protocol amendment and cannot change
the already frozen primary results.

