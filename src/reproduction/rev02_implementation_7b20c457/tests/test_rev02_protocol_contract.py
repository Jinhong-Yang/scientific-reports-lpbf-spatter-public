"""Data-free regression checks for the committed REV02 design.

Protocol checks read only protocol documents. Access-gate checks import the
side-effect-free common module and use temporary toy receipts, never real input
manifests, specimen identities, or test payloads. They do not authorize execution
or T7; passing them is not a substitute for full implementation review.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


REV_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REV_ROOT / "protocol"
PROTOCOL_ID = "LPBF-WEAK-PHYSICS-REV02-20260902-v1"
PROTOCOL_COMMIT = "8e3aa248dd9849149561f7c938ab455392d7b5b0"
FROZEN_HASH_RECEIPT_SHA256 = "d4cb8fa32ffdc36fbf0770558c92103e75461aad74ecbeae8df60d7d6b07014d"
TASKS = ("MASTER", "T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B", "T7", "T8")
GENERATOR_SEEDS = list(range(41001, 41011))
DETECTOR_SEEDS = [51001, 51002, 51003]
PRIMARY_ENDPOINTS = [
    "fresh_collocation_proxy_RMS", "reconstruction_PSNR", "PFFD_std",
    "within_condition_one_minus_SSIM", "boundary_energy",
    "moment_centroid_error", "moment_spread_error",
]
PRIMARY_DETECTOR_CONTRASTS = [
    "D2_minus_D0", "D4_minus_D0", "D2_minus_D5_at_D2_ratio",
    "D4_minus_D5_at_D4_ratio",
]


def protocol(task: str) -> dict:
    if task not in TASKS:
        raise ValueError("Only explicitly enumerated protocol documents may be read")
    value = yaml.safe_load((PROTOCOL_ROOT / f"REV02_{task}.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Protocol {task} is not a mapping")
    return value


class FrozenProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = {task: protocol(task) for task in TASKS}

    def test_hash_receipt_itself_matches_reviewed_commit(self):
        actual = hashlib.sha256((PROTOCOL_ROOT / "hashes.json").read_bytes()).hexdigest()
        self.assertEqual(actual, FROZEN_HASH_RECEIPT_SHA256, "Receipt differs from independently reviewed protocol commit")

    def test_all_and_only_expected_protocol_files_are_hash_bound(self):
        receipt = json.loads((PROTOCOL_ROOT / "hashes.json").read_text(encoding="utf-8"))
        expected = {f"protocol/REV02_{task}.yaml" for task in TASKS}
        expected |= {"protocol/INTERPRETATION_NOTES.md", "protocol/IMPLEMENTATION_INTERFACE.md"}
        self.assertEqual(receipt["algorithm"], "SHA256")
        self.assertEqual(receipt["protocol_id"], PROTOCOL_ID)
        self.assertEqual(set(receipt["files"]), expected)
        for relative, digest in receipt["files"].items():
            target = (REV_ROOT / relative).resolve()
            self.assertEqual(target.parent, PROTOCOL_ROOT.resolve())
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest, relative)

    def test_every_task_binds_same_protocol_identity(self):
        for task, spec in self.p.items():
            with self.subTest(task=task):
                self.assertEqual(spec["protocol_id"], PROTOCOL_ID)
                self.assertEqual(spec["task_id"], task)

    def test_ten_generator_and_three_detector_seeds_are_consistent(self):
        master = self.p["MASTER"]["seeds"]
        self.assertEqual(master["generator"], GENERATOR_SEEDS)
        self.assertEqual(master["detector"], DETECTOR_SEEDS)
        self.assertEqual(master["coefficient_sensitivity"], GENERATOR_SEEDS[:3])
        self.assertTrue(set(GENERATOR_SEEDS).isdisjoint(DETECTOR_SEEDS))
        self.assertEqual(self.p["T1"]["fresh_training"]["seeds"], GENERATOR_SEEDS)
        self.assertEqual(self.p["T3A"]["scope"]["seeds"], GENERATOR_SEEDS)
        self.assertEqual(self.p["T3B"]["pool"]["generator_seeds"], GENERATOR_SEEDS)
        self.assertEqual(self.p["T5"]["factorial_design"]["seeds"], GENERATOR_SEEDS)
        for task in ("T2", "T6A", "T6B"):
            self.assertEqual(self.p[task]["detector_seeds"], DETECTOR_SEEDS)

    def test_split_counts_and_image_cluster_sizes_are_consistent(self):
        split = self.p["T1"]["split"]
        self.assertEqual(split["seed"], self.p["MASTER"]["seeds"]["split"])
        self.assertEqual(split["generation_count"], 1)
        self.assertEqual(split["expected_specimens"], {"train": 112, "validation": 32, "test": 32})
        self.assertEqual(sum(split["expected_specimens"].values()), 176)
        for name, count in split["expected_specimens"].items():
            self.assertEqual(split["expected_images"][name], 32 * count)
        self.assertIn("specimen", split["unit"])
        self.assertIn("no test IDs", split["public_content"])
        self.assertIn("without opening new test image bytes or label JSON", split["metadata_only_steward"])

    def test_new_generator_roster_and_prospective_G3_omission(self):
        fresh = self.p["T1"]["fresh_training"]
        self.assertEqual(fresh["generator_arms"], ["G0", "F00", "F01", "F10", "F11", "G2N"])
        self.assertEqual(fresh["aliases"], {"G1": "F00", "G2": "F11"})
        self.assertTrue(fresh["aliases_are_not_extra_fits"])
        self.assertEqual(fresh["not_run_prospectively"], ["G3", "D3"])
        self.assertEqual(fresh["detector_arms"], ["D0", "D1", "D2", "D4", "D5"])
        self.assertEqual(len(fresh["generator_arms"]) * len(GENERATOR_SEEDS), 60)

    def test_factorial_objectives_share_declared_weights(self):
        objectives = self.p["T1"]["generator_base"]["objectives"]
        expected = {"F00": (0, 0, 0), "F01": (0, .01, .02), "F10": (.035, 0, 0), "F11": (.035, .01, .02)}
        for arm, weights in expected.items():
            row = objectives[arm]
            self.assertEqual(tuple(row[key] for key in ("proxy", "boundary", "moment")), weights)
            self.assertEqual(row["decoder"], "pirate")
        self.assertEqual(self.p["T3A"]["losses"]["reference_G2"]["proxy_weight"], .035)

    def test_generator_evaluation_counts_and_train_only_scaler(self):
        evaluation = self.p["T3A"]["evaluation"]
        self.assertEqual(evaluation["evaluator_rng_seed"], 61003)
        self.assertEqual(self.p["T1"]["evaluation"]["common_seed"], 61003)
        self.assertEqual(evaluation["fresh_proxy"]["points_per_image"], 32)
        self.assertEqual(self.p["T3C"]["evaluation"]["residual_points_per_image"], 32)
        self.assertEqual(self.p["T3C"]["evaluation"]["residual_coordinate_seed"], 61003)
        self.assertIn("128", evaluation["diversity"]["conditions"])
        self.assertIn("stratum", evaluation["diversity"]["conditions"])
        self.assertEqual(evaluation["diversity"]["generated_samples_per_condition"], 4)
        self.assertIn("32_fixed_boundary_points", evaluation["boundary_energy"])
        self.assertIn("16x16", evaluation["centroid_error"])
        pffd = evaluation["PFFD_std"]
        self.assertEqual(len(pffd["features"]), 11)
        self.assertEqual(set(pffd["features"]), set(pffd["feature_definitions"]))
        self.assertIn("3584_new_train_images", pffd["standardization"])
        self.assertIn("below_1e-6_with_1", pffd["standardization"])
        self.assertIn("population_SD", pffd["standardization"])
        self.assertIn("exactly_equal", pffd["duplicate_feature_disclosure"])
        self.assertIn("PSD", pffd["matrix_square_root"])

    def test_primary_endpoints_and_secondary_ssim_keep_42_tests(self):
        self.assertEqual(self.p["MASTER"]["reporting"]["primary_factorial_endpoints"], PRIMARY_ENDPOINTS)
        self.assertEqual(self.p["T3A"]["evaluation"]["primary_endpoints"], PRIMARY_ENDPOINTS)
        self.assertEqual([row["id"] for row in self.p["T5"]["endpoints"]["primary"]], PRIMARY_ENDPOINTS)
        self.assertEqual(self.p["T5"]["endpoints"]["primary_count"], 7)
        self.assertEqual(self.p["T3A"]["evaluation"]["mandatory_secondary_endpoints"], ["reconstruction_SSIM"])
        self.assertEqual(self.p["T5"]["endpoints"]["mandatory_secondary"][0]["id"], "reconstruction_SSIM")
        self.assertEqual(3 * len(PRIMARY_ENDPOINTS) * 2, 42)

    def test_coefficient_sensitivity_has_exactly_18_new_fits(self):
        design = self.p["T3C"]["design"]
        variants = self.p["T3C"]["variants"]
        self.assertEqual(design["seeds"], GENERATOR_SEEDS[:3])
        self.assertEqual(design["relative_changes"], [-.5, .5])
        self.assertEqual(len(variants), 6)
        self.assertEqual(len(variants) * len(design["seeds"]), design["training_cells"])
        self.assertEqual(design["training_cells"], 18)
        for factor in design["factors"]:
            rows = [row for row in variants if row["parameter"] == factor]
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(round(row["value"] / row["baseline"], 10) for row in rows), [.5, 1.5])
        self.assertIn("baseline_coefficients", self.p["T3C"]["evaluation"]["common_RMS"])

    def test_pool_allocation_and_donor_matching(self):
        pool = self.p["T3B"]["pool"]
        self.assertEqual(pool["count_per_arm"], 7168)
        self.assertEqual(len(pool["allocations"]), len(GENERATOR_SEEDS))
        self.assertEqual(sum(pool["allocations"]), 7168)
        self.assertEqual(max(pool["allocations"]) - min(pool["allocations"]), 1)
        self.assertTrue(pool["train_only"])
        self.assertIn("identical_ordered_donor_triples", pool["required_checks"])
        self.assertIn("zero_JSON_manifest_box_mismatches", pool["required_checks"])
        self.assertIn("no latent noise or pixel noise", pool["pixel_control"])

    def test_ratio_grid_and_selection_roster(self):
        spec = self.p["T6A"]
        ratios = [.125, .25, .5, 1., 2.]
        self.assertEqual(spec["ratios"], ratios)
        self.assertEqual(spec["synthetic_counts"], [448, 896, 1792, 3584, 7168])
        self.assertEqual([int(3584 * ratio) for ratio in ratios], spec["synthetic_counts"])
        self.assertEqual(spec["pool_lineage"]["strides"], [16, 8, 4, 2, 1])
        self.assertEqual(spec["run_grid"]["required_completed_cells"], 2 * 5 * 3)
        self.assertEqual(spec["selection"]["ratio_tie_tolerance_absolute"], .002)
        self.assertEqual(spec["selection"]["split"], "validation_only")
        self.assertFalse(spec["selection"]["official_COCO_metric_used_for_selection"])
        expected_updates = sum(int(3584 * (1 + ratio) / 4) * 2 for ratio in ratios) * 2 * 3
        self.assertEqual(expected_updates, spec["run_grid"]["training_update_total"])
        self.assertEqual(expected_updates, 95424)

    def test_B4_matches_real_exposure_and_steps_not_compute(self):
        spec = self.p["T2"]["budgets"]
        b4 = spec["B4"]
        self.assertEqual(b4["work_order_option"], "b")
        self.assertEqual(b4["real_images_per_optimizer_step"], 4)
        self.assertEqual(b4["optimizer_step_milestones"], [896, 1792, 4480, 8960])
        self.assertEqual(b4["optimizer_step_milestones"], spec["B2"]["optimizer_step_milestones"])
        self.assertEqual(b4["lr_decay"], spec["B2"]["lr_decay"])
        self.assertEqual(b4["real_exposure_milestones"], [4 * s for s in b4["optimizer_step_milestones"]])
        for ratio in self.p["T6A"]["ratios"]:
            cumulative = 0
            for step in range(1, 8961):
                count = math.floor(4 * ratio * step) - math.floor(4 * ratio * (step - 1))
                self.assertGreaterEqual(count, 0)
                self.assertLessEqual(4 + count, b4["maximum_expanded_batch_size"])
                cumulative += count
                if step in b4["optimizer_step_milestones"]:
                    self.assertEqual(cumulative, int(4 * ratio * step))
        self.assertEqual(self.p["T6B"]["budget"]["optimizer_step_milestones"], b4["optimizer_step_milestones"])

    def test_unique_trajectory_and_test_checkpoint_rosters(self):
        main = self.p["T2"]["run_grid"]
        second = self.p["T6B"]["run_grid"]
        self.assertEqual((main["minimum_trajectories"], main["maximum_trajectories"]), (33, 42))
        self.assertEqual((second["minimum_trajectories"], second["maximum_trajectories"]), (12, 15))
        for selected_ratio_count in (1, 2):
            main_runs = 6 + 3 * 3 * (2 + selected_ratio_count)
            second_runs = 3 * (3 + selected_ratio_count)
            all_detector_training_runs = 30 + main_runs + second_runs
            test_checkpoints = 78 + 4 * (main_runs + second_runs)
            self.assertIn(all_detector_training_runs, (75, 87))
            self.assertIn(test_checkpoints, (258, 306))
        self.assertEqual(self.p["T2"]["prospective_reuse"]["D0"], "one_trajectory_per_seed_serves_B1_B3_B2_and_B4")
        self.assertEqual(self.p["T2"]["prospective_reuse"]["D1"], "one_trajectory_per_seed_serves_B1_B3_B2_and_B4")

    def test_consistent_detector_contrast_signs_and_multiplicity(self):
        self.assertEqual(self.p["T2"]["estimands"]["primary_contrasts"], PRIMARY_DETECTOR_CONTRASTS)
        self.assertEqual(self.p["T6B"]["estimands"]["contrasts"], PRIMARY_DETECTOR_CONTRASTS)
        stats = self.p["T5"]["detector_statistics"]
        self.assertEqual(stats["primary_contrast_order"], PRIMARY_DETECTOR_CONTRASTS)
        self.assertEqual(stats["multiplicity"]["nominal_Faster_RCNN_family_size"], 4 * 4 * 3)
        self.assertEqual(stats["multiplicity"]["nominal_RetinaNet_family_size"], 4 * 4)
        self.assertEqual(stats["multiplicity"]["alpha"], .05)
        self.assertFalse(stats["duplicated_budget_aliases"]["aliases_are_independent_observations"])
        self.assertEqual(self.p["T2"]["data_contract"]["source_provenance_field"], "source_kind")
        interface = (PROTOCOL_ROOT / "IMPLEMENTATION_INTERFACE.md").read_text(encoding="utf-8")
        self.assertIn("source_kind=real|synthetic", interface)

    def test_donor_blend_ratios_are_comparator_matched_not_selected(self):
        for task in ("T2", "T6B"):
            assignment = self.p[task]["ratio_assignment"]
            key = "D5_own_selection" if task == "T2" else "D5_own_ratio_selection"
            self.assertEqual(assignment[key], "prohibited")
            self.assertIn("union", assignment["D5"])
        self.assertEqual(self.p["T6B"]["ratio_assignment"]["family_specific_ratio_selection"], "prohibited")
        self.assertIn("same three seeds", self.p["T3B"]["detector"]["second_family"])

    def test_audit_control_population_and_flip_canonicalization(self):
        audit = self.p["T4"]
        controls = audit["controls"]
        interpolation = controls["interpolation_controls"]
        self.assertEqual(len(controls["base_control_classes"]), 5)
        self.assertEqual(interpolation["alpha_levels"], [.99, .95, .90, .80])
        self.assertEqual(controls["anchor_count"], 50)
        self.assertEqual(interpolation["count_per_alpha"], 50)
        self.assertEqual(controls["total_classes"], 9)
        self.assertEqual(controls["total_controls"], (5 + 4) * 50)
        self.assertEqual(interpolation["latent_dimension"], 16)
        self.assertEqual(interpolation["epsilon_scale"], .08)
        self.assertEqual(interpolation["extra_pixel_space_noise"], "prohibited")
        self.assertIn("Every calibration query", audit["metric_engine"]["canonical_input"])
        self.assertIn("64 by 64", audit["metric_engine"]["canonical_input"])
        self.assertEqual(audit["lookup"]["query_orientations"], ["original", "horizontal_flip"])
        self.assertEqual(audit["decision_rule"]["option"], "b_remove_feature_phash_conjunction")
        self.assertEqual(audit["metric_engine"]["phash"]["decision_role"], "diagnostic_only")
        self.assertIn("0.95_0.90_or_0.80", audit["decision_for_manuscript"]["failure_condition"])

    def test_audit_threshold_lock_precedes_controls_and_pools(self):
        calibration = self.p["T4"]["calibration"]
        self.assertEqual(calibration["query_split"], "validation")
        self.assertEqual(calibration["reference_split"], "train")
        self.assertFalse(calibration["thresholds_reused_from_historical_audit"])
        self.assertTrue(calibration["calibration_uses_flip_aware_metric_engine"])
        self.assertEqual(calibration["threshold_lock_precedes"], ["any_synthetic_pool_audit", "any_control_sensitivity_assessment"])
        self.assertEqual(calibration["post_control_or_pool_result_retuning"], "prohibited")

    def test_exact_sign_flip_degeneracy_and_family_accounting(self):
        stats = self.p["T5"]
        inference = stats["generator_inference"]
        self.assertEqual(inference["permutation_count"], 2 ** len(GENERATOR_SEEDS))
        self.assertEqual(inference["minimum_two_sided_p_for_nondegenerate_nonzero_effects"], 2 / 1024)
        self.assertFalse(inference["plus_one_Monte_Carlo_correction"])
        self.assertIsNone(inference["degenerate_policy"]["public_p"])
        self.assertTrue(inference["degenerate_policy"]["exclude_from_Holm"])
        self.assertEqual(inference["equivalence_testing"]["TOST"], "prohibited")
        multiplicity = stats["multiplicity"]
        self.assertEqual(multiplicity["primary_Holm"]["nominal_family_size"], 3)
        bh = multiplicity["mandatory_BH_sensitivity"]
        self.assertEqual(bh["nominal_family_size"], 42)
        self.assertEqual(bh["degenerate_internal_placeholder_p"], 1.)
        self.assertTrue(bh["nominal_denominator_remains_42"])
        self.assertIsNone(bh["degenerate_public_q"])
        self.assertEqual(bh["finalization_gate"], "only_after_T7_locked_test_evaluation")

    def test_global_test_campaign_requires_all_pretest_tasks(self):
        t7 = self.p["T7"]
        self.assertEqual(t7["campaign_count"], 1)
        self.assertEqual(t7["prerequisites"]["tasks"], ["T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B"])
        self.assertIn("Ratio-selection candidate models are not test-evaluated", t7["prerequisites"]["roster"])
        self.assertIn("not one model forward pass", t7["meaning"])
        self.assertIn("cannot authorize itself", t7["unlock"]["authority"])
        self.assertIn("Strictly after", t7["unlock"]["time"])
        self.assertIn("skip", t7["recovery"]["completed_cell"])
        self.assertEqual(t7["acceptance"]["choices_after_unlock"], "zero")
        self.assertIn("Test-dependent cells explicitly remain locked_pending_T7", self.p["MASTER"]["freeze_and_provenance"]["pretest_complete"])

    def test_manuscript_claims_preserve_history_and_conditionality(self):
        master = self.p["MASTER"]
        self.assertIn("previously studied corpus", master["scope"]["inference"])
        self.assertIn("not a historically untouched", master["scope"]["inference"])
        self.assertIn("Conditional on measured outcomes only", master["reporting"]["negative_results"])
        self.assertIn("do not move originals", master["reporting"]["historical_preservation"])
        self.assertEqual(self.p["T8"]["targets"], {"main_pages_max": 12, "main_tables_max": 10})
        self.assertIn("not that no beneficial ratio exists", self.p["T8"]["content"]["ratio_boundary"])
        self.assertIn("zero flags never establish no memorization", self.p["T8"]["content"]["memorization"])


class FailClosedAccessContractTests(unittest.TestCase):
    """Negative tests use fabricated temporary metadata, not study data."""

    @classmethod
    def setUpClass(cls):
        source = REV_ROOT / "scripts" / "rev02_common.py"
        if not source.exists():
            raise unittest.SkipTest("Common implementation not yet available")
        spec = importlib.util.spec_from_file_location("rev02_common_contract_audit", source)
        cls.common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.common)

    def test_empty_protocol_receipt_cannot_satisfy_integrity_gate(self):
        with tempfile.TemporaryDirectory(prefix="rev02_contract_") as directory:
            root = Path(directory)
            (root / "protocol").mkdir()
            (root / "protocol" / "hashes.json").write_text(
                json.dumps({"protocol_id": PROTOCOL_ID, "algorithm": "SHA256", "files": {}}),
                encoding="utf-8",
            )
            with patch.object(self.common, "REV_ROOT", root):
                with self.assertRaises((PermissionError, RuntimeError, ValueError)):
                    self.common.verify_protocols()

    def test_empty_source_lock_cannot_satisfy_implementation_gate(self):
        with tempfile.TemporaryDirectory(prefix="rev02_contract_") as directory:
            root = Path(directory)
            (root / "IMPLEMENTATION_LOCK.json").write_text(
                json.dumps({"implementation_commit": "toy", "source_hashes": {}}),
                encoding="utf-8",
            )
            with patch.object(self.common, "REV_ROOT", root):
                with self.assertRaises((PermissionError, RuntimeError, ValueError)):
                    self.common.verify_implementation_lock()

    def test_self_consistent_but_empty_pretest_receipts_do_not_unlock_test(self):
        with tempfile.TemporaryDirectory(prefix="rev02_contract_") as directory:
            root = Path(directory)
            freeze = {
                "last_training_validation_end_utc": "2026-01-01T00:00:00+00:00",
                "receipt_hashes": {},
            }
            freeze_path = root / "PRETEST_FREEZE.json"
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            unlock = {
                "status": "unlocked", "campaign_count": 1,
                "unlocked_utc": "2026-01-01T00:00:01+00:00",
                "pretest_freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            }
            (root / "T7_UNLOCK.json").write_text(json.dumps(unlock), encoding="utf-8")
            with patch.object(self.common, "REV_ROOT", root):
                with self.assertRaises((PermissionError, RuntimeError, ValueError)):
                    self.common.require_test_unlocked()

    def test_unlocked_test_manifest_still_must_match_frozen_split_hash(self):
        with tempfile.TemporaryDirectory(prefix="rev02_contract_") as directory:
            root = Path(directory)
            artifact = root / "fabricated_artifacts"
            sealed = artifact / "sealed"
            sealed.mkdir(parents=True)
            (root / "splits").mkdir()
            specimen = sealed / "test_specimens.json"
            specimen.write_text("{}", encoding="utf-8")
            manifest = sealed / "test_manifest.csv"
            manifest.write_text("changed fabricated bytes", encoding="utf-8")
            frozen = {"sealed_test_hashes": {
                "test_manifest.csv": hashlib.sha256(b"original fabricated bytes").hexdigest(),
                "test_specimens.json": hashlib.sha256(specimen.read_bytes()).hexdigest(),
            }}
            split = root / "splits" / "rev02_split.json"
            split.write_text(json.dumps(frozen), encoding="utf-8")
            (root / "splits" / "rev02_split.sha256.json").write_text(
                json.dumps({"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}), encoding="utf-8")
            with patch.object(self.common, "REV_ROOT", root), patch.object(self.common, "ARTIFACT_ROOT", artifact), \
                    patch.object(self.common, "require_test_unlocked", return_value={}), \
                    patch.object(self.common, "record_event"):
                with self.assertRaises((PermissionError, RuntimeError, ValueError)):
                    self.common.manifest_path("test")


class CoordinatorRosterContractTests(unittest.TestCase):
    """Inspect pure run-roster builders; never resolve study manifests or results."""

    @classmethod
    def setUpClass(cls):
        common_spec = importlib.util.spec_from_file_location(
            "rev02_common", REV_ROOT / "scripts" / "rev02_common.py")
        common = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common)
        source = REV_ROOT / "scripts" / "run_rev02_pipeline.py"
        if not source.exists():
            raise unittest.SkipTest("Coordinator implementation not yet available")
        spec = importlib.util.spec_from_file_location("rev02_pipeline_contract_audit", source)
        cls.pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"rev02_common": common}):
            spec.loader.exec_module(cls.pipeline)

    def test_initial_jobs_are_unique_and_never_request_test(self):
        jobs = self.pipeline.initial_jobs()
        self.assertEqual(len({j["job_id"] for j in jobs}), len(jobs))
        self.assertTrue(all("test" not in j["arguments"] and "all" not in j["arguments"] for j in jobs))
        train = [j for j in jobs if j["module"] == "rev02_generator" and j["arguments"][0] == "train"]
        validation = [j for j in jobs if j["module"] == "rev02_generator" and j["arguments"][0] == "evaluate"]
        ratio = [j for j in jobs if j["module"] == "rev02_detector" and j["arguments"][0] == "ratio"]
        self.assertEqual((len(train), len(validation), len(ratio)), (78, 78, 30))
        fit_keys = {(j["arguments"][2], int(j["arguments"][4])) for j in train}
        expected = {(arm, seed) for arm in ("G0", "F00", "F01", "F10", "F11", "G2N") for seed in GENERATOR_SEEDS}
        expected |= {(f"F11_{constant}_{level}", seed) for constant in ("kappa", "source", "decay") for level in ("low", "high") for seed in GENERATOR_SEEDS[:3]}
        self.assertEqual(fit_keys, expected)
        self.assertEqual(jobs[-1]["arguments"], ["select-ratios"])

    def test_every_possible_selected_ratio_pair_preserves_unique_budget_roster(self):
        for r2 in (.125, .25, .5, 1., 2.):
            for r4 in (.125, .25, .5, 1., 2.):
                with self.subTest(D2=r2, D4=r4):
                    rows = self.pipeline.budget_configurations({"D2": r2, "D4": r4})
                    unique_ratios = len({r2, r4})
                    keys = [(r["family"], r["arm"], r["axis"], r["ratio"], r["seed"]) for r in rows]
                    self.assertEqual(len(keys), len(set(keys)))
                    self.assertEqual(len(rows), 33 + 12 * unique_ratios)
                    self.assertEqual(sum(r["family"] == "fasterrcnn" for r in rows), 24 + 9 * unique_ratios)
                    self.assertEqual(sum(r["family"] == "retinanet" for r in rows), 9 + 3 * unique_ratios)
                    for row in rows:
                        self.assertIn(row["seed"], DETECTOR_SEEDS)
                        if row["family"] == "retinanet":
                            self.assertEqual(row["axis"], "B4")
                            self.assertNotEqual(row["arm"], "D1")
                        elif row["arm"] in ("D0", "D1"):
                            self.assertEqual(row["axis"], "B2")
                        allowed = {0.} if row["arm"] in ("D0", "D1") else {r2, r4} if row["arm"] == "D5" else {r2 if row["arm"] == "D2" else r4}
                        self.assertIn(row["ratio"], allowed)


class DetectorSpecimenContractTests(unittest.TestCase):
    def test_thirty_two_images_are_not_thirty_two_camera_views(self):
        source = REV_ROOT / "scripts" / "rev02_detector.py"
        spec = importlib.util.spec_from_file_location("rev02_detector_contract_audit", source)
        detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detector)
        rows = [{"specimen": "fabricated_specimen", "sample_id": f"fabricated_image_{i:02d}",
                 "view": "ON_AXIS" if i < 16 else "OFF_AXIS", "source_kind": "real"}
                for i in range(32)]
        detector._validate_specimen_roster(rows, 1)
        duplicate = [dict(row) for row in rows]
        duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
        with self.assertRaises((RuntimeError, ValueError)):
            detector._validate_specimen_roster(duplicate, 1)


if __name__ == "__main__":
    unittest.main()
