"""Synthetic-only unit tests; no real train, validation or test files are read."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"rev02_generator.py"
spec = importlib.util.spec_from_file_location("rev02_generator_unit", SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def row(index=0, material="SUS316L", view="ON_AXIS", stratum="SUS|100|500"):
    return {"sample_id": f"sample{index:03d}", "specimen": f"spec{index}", "stratum": stratum,
            "view": view, "material": material, "split": "train", "source_kind": "real",
            "image_path": f"toy{index}.png", "label_path": f"toy{index}.json", "laser_power_w": "100",
            "scan_speed_mm_s": "500", "line_energy_j_mm": ".2", "bbox_x_px": "20", "bbox_y_px": "30",
            "bbox_width_px": "4", "bbox_height_px": "12", "image_width_px": "300", "image_height_px": "300",
            "lamination_direction_deg": "0", "image_sha256": f"toyhash{index}"}


def test_arm_roster_and_weights():
    assert g.canonical_arm("G1") == "F00"
    assert g.canonical_arm("G2") == "F11"
    assert len(g.BASE_ARMS) == 6 and len(g.SENSITIVITY) == 6
    for value in ("G3", "D3", "other"):
        with pytest.raises(ValueError):
            g.canonical_arm(value)
    assert not any(g.objective_weights("F00").values())
    assert g.objective_weights("G2N") == {"proxy": 0., "boundary": .01, "moment": .02, "blob": .035, "laplacian": .035}


def test_independent_input_streams_pair_across_arms():
    a = g.draw_training_inputs(41001, 13, 2, 64, 8, 16, 32, 32, "cpu")
    torch.randn(999)  # unrelated global draws cannot alter paired inputs
    b = g.draw_training_inputs(41001, 13, 2, 64, 8, 16, 32, 32, "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    changed = g.draw_training_inputs(41002, 13, 2, 64, 8, 16, 32, 32, "cpu")
    assert not torch.equal(a[0], changed[0])
    assert a[2].shape == (2, 32, 2)
    assert (a[3].abs() == 1).any(-1).all()


def test_fixed_boundary_protocol():
    boundary = g.fixed_boundary()
    assert boundary.shape == (32, 2)
    assert len(torch.unique(boundary, dim=0)) == 28
    assert torch.equal(boundary[:8, 0], -torch.ones(8))


def test_coefficients_defaults_and_single_override():
    condition = g.condition_tensor([row()])
    baseline = g.coefficients(condition)
    changed = g.coefficients(condition, {"diffusivity_intercept": .009})
    assert torch.allclose(changed["diffusivity"], baseline["diffusivity"]-.009)
    for key in baseline:
        if key != "diffusivity":
            assert torch.equal(changed[key], baseline[key])
    with pytest.raises(ValueError):
        g.coefficients(condition, {"unknown": 1})


def test_nopde_blob_unit_amplitude_and_unweighted_laplacian():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.1))
        def forward(self, coords, condition, latent):
            return self.scale*(coords[..., :1]**2+coords[..., 1:]**2)+.2
    field = Field()
    cond = g.condition_tensor([row()])
    z = torch.zeros(1, 16)
    coords = torch.tensor([[[.1, .2], [.3, .4], [-.1, .6]]])
    boundary = g.fixed_boundary()[None]
    moment = g.make_coordinate_grid(16)[None]
    values = g.regularizer_components(field, cond, z, coords, boundary, moment)
    assert values["laplacian"].item() == pytest.approx(.16)
    loss = sum(g.objective_weights("G2N")[key]*value for key, value in values.items())
    loss.backward()
    assert torch.isfinite(field.scale.grad)
    f, grad, lap, xy = g.field_derivatives(field, cond, z, cond[:, None, 7:9])
    _, gaussian = g.residual_from_derivatives(f, grad, lap, xy, cond)
    assert gaussian.item() == pytest.approx(1.)


def test_condition_shape_and_canonical_geometry():
    condition = g.condition_tensor([row()])
    assert condition.shape == (1, 13)
    assert condition[0, 7].item() == pytest.approx(2*(20+2)/300-1)


def test_donor_plan_deterministic_ties_and_shared_noise():
    rows = [row(i) for i in range(5)]
    plan = g.create_donor_plan(rows, rows, 61002)
    repeat = g.create_donor_plan(list(reversed(rows)), rows, 61002)
    assert plan == repeat
    assert all(.2 <= e["alpha"] <= .8 and len(e["epsilon"]) == 16 for e in plan)
    assert len({tuple(e["epsilon"]) for e in plan}) == len(plan)
    text = json.dumps(plan)
    assert json.loads(text) == plan


def test_donor_material_view_enforced_no_fallback():
    with pytest.raises(RuntimeError, match="No matching"):
        g.create_donor_plan([row(0)], [row(1, material="INCONEL")], 61002)


def test_allocations_and_target_order():
    rows = [row(i) for i in range(3)]
    target = [r for r in rows for _ in range(2)]
    plan = g.create_donor_plan(rows, target, 61002, [(41001, 3), (41002, 3)])
    assert [e["generator_seed"] for e in plan] == [41001]*3+[41002]*3
    assert [e["target_sample_id"] for e in plan] == ["sample000", "sample000", "sample001", "sample001", "sample002", "sample002"]
    with pytest.raises(ValueError):
        g.create_donor_plan(rows, target, 1, [(41001, 5)])


def test_diversity_is_stratum_view_balanced():
    rows = [row(i, view=view, stratum=stratum) for stratum in ("a", "b") for view in ("ON", "OFF") for i in range(4)]
    selected = g.select_diversity_rows(rows)
    assert len(selected) == 8
    assert {r["sample_id"] for r in selected} == {"sample000", "sample001"}


def test_fresh_evaluation_coordinates_shared():
    rows = [row(i) for i in range(3)]
    plan = g.create_donor_plan(rows, rows, 61003, evaluation=True)
    assert np.asarray(plan[0]["collocation"]).shape == (32, 2)
    assert plan == g.create_donor_plan(rows, rows, 61003, evaluation=True)


def test_train_scaler_and_duplicate_feature_definition():
    gen = np.random.default_rng(1)
    matrix = g.features(gen.random((8, 8, 8)))
    assert np.array_equal(matrix[:, 0], matrix[:, 2])
    scaler = g.fit_scaler(matrix)
    scaled = g.standardized(matrix, scaler)
    assert np.allclose(scaled.mean(0), 0., atol=1e-12)
    constant = g.fit_scaler(np.ones((3, 11)))
    assert constant["scale"] == [1.]*11
    assert constant["fit_scope"] == "train_only"


def test_frechet_psd_symmetry_identity_shift():
    generator = np.random.default_rng(4)
    real = generator.normal(size=(100, 4))
    real[:, 3] = real[:, 0]  # intentional inherited duplicate coordinate
    assert g.frechet_distance(real, real) == pytest.approx(0., abs=1e-7)
    changed = real+1
    assert g.frechet_distance(real, changed) == pytest.approx(4., abs=1e-7)
    assert g.frechet_distance(real, changed) == pytest.approx(g.frechet_distance(changed, real), abs=1e-7)
    with pytest.raises(FloatingPointError):
        g.psd_eigh(np.diag([-1e-4, 1.]))


def test_frechet_invalid_inputs():
    with pytest.raises(ValueError):
        g.frechet_distance(np.ones((1, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        g.frechet_distance(np.full((2, 3), np.nan), np.ones((2, 3)))


def test_pixel_blend_and_conversion_shared():
    left, right = np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32)
    blended = g.pixel_blend(left, right, .25)
    assert np.allclose(blended, .75)
    pixels = g.output_pixels(blended)
    assert pixels.shape == (300, 300) and pixels.dtype == np.uint8
    assert np.unique(pixels).tolist() == [191]


def test_synthetic_label_matches_target_box():
    entry = g.create_donor_plan([row()], [row()], 61002, [(41001, 1)])[0]
    label = g.synthetic_label(entry, "D5", "toy.png")
    for short, value in (("x", 20), ("y", 30), ("width", 4), ("height", 12)):
        assert label[f"anotation.bbox.{short}"] == value


def test_latent_plan_exact_pairing():
    entries = g.create_donor_plan([row(i) for i in range(2)], [row()], 1)
    bank = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    result = g.plan_latent(entries, bank)
    entry = entries[0]
    expected = entry["alpha"]*bank[entry["donor_a_index"]]+(1-entry["alpha"])*bank[entry["donor_b_index"]]+.08*torch.tensor(entry["epsilon"])
    assert torch.allclose(result[0], expected)


def test_stratified_halves_no_overlap_equal_groups():
    rows = [row(i, view=v, stratum=s) for s in ("a", "b") for v in ("on", "off") for i in range(4)]
    a, b = g.stratified_halves(rows, 61006)
    assert len(a) == len(b) == 8
    assert not set(a)&set(b)
    assert set(a)|set(b) == set(range(16))
    assert np.array_equal(a, g.stratified_halves(rows, 61006)[0])


def test_descriptive_not_equivalence():
    rows = []
    for arm in ("F11", "G2N"):
        for seed in range(41001, 41011):
            rows.append({"arm": arm, "seed": seed, "split": "validation", **{key: 1. for key in g.PRIMARY}})
    output = g.descriptive_G2N(rows)
    assert len(output) == 7 and all(r["descriptive_within_seed_SD"] for r in output)
    assert not any(r["equivalence_claim"] for r in output)
    rows[-1][g.PRIMARY[0]] = 3.
    assert g.descriptive_G2N(rows)[0]["sample_SD_of_paired_differences"] > 0


def test_atomic_checkpoint_full_optimizer_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    state = g.rng_state()
    path = tmp_path/"resume.pt"
    g.atomic_torch_save(path, {"model": model.state_dict(), "optimizer": opt.state_dict(), "rng": state, "next_step": 1})
    restored = torch.load(path, weights_only=False)
    assert restored["next_step"] == 1 and restored["optimizer"]["state"]
    g.restore_rng(restored["rng"])
    a = torch.randn(3)
    g.restore_rng(state)
    assert torch.equal(a, torch.randn(3))


def test_gate_test_before_any_loader(monkeypatch):
    def deny():
        raise PermissionError("locked")
    fake = SimpleNamespace(require_test_unlocked=deny, load_rows=lambda split: pytest.fail("must not load"))
    monkeypatch.setattr(g, "common", lambda: fake)
    with pytest.raises(PermissionError):
        g.sorted_rows("test")
    with pytest.raises(PermissionError):
        g.ensure_evaluation_plan("test")


def test_training_prohibited_after_unlock(tmp_path, monkeypatch):
    (tmp_path/"T7_UNLOCK.json").write_text("{}")
    monkeypatch.setattr(g, "common", lambda: SimpleNamespace(REV_ROOT=tmp_path))
    with pytest.raises(PermissionError):
        g.train("F11", 41001)


def test_cli_requires_evaluation_split():
    with pytest.raises(SystemExit):
        g.main(["evaluate", "--arm", "F11", "--seed", "41001"])


def test_source_does_not_call_unsafe_loaders():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dataset_splits(" not in source
    assert "evaluate_checkpoint(" not in source
    assert "manifest.csv\"" in source  # only the new pool manifest is used


def test_test_roster_unique_path_and_hash(tmp_path, monkeypatch):
    run = tmp_path/"F11_s41001"
    run.mkdir()
    checkpoint = run/"best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    entry = {"arm": "F11", "seed": 41001, "checkpoint": str(checkpoint), "checkpoint_sha256": digest}
    roster = {"test_roster": {"generators": [entry]}}
    fake = SimpleNamespace(REV_ROOT=tmp_path, require_test_unlocked=lambda: None,
        load_json=lambda path: roster, generator_run_dir=lambda arm, seed: run,
        sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(g, "common", lambda: fake)
    assert g.verify_test_roster_entry("G2", 41001) == entry
    with pytest.raises(PermissionError, match="absent"):
        g.verify_test_roster_entry("F10", 41001)
    roster["test_roster"]["generators"].append(dict(entry))
    with pytest.raises(PermissionError, match="duplicated"):
        g.verify_test_roster_entry("F11", 41001)
    roster["test_roster"]["generators"].pop()
    checkpoint.write_bytes(b"tampered synthetic checkpoint")
    with pytest.raises(PermissionError, match="hash differs"):
        g.verify_test_roster_entry("F11", 41001)


def test_completed_artifact_rejects_identity_and_byte_tampering(tmp_path, monkeypatch):
    identity = {"example": 1}
    target = tmp_path/"best.pt"
    target.write_bytes(b"fixture")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    summary = {"status": "completed", "identity_sha256": g.object_hash(identity), "artifact_hashes": {"best.pt": digest}}
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    fake = SimpleNamespace(load_json=lambda path: json.loads(Path(path).read_text()),
                           sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(g, "common", lambda: fake)
    assert g.check_completed(tmp_path, identity)["status"] == "completed"
    with pytest.raises(RuntimeError, match="identity differs"):
        g.check_completed(tmp_path, {"example": 2})
    target.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="corrupted"):
        g.check_completed(tmp_path, identity)


def test_full_optimizer_resume_next_step_equivalence(tmp_path):
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    def step(net, opt):
        opt.zero_grad()
        loss = (net(torch.ones(3, 2))-.5).square().mean()
        loss.backward()
        opt.step()
    step(model, optimizer)
    scheduler.step()
    path = tmp_path/"full.pt"
    g.atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": g.rng_state()})
    step(model, optimizer)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    restored = torch.nn.Linear(2, 1)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.5)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10, eta_min=1e-5)
    saved = torch.load(path, weights_only=False)
    restored.load_state_dict(saved["model"])
    opt2.load_state_dict(saved["optimizer"])
    sched2.load_state_dict(saved["scheduler"])
    g.restore_rng(saved["rng"])
    step(restored, opt2)
    assert all(torch.equal(value, expected[key]) for key, value in restored.state_dict().items())
    assert sched2.state_dict() == scheduler.state_dict()
