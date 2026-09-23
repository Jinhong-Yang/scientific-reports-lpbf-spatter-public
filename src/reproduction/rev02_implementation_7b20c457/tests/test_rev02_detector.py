"""Toy-only detector contracts; never open any real data or test manifest."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import rev02_detector as d


@pytest.mark.parametrize("ratio", [.125, .25, .5, 1., 2.])
def test_exact_B4_counts_all_frozen_milestones(ratio):
    sampler = d.ExposureSampler(3584, int(3584*ratio), ratio, "B4", 51001)
    real = syn = 0
    for step in range(1, 8961):
        indices = sampler.batch(step)
        nreal = sum(i < 3584 for i in indices)
        nsyn = len(indices)-nreal
        assert nreal == 4
        assert nsyn == d.synthetic_count(ratio, step)
        assert len(indices) <= 12
        real += nreal
        syn += nsyn
        if step in [896, 1792, 4480, 8960]:
            assert real == 4*step
            assert syn == int(4*ratio*step)


@pytest.mark.parametrize("axis", ["B1B3", "B2", "B4"])
def test_sampler_resume_is_bit_exact_across_epoch_boundary(axis):
    first = d.ExposureSampler(32, 16, .5, axis, 51001)
    for step in range(1, 12):
        first.batch(step)
    state = first.state_dict()
    expected = [first.batch(step) for step in range(12, 51)]
    restored = d.ExposureSampler(32, 16, .5, axis, 17)
    restored.load_state_dict(state)
    assert [restored.batch(step) for step in range(12, 51)] == expected


def test_full_mixed_epoch_exact_counts():
    sampler = d.ExposureSampler(32, 16, .5, "B1B3", 51001)
    for epoch in range(3):
        observed = [i for s in range(epoch*12+1, (epoch+1)*12+1) for i in sampler.batch(s)]
        assert sorted(observed) == list(range(48))


def test_wrong_source_and_noninteger_roster_rejected():
    with pytest.raises(ValueError):
        d.ExposureSampler(32, 15, .5, "B2", 1)
    with pytest.raises(ValueError):
        d.RowsDataset([{"source_kind": "unknown"}], "fasterrcnn", resolver=Path)


def test_prospective_alias_and_unique_grid_counts():
    ids = [d.budget_run_id("fasterrcnn", "D0", a, 0, 51001) for a in ["B1B3", "B2", "B4"]]
    assert len(set(ids)) == 1
    assert ids[0] == "budget_fasterrcnn_D0_r0_shared_s51001"
    for choices, main_count, retina_count in [({"D2": .125, "D4": .125}, 33, 12), ({"D2": .125, "D4": .5}, 42, 15)]:
        rows = d.expected_budget_roster(choices, [51001,51002,51003])
        assert len({r["run_id"] for r in rows if r["family"] == "fasterrcnn"}) == main_count
        assert len({r["run_id"] for r in rows if r["family"] == "retinanet"}) == retina_count


def test_ratio_grid_closed_and_smaller_near_tie():
    ratios = [.125,.25,.5,1.,2.]
    seeds = [51001,51002,51003]
    rows = [{"ratio": r, "seed": s, "validation_study_map_50_95": .60 + (.001 if r == .25 else 0)} for r in ratios for s in seeds]
    assert d.select_grid(rows, ratios, seeds)["selected_ratio"] == .125
    with pytest.raises(ValueError):
        d.select_grid(rows[:-1], ratios, seeds)
    with pytest.raises(ValueError):
        d.select_grid(rows + [rows[0]], ratios, seeds)
    rows[0]["validation_study_map_50_95"] = float("nan")
    with pytest.raises(ValueError):
        d.select_grid(rows, ratios, seeds)


def _record(boxes, scores, gt=True):
    return {"gt_boxes": [[10,10,30,30]] if gt else [], "boxes": boxes, "scores": scores,
            "labels": [1]*len(boxes), "width": 300, "height": 300}


def test_official_COCO_perfect_empty_predictions_and_empty_GT():
    perfect = d.official_coco_metrics([_record([[10,10,30,30]], [.9])])
    assert perfect["COCO_AP"] == pytest.approx(1)
    assert perfect["COCO_AP_small"] == pytest.approx(1)
    assert perfect["COCO_AP_large"] is None
    missing = d.official_coco_metrics([_record([], [])])
    assert missing["COCO_AP"] == 0
    no_gt = d.official_coco_metrics([_record([[10,10,30,30]], [.9], False)])
    assert no_gt["COCO_AP"] is None
    assert d.official_coco_metrics([])["COCO_images"] == 0


def test_official_COCO_rejects_foreground_mapping_error():
    row = _record([[10,10,30,30]], [.9])
    row["labels"] = [0]
    with pytest.raises(ValueError):
        d.official_coco_metrics([row])


def test_toy_dataset_label_mapping_and_explicit_source(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((300,300), 127, dtype=np.uint8)).save(image_path)
    row = {"source_kind": "real", "sample_id": "toy1", "specimen": "toy", "view": "1",
           "image_path": str(image_path), "bbox_x_px": "10", "bbox_y_px": "20",
           "bbox_width_px": "30", "bbox_height_px": "40"}
    image, target, meta = d.RowsDataset([row], "retinanet", resolver=Path)[0]
    assert image.shape == (3,300,300)
    assert target["labels"].tolist() == [0]
    assert target["boxes"].tolist() == [[10,20,40,60]]
    assert meta["source_kind"] == "real"
    assert d.RowsDataset([row], "fasterrcnn", resolver=Path)[0][1]["labels"].tolist() == [1]


def test_rng_capture_restores_python_numpy_torch():
    random = __import__("random")
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    state = d._rng_state()
    first = (random.random(), np.random.rand(), torch.rand(3))
    d._restore_rng(state)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_schedule_four_canonical_labels():
    p2 = {"budgets": {"B2": {"optimizer_step_milestones": [896,1792,4480,8960]},
                       "B1_B3": {"epoch_milestones": [1,2,5,10]}}}
    assert d._schedule("budget", "B1B3", 3584, 1792, p2, {}) == {896:1344,1792:2688,4480:6720,8960:13440}
    assert d._schedule("budget", "B4", 3584, 1792, p2, {}) == {896:896,1792:1792,4480:4480,8960:8960}


def test_one_class_retinanet_keeps_v2_head_semantics():
    # Random initialization only: no pretrained download and no image/data load.
    model = d.build_model("retinanet", pretrained=False)
    head = model.head.classification_head
    assert head.num_classes == 1
    assert head.cls_logits.out_channels == head.num_anchors
    assert model.head.regression_head._loss_type == "giou"
    assert any(isinstance(m, torch.nn.GroupNorm) for m in head.modules())
    assert model.score_thresh == .05
    assert model.nms_thresh == .5
    assert model.topk_candidates == 1000
    assert model.detections_per_img == 100


def test_evaluation_is_idempotent_and_foreground_only(tmp_path):
    image_path = tmp_path / "toy.png"
    Image.fromarray(np.full((300,300), 127, dtype=np.uint8)).save(image_path)
    row = {"source_kind": "real", "sample_id": "toy1", "specimen": "toy", "view": "1",
           "image_path": str(image_path), "bbox_x_px": "10", "bbox_y_px": "10", "bbox_width_px": "20", "bbox_height_px": "20"}
    dataset = d.RowsDataset([row], "retinanet", resolver=Path)
    class ToyModel:
        calls = 0
        def eval(self):
            return self
        def __call__(self, images):
            self.calls += 1
            return [{"labels": torch.tensor([0,1]), "boxes": torch.tensor([[10.,10.,30.,30.],[50.,50.,70.,70.]]),
                     "scores": torch.tensor([.9,.95])} for _ in images]
    model = ToyModel()
    metadata = {"checkpoint_sha256": "a"*64, "milestone": 896}
    result = d._output_evaluation(model, dataset, torch.device("cpu"), tmp_path / "run", 896, "validation", metadata)
    assert result["metrics"]["COCO_AP"] == pytest.approx(1)
    assert result["metrics"]["study_mAP_50_95"] == pytest.approx(1)
    assert model.calls == 1
    assert d._output_evaluation(model, dataset, torch.device("cpu"), tmp_path / "run", 896, "validation", metadata) == result
    assert model.calls == 1


def test_partial_evaluation_does_not_silently_rerun(tmp_path):
    c = d.common()
    destination = tmp_path / "test"
    c.write_json(destination / "milestone_896_started.json", {"partial": True})
    with pytest.raises(RuntimeError, match="documented recovery"):
        d._output_evaluation(None, None, None, tmp_path, 896, "test", {"checkpoint_sha256": "a"*64})


def test_training_guard_fails_before_protocol_or_data_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path / "T7_UNLOCK.json").write_text("{}")
    monkeypatch.setattr(d, "common", lambda: SimpleNamespace(REV_ROOT=tmp_path))
    with pytest.raises(PermissionError):
        d._validate_request("ratio", "fasterrcnn", "D2", "B1B3", .5, 51001)


def test_checkpoint_atomic_retention_and_hash_detection(tmp_path):
    path = tmp_path / "checkpoint.pt"
    digest = d._atomic_torch(path, {"toy": torch.tensor([1.,2.])})
    assert torch.equal(torch.load(path, weights_only=True)["toy"], torch.tensor([1.,2.]))
    d._verify_files({str(path): digest})
    with pytest.raises(FileExistsError):
        d._atomic_torch(path, {"toy": torch.tensor([3.])})
    path.write_bytes(b"corrupted toy checkpoint")
    with pytest.raises(RuntimeError):
        d._verify_files({str(path): digest})


@pytest.mark.parametrize("ledger", [{}, {"toy.pt": "not-a-sha256"}, []])
def test_empty_or_malformed_ledgers_fail_closed(ledger):
    with pytest.raises(RuntimeError):
        d._verify_files(ledger)


def test_specimen_sample_matrix_accepts_two_camera_views_and_rejects_duplicate_ID():
    rows = [{"specimen": "toy", "sample_id": f"toy_{v:02d}",
             "view": "ON_AXIS" if v < 16 else "OFF_AXIS"} for v in range(32)]
    d._validate_specimen_roster(rows, 1)
    rows[-1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(RuntimeError):
        d._validate_specimen_roster(rows, 1)
