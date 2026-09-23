from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rev02_common as common
from prepare_rev02 import canonical_box


def test_protocol_hashes():
    assert common.verify_protocols()["protocol_id"].endswith("20260902-v1")


def test_explicit_test_gate_precedes_read(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REV_ROOT", tmp_path / "public")
    monkeypatch.setattr(common, "ARTIFACT_ROOT", tmp_path / "large")
    sealed = tmp_path / "large" / "sealed" / "test_manifest.csv"
    with pytest.raises(PermissionError):
        common.manifest_path("test")
    with pytest.raises(PermissionError):
        common.read_csv(sealed)
    with pytest.raises(PermissionError):
        common.sha256_file(sealed)
    with pytest.raises(PermissionError):
        common.load_json(sealed.with_suffix(".json"))
    assert not sealed.exists()


def test_box_floor_and_edge():
    source = dict(bbox_x_px="299", bbox_y_px="-1", bbox_width_px="1", bbox_height_px="2", image_width_px="300", image_height_px="300")
    actual, changed = canonical_box(source)
    assert changed and actual == dict(bbox_x_px=296, bbox_y_px=0, bbox_width_px=4, bbox_height_px=4, bbox_area_px2=16)


def test_zero_box_is_floored_but_negative_is_rejected():
    source = dict(bbox_x_px=100, bbox_y_px=100, bbox_width_px=0, bbox_height_px=0, image_width_px=300, image_height_px=300)
    assert canonical_box(source)[0]["bbox_width_px"] == 4
    with pytest.raises(ValueError):
        canonical_box({**source, "bbox_width_px": -1})


def test_immutable_finite_json(tmp_path):
    path = tmp_path / "data.json"
    common.write_json(path, {"value": None})
    with pytest.raises(FileExistsError):
        common.write_json(path, {"value": 2})
    with pytest.raises(ValueError):
        common.write_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert common.load_json(path) == {"value": None}


def test_event_chain_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REV_ROOT", tmp_path)
    common.record_event("first", {"value": 1})
    common.record_event("second", {"value": 2})
    events = common.verify_event_chain()
    assert len(events) == 2 and events[1]["previous_sha256"] == events[0]["event_sha256"]
    path = tmp_path / "logs" / "events.jsonl"
    value = path.read_text(encoding="utf-8").replace('"value":1', '"value":3')
    path.write_text(value, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Modified"):
        common.verify_event_chain()


def test_empty_freeze_cannot_unlock(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REV_ROOT", tmp_path)
    freeze = tmp_path / "PRETEST_FREEZE.json"
    common.write_json(freeze, {"receipt_hashes": {}, "status": "pretest_complete"})
    common.write_json(tmp_path / "T7_UNLOCK.json", {"status": "unlocked", "campaign_count": 1,
                                                   "pretest_freeze_sha256": common.sha256_file(freeze)})
    with pytest.raises(PermissionError, match="nine"):
        common.require_test_unlocked()


@pytest.mark.parametrize("tampered_name", [None, "test_manifest.csv", "test_specimens.json"])
def test_test_manifest_binds_both_sealed_files_after_unlock(tmp_path, monkeypatch, tampered_name):
    monkeypatch.setattr(common, "REV_ROOT", tmp_path / "public")
    monkeypatch.setattr(common, "ARTIFACT_ROOT", tmp_path / "large")
    monkeypatch.setattr(common, "require_test_unlocked", lambda: {"status": "unlocked"})
    sealed = common.ARTIFACT_ROOT / "sealed"
    common.atomic_text(sealed / "test_manifest.csv", "sample_id\nfabricated_sample\n")
    common.write_json(sealed / "test_specimens.json", {"specimens": ["fabricated_specimen"]})
    split_path = common.REV_ROOT / "splits" / "rev02_split.json"
    common.write_json(split_path, {"sealed_test_hashes": {
        name: common.sha256_file(sealed / name) for name in ("test_manifest.csv", "test_specimens.json")}})
    common.write_json(split_path.with_name("rev02_split.sha256.json"), {"sha256": common.sha256_file(split_path)})
    if tampered_name is not None:
        common.atomic_text(sealed / tampered_name, "fabricated_changed_bytes\n", overwrite=True)
        with pytest.raises(RuntimeError, match="Frozen sealed test file changed"):
            common.manifest_path("test")
    else:
        assert common.manifest_path("test") == sealed / "test_manifest.csv"
