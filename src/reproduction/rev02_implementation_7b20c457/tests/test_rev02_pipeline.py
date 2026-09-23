from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_rev02_pipeline as pipeline


def test_generator_training_roster_exact():
    jobs = pipeline.initial_jobs()
    training = [job for job in jobs if job["module"] == "rev02_generator" and job["arguments"][0] == "train"]
    assert len(training) == 78
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert all("test" not in job["arguments"] for job in jobs)


def test_ratio_roster_is_frozen_thirty_cells():
    jobs = pipeline.initial_jobs()
    ratios = [job for job in jobs if job["module"] == "rev02_detector" and job["arguments"][0] == "ratio"]
    assert len(ratios) == 30
    assert not any("D5" in row["arguments"] for row in ratios)


def test_budget_counts_with_one_or_two_selected_ratios():
    same = pipeline.budget_configurations({"D2": 0.125, "D4": 0.125})
    different = pipeline.budget_configurations({"D2": 0.125, "D4": 0.5})
    assert len(same) == 45 and len(different) == 57
    assert sum(row["family"] == "fasterrcnn" for row in same) == 33
    assert sum(row["family"] == "retinanet" for row in different) == 15
    assert {row["ratio"] for row in different if row["arm"] == "D5"} == {0.125, 0.5}
    assert all(row["axis"] == "B4" for row in different if row["family"] == "retinanet")


def test_no_g3_d3_or_independent_d5_selection():
    assert "G3" not in pipeline.BASE_ARMS
    configs = pipeline.budget_configurations({"D2": 2.0, "D4": 0.25})
    assert all(row["arm"] != "D3" for row in configs)
    assert all(row["axis"] == "B2" for row in configs if row["family"] == "fasterrcnn" and row["arm"] in ("D0", "D1"))


def test_test_campaign_uses_lexicographic_frozen_run_order():
    roster = {"generators": [{"arm": "G0", "seed": 41001},
                              {"arm": "F00", "seed": 41002},
                              {"arm": "F00", "seed": 41001}],
              "detectors": [{"run_id": "budget_retinanet_z"},
                            {"run_id": "budget_fasterrcnn_a"}]}
    jobs = pipeline.test_jobs(roster)
    gen_jobs = [row["arguments"] for row in jobs if row["module"] == "rev02_generator" and row["arguments"][0] == "evaluate"]
    assert [(row[2], row[4]) for row in gen_jobs] == [("F00", "41001"), ("F00", "41002"), ("G0", "41001")]
    det_jobs = [row["arguments"] for row in jobs if row["module"] == "rev02_detector" and row["arguments"][0] == "evaluate"]
    assert [row[2] for row in det_jobs] == ["budget_fasterrcnn_a", "budget_retinanet_z"]
