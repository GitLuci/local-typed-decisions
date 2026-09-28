"""test2_fastpath.py: never resumes over committed reports without --force, and the summary tolerates `random`."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import test2_fastpath as fp  # noqa: E402


def test_guard_refuses_committed_folder_without_force(tmp_path):
    with pytest.raises(SystemExit, match="--tag"):
        fp.guard_output(tmp_path / "fastpath-06b", force=False, tracked=["summary.json"])


def test_guard_allows_force_and_untracked_folders(tmp_path):
    fp.guard_output(tmp_path / "fastpath-06b", force=True, tracked=["summary.json"])
    fp.guard_output(tmp_path / "fastpath-06b-repro", force=False, tracked=[])


def test_guard_detects_the_real_committed_baseline():
    out = ROOT / "reports/test-2/fastpath-06b"
    assert fp.tracked_files(out), "the committed 0.6B baseline must count as committed"
    with pytest.raises(SystemExit):
        fp.guard_output(out, force=False)
    assert fp.tracked_files(ROOT / "reports/test-2/fastpath-does-not-exist") == []


def test_domain_counts_tolerates_random_domain_without_correct_a():
    domains = {"random": {"n_labelled": 0, "tv_known_distribution_mean": 0.53},
               "factual": {"n_labelled": 18, "correct_A": 6, "correct_B": 6}}
    assert fp.domain_counts(domains) == {"random": (None, 0), "factual": (6, 18)}
