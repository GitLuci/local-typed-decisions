"""fetch_goemotions: shipped states have the GoEmotions texts redacted; restoration is pinned and verified."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_goemotions as F  # noqa: E402


def test_shipped_states_have_no_goemotions_text():
    rows = [json.loads(l) for l in F.STATES.read_text(encoding="utf-8").splitlines() if l.strip()]
    goe = [r for r in rows if r["source"].get("dataset") == F.DATASET]
    assert len(goe) == 44 and all(r["text"] is None and r["source"]["orig_id"] for r in goe)
    audit = [json.loads(l) for l in F.AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(r["state_text"] is None for r in audit if "state_source" in r)


def test_unpinned_tsv_is_refused():
    with pytest.raises(SystemExit):
        F.texts_by_id(b"some text\t1\tabc\n")


def test_restore_strips_and_fills_only_redacted_rows():
    rows = [{"text": None, "source": {"dataset": F.DATASET, "orig_id": "x1"}},
            {"text": "kept", "source": {"dataset": F.DATASET, "orig_id": "x2"}},
            {"text": "other", "source": {"kind": "judged"}}]
    n = F.restore_rows(rows, {"x1": "hello", "x2": "changed"}, "text", lambda r: r.get("source"))
    assert n == 1 and [r["text"] for r in rows] == ["hello", "kept", "other"]
