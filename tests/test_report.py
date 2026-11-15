"""Report text helpers."""

from __future__ import annotations

from brainheroes.report import _id_ranges, _near_misses, describe_qc_failures


def test_id_ranges():
    assert _id_ranges([9, 3, 4, 5, 11]) == "3–5, 9, 11"


def test_qc_failures_are_grouped_by_fault_and_session():
    reasons = {
        "3": "bs_short_850nm_saturated;bh_short_850nm_saturated",
        "4": "bs_short_850nm_saturated+low_sci;bh_short_850nm_saturated+low_sci",
        "35": "bs_short_760nm_dropout",
        "40": "bh_short_low_sci",
    }
    assert describe_qc_failures(reasons) == (
        "u ID 3–4 nasycenie toru `850 nm` w obu sesjach, u ID 35 zanik toru `760 nm` w `BS`, u ID 40 niskie SCI w `BH`"
    )


def test_near_misses():
    block = {"positives": [{"id": 1, "alarm_pct": 95.0}, {"id": 2, "alarm_pct": 87.2}, {"id": 3, "alarm_pct": 50.0}]}
    assert (
        _near_misses(block, 90, 85)
        == "ID 2 jest tuż pod progiem 90 (87,2 — przekracza próg 85), a ID 3 nie przekracza progów 90/85"
    )
