from datetime import UTC, datetime

import pytest

from diagsys.contracts import Action, Recommendation, Source
from diagsys.decisions import (
    DecisionError,
    list_decisions,
    log_path,
    recommendation_hash,
    record_decision,
)
from diagsys.machines import create_machine


def rec(text="Serwis za 40 h"):
    return Recommendation(
        machine_id="pompa-01",
        actions=[Action(description=text), Action(description="Praca do awarii", risk="high")],
        sources=[Source(kind="tool", ref="prescribe:risk")],
    )


def test_decisions_are_appended_with_who_when_and_a_copy_of_the_action(settings):
    create_machine(settings, "pompa-01", "Pompa")
    r = rec()
    when = datetime(2026, 2, 1, 8, 0, tzinfo=UTC)
    a = record_decision(settings, r, 0, "approved", decided_by="AM", now=when)
    b = record_decision(
        settings, r, 1, "rejected", decided_by="AM", comment="Zbyt ryzykowne", now=when
    )
    out = list_decisions(settings, "pompa-01")
    assert [d.decision_id for d in out] == [a.decision_id, b.decision_id]
    assert out[0].action.description == "Serwis za 40 h" and out[0].verdict == "approved"
    assert out[0].recommendation_hash == recommendation_hash(r)
    assert out[1].comment == "Zbyt ryzykowne" and out[1].decided_at == when


def test_the_hash_changes_when_the_recommendation_changes():
    assert recommendation_hash(rec()) == recommendation_hash(rec())
    assert recommendation_hash(rec()) != recommendation_hash(rec("Serwis za 41 h"))


def test_invalid_decisions_are_refused_and_nothing_is_written(settings):
    create_machine(settings, "pompa-01", "Pompa")
    r = rec()
    with pytest.raises(DecisionError, match="no action number 3"):
        record_decision(settings, r, 2, "approved", decided_by="AM")
    with pytest.raises(DecisionError, match="who decides"):
        record_decision(settings, r, 0, "approved", decided_by=" ")
    with pytest.raises(DecisionError, match="reason"):
        record_decision(settings, r, 0, "rejected", decided_by="AM")
    with pytest.raises(DecisionError, match="instead"):
        record_decision(settings, r, 0, "modified", decided_by="AM", comment="inny termin")
    assert not log_path(settings, "pompa-01").exists()
    ok = record_decision(
        settings, r, 0, "modified", decided_by="AM", comment="postój w piątek",
        modified_description="Serwis w piątek o 6:00",
    )  # fmt: skip
    assert ok.modified_description == "Serwis w piątek o 6:00"


def test_a_damaged_log_line_is_reported(settings):
    create_machine(settings, "pompa-01", "Pompa")
    record_decision(settings, rec(), 0, "approved", decided_by="AM")
    with log_path(settings, "pompa-01").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    with pytest.raises(DecisionError, match="line 2 is damaged"):
        list_decisions(settings, "pompa-01")
