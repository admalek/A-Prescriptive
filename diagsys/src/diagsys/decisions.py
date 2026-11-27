"""The supervisor's decisions on the system's recommendations: an append-only log per machine.

The system only advises. A recommendation becomes a decision when a human approves, rejects or
modifies it, and that is written down with who decided, when and why, together with a copy of
what was decided on and the hash of the whole recommendation (so later changes to a
recommendation cannot be mistaken for the version that was approved).
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from .config import Settings
from .contracts import Action, Contract, Recommendation
from .machines import machine_dir

Verdict = Literal["approved", "rejected", "modified"]


class DecisionError(Exception):
    """The decision cannot be recorded; the message says why."""


class Decision(Contract):
    decision_id: str
    machine_id: str
    recommendation_hash: str
    action_index: int = Field(ge=0)
    action: Action  # a copy of what was decided on
    verdict: Verdict
    comment: str = ""
    decided_by: str
    decided_at: datetime
    modified_description: str | None = None  # what was actually decided, for "modified"

    @field_validator("decided_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        return value.astimezone(UTC)


def recommendation_hash(recommendation: Recommendation) -> str:
    canonical = json.dumps(recommendation.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def log_path(settings: Settings, machine_id: str) -> Path:
    return machine_dir(settings, machine_id) / "reports" / "decisions.jsonl"


def record_decision(
    settings: Settings,
    recommendation: Recommendation,
    action_index: int,
    verdict: Verdict,
    *,
    decided_by: str,
    comment: str = "",
    modified_description: str | None = None,
    now: datetime | None = None,
) -> Decision:
    if not 0 <= action_index < len(recommendation.actions):
        raise DecisionError(f"the recommendation has no action number {action_index + 1}")
    if not decided_by.strip():
        raise DecisionError("say who decides (name or initials)")
    if verdict in ("rejected", "modified") and len(comment.strip()) < 3:
        raise DecisionError("a rejection or a change needs a reason in the comment")
    if verdict == "modified" and not (modified_description or "").strip():
        raise DecisionError("describe what is decided instead")
    try:
        decision = Decision(
            decision_id=uuid.uuid4().hex[:12],
            machine_id=recommendation.machine_id,
            recommendation_hash=recommendation_hash(recommendation),
            action_index=action_index,
            action=recommendation.actions[action_index],
            verdict=verdict,
            comment=comment.strip(),
            decided_by=decided_by.strip(),
            decided_at=now or datetime.now(UTC),
            modified_description=(modified_description or "").strip() or None,
        )
    except ValidationError as exc:
        raise DecisionError(str(exc)) from exc
    path = log_path(settings, recommendation.machine_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(decision.model_dump_json() + "\n")
    return decision


def list_decisions(settings: Settings, machine_id: str) -> list[Decision]:
    """All decisions of a machine, oldest first. A damaged line is reported, not skipped."""
    path = log_path(settings, machine_id)
    if not path.is_file():
        return []
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(Decision.model_validate_json(line))
        except ValidationError as exc:
            raise DecisionError(f"{path.name} line {number} is damaged: {exc}") from exc
    return out
