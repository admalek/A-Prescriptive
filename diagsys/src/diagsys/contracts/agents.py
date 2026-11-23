"""Interfaces I3 and I5: tasks and results exchanged with agents, and chat messages for the LLM."""

from typing import Literal

from pydantic import Field, model_validator

from .common import Contract, ErrorInfo, Source, Status


class ChatMessage(Contract):
    role: Literal["system", "user", "assistant"]
    content: str


class TaskLimits(Contract):
    max_iterations: int = Field(default=5, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)


class AgentTask(Contract):
    task_id: str
    step: str
    machine_id: str
    input_refs: list[str] = Field(default_factory=list)
    limits: TaskLimits = Field(default_factory=TaskLimits)


class AgentResult(Contract):
    task_id: str
    status: Status
    output_refs: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    trace_id: str
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _check_error(self) -> "AgentResult":
        if self.status is Status.FAILED and self.error is None:
            raise ValueError("a failed result must carry an error")
        if self.status is Status.OK and self.error is not None:
            raise ValueError("an ok result must not carry an error")
        return self
