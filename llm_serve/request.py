from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_EOS = "finished_eos"
    FINISHED_LENGTH = "finished_length"


@dataclass
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_token_id: Optional[int] = None


@dataclass
class SequenceRequest:
    """Tracks token state, generation progress, and lifecycle status for a single user request."""

    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_step: int = 0

    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    first_token_step: Optional[int] = None
    finish_step: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least 1 token")

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_new_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (
            SequenceStatus.FINISHED_EOS,
            SequenceStatus.FINISHED_LENGTH,
        )

    def append_output_token(self, token_id: int, current_step: int) -> None:
        if self.first_token_step is None:
            self.first_token_step = current_step
        self.output_token_ids.append(token_id)

        if (
            self.sampling_params.eos_token_id is not None
            and token_id == self.sampling_params.eos_token_id
        ):
            self.status = SequenceStatus.FINISHED_EOS
            self.finish_step = current_step
        elif len(self.output_token_ids) >= self.sampling_params.max_new_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH
            self.finish_step = current_step
