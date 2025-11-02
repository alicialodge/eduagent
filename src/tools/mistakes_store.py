from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field

from .base import Tool


@dataclass
class MistakeRecord:
    topic: str
    detail: str


class MistakeStoreArgs(BaseModel):
    topic: str = Field(..., description="Subject area for the learner mistake.")
    detail: str = Field(..., description="Description of what went wrong.")


_MISTAKE_MEMORY: List[MistakeRecord] = []


def get_mistake_memory() -> List[MistakeRecord]:
    return _MISTAKE_MEMORY


class MistakesStoreTool(Tool):
    name = "mistakes_store"
    description = (
        "Log a repeated learner mistake after the same concept is missed twice. "
        "Provide `topic` with the concept name (e.g. 'present tense nosotros') and "
        "`detail` with a one sentence summary of the misconception."
    )
    args_schema = MistakeStoreArgs

    def run(self, *, topic: str, detail: str) -> str:
        record = MistakeRecord(topic=topic, detail=detail)
        _MISTAKE_MEMORY.append(record)
        return f"Stored mistake for topic '{topic}'."
