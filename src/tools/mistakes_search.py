from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .base import Tool
from .mistakes_store import MistakeRecord, get_mistake_memory


class MistakeSearchArgs(BaseModel):
    topic: Optional[str] = Field(
        default=None,
        description="Optional topic filter. Return all mistakes if omitted.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of mistakes to return.",
    )


class MistakesSearchTool(Tool):
    name = "mistakes_search"
    description = "Search for similar concepts the user has made mistakes with in the past."
    args_schema = MistakeSearchArgs

    def run(self, *, topic: Optional[str], limit: int) -> str:
        candidates: List[MistakeRecord] = get_mistake_memory()
        if topic:
            matches = [m for m in candidates if m.topic.lower() == topic.lower()]
        else:
            matches = list(candidates)

        summary_lines = []
        for record in matches[:limit]:
            summary_lines.append(f"- {record.topic}: {record.detail}")

        if not summary_lines:
            return "No mistakes found."

        return "\n".join(summary_lines)
