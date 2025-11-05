from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from .base import Tool


class _BasicUserInfoArgs(BaseModel):
    """No arguments are required for retrieving learner profile information."""


class BasicUserInfoTool(Tool):
    name = "basic_user_info"
    description = (
        "Retrieve stored learner details to personalise the session. Returns the user's "
        "preferred name, location, interests, preferred conversation style, and the time "
        "they have already spent learning the language."
    )
    args_schema = _BasicUserInfoArgs

    def run(self) -> Dict[str, Any]:
        interests: List[str] = ["Knitting"]
        return {
            "name": "Alicia",
            "location": "London, United Kingdom",
            "interests": interests,
            "preferred_conversation_style": "Warm, encouraging, and slightly informal.",
            "time_spent_learning_hours": 360,
        }
