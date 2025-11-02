from __future__ import annotations

from pydantic import BaseModel

from .base import Tool


class _UserNameArgs(BaseModel):
    """No arguments are required for retrieving the user name."""


class UserNameTool(Tool):
    name = "user_name"
    description = "Retrieve the learner's preferred name for the conversation."
    args_schema = _UserNameArgs

    def run(self) -> str:
        return "Alicia"
