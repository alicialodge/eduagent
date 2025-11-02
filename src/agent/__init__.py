"""Agent package exposing the educational agent implementation."""

from .agent import EducationalAgent, AgentLoopError
from .prompt import SYSTEM_PROMPT, USER_WRAPPER

__all__ = [
    "EducationalAgent",
    "AgentLoopError",
    "SYSTEM_PROMPT",
    "USER_WRAPPER",
]
