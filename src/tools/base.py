from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel, ValidationError


class ToolRegistryError(Exception):
    """Base error for tool registry operations."""


class ToolInvocationError(ToolRegistryError):
    """Raised when a tool cannot be invoked or validation fails."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when the requested tool is not registered."""


class Tool(ABC):
    """Abstract base class describing a callable tool."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema.model_json_schema(),
        }

    def invoke(self, arguments: Dict[str, Any]) -> str:
        try:
            parsed = self.args_schema.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInvocationError(str(exc)) from exc

        result = self.run(**parsed.model_dump())

        if isinstance(result, str):
            return result

        return json.dumps(result)

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:  # pragma: no cover - implemented in subclasses
        """Execute the underlying tool logic."""


class ToolRegistry:
    """Registry for keeping track of available tools."""

    def __init__(self, tools: Optional[Sequence[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "description": tool.description}
            for name, tool in self._tools.items()
        ]

    def as_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self.get(name)
        return tool.invoke(arguments)
