from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.console import Console

from src.tools.base import ToolInvocationError, ToolRegistry

from .prompt import SYSTEM_PROMPT, USER_WRAPPER


def _scrub_payload_for_debug(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of the outgoing request."""

    def scrub_message(message: Dict[str, Any]) -> Dict[str, Any]:
        scrubbed: Dict[str, Any] = {"role": message.get("role")}
        if "name" in message:
            scrubbed["name"] = message["name"]
        if "content" in message:
            scrubbed["content"] = message["content"]
        if "tool_calls" in message:
            scrubbed["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            scrubbed["tool_call_id"] = message["tool_call_id"]
        return scrubbed

    return {
        "model": payload.get("model"),
        "tool_choice": payload.get("tool_choice"),
        "tools": payload.get("tools"),
        "messages": [scrub_message(message) for message in payload.get("messages", [])],
    }


class AgentLoopError(RuntimeError):
    """Raised when the agent exceeds its allowed number of tool loops."""


class AgentConversation:
    """Stateful wrapper for an interactive conversation with the agent."""

    def __init__(self, agent: "EducationalAgent", *, verbose: bool) -> None:
        self._agent = agent
        self._verbose = verbose
        self._messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._has_primary_goal = False
        if verbose:
            tool_definitions = self._agent.registry.as_openai_tools()
            self._agent._console.print("[yellow]Tool definitions provided to the model:[/yellow]")
            self._agent._console.print_json(data={"tools": tool_definitions})

    def ask(self, user_input: str, *, max_turns: int = 6) -> str:
        if not self._has_primary_goal:
            content = USER_WRAPPER.format(goal=user_input)
            self._has_primary_goal = True
        else:
            content = user_input

        self._messages.append({"role": "user", "content": content})
        return self._agent._run_loop(self._messages, max_turns=max_turns, verbose=self._verbose)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages


class EducationalAgent:
    """Simple educational agent that can reason with tools via the OpenAI API."""

    def __init__(
        self,
        client: OpenAI,
        registry: ToolRegistry,
        *,
        model: str = "gpt-4o-mini",
        console: Optional[Console] = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._model = model
        self._console = console or Console()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def run(self, goal: str, *, max_turns: int = 15, verbose: bool = False) -> str:
        """Run the agent loop until a final answer is produced."""
        conversation = self.start_conversation(verbose=verbose)
        return conversation.ask(goal, max_turns=max_turns)

    def start_conversation(self, *, verbose: bool = False) -> AgentConversation:
        """Create a stateful conversation for interactive CLI usage."""
        return AgentConversation(self, verbose=verbose)

    def _run_loop(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_turns: int,
        verbose: bool,
    ) -> str:
        tool_definitions = self._registry.as_openai_tools()
        for step in range(max_turns):
            request_payload = {
                "model": self._model,
                "messages": messages,
                "tools": tool_definitions,
                "tool_choice": "auto",
            }
            if verbose:
                self._console.print(
                    f"[yellow]--- Request to OpenAI (step {step + 1}) ---[/yellow]"
                )
                self._console.print_json(data=_scrub_payload_for_debug(request_payload))

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
            )
            choice = response.choices[0]
            assistant_message = choice.message

            tool_calls_payload: Optional[List[Dict[str, Any]]] = None
            if assistant_message.tool_calls:
                tool_calls_payload = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in assistant_message.tool_calls
                ]

            if verbose:
                response_snapshot: Dict[str, Any] = {"role": assistant_message.role}
                if assistant_message.content is not None:
                    response_snapshot["content"] = assistant_message.content
                if tool_calls_payload:
                    response_snapshot["tool_calls"] = tool_calls_payload

                self._console.print("[yellow]--- Response from OpenAI ---[/yellow]")
                self._console.print_json(data=response_snapshot)

            messages.append(
                {
                    "role": assistant_message.role,
                    "content": assistant_message.content or "",
                    **({"tool_calls": tool_calls_payload} if tool_calls_payload else {}),
                }
            )

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    result = self._handle_tool_call(
                        tool_call.id,
                        tool_call.function,
                        verbose=verbose,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                continue

            final_answer = (assistant_message.content or "").strip()
            return final_answer

        raise AgentLoopError(
            "Agent exceeded maximum number of turns without producing a final answer."
        )

    def _handle_tool_call(
        self,
        call_id: str,
        function: Any,
        *,
        verbose: bool = False,
    ) -> str:
        if verbose:
            self._console.print(
                f"[blue]Calling tool[/blue] {function.name} with args {function.arguments}"
            )

        arguments = json.loads(function.arguments or "{}")
        try:
            result = self._registry.invoke(function.name, arguments)
        except ToolInvocationError as exc:
            error_message = f"Tool {function.name} failed: {exc}"
            if verbose:
                self._console.print(f"[red]{error_message}[/red]")
            result = error_message

        if verbose:
            self._console.print(f"[magenta]Tool result[/magenta] ({call_id}): {result}")
        return result

    def list_tools(self) -> List[Dict[str, str]]:
        """Return tool metadata for CLI rendering."""
        return self._registry.list_tools()
