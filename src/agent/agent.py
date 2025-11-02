from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.console import Console

from src.tools.base import ToolInvocationError, ToolRegistry
from src.utils.transcript import TranscriptWriter

from .prompt import SYSTEM_PROMPT, USER_WRAPPER


def _truncate(value: str, limit: int = 100) -> str:
    """Collapse whitespace and trim strings to a compact preview."""
    collapsed = " ".join(str(value).split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


class AgentLoopError(RuntimeError):
    """Raised when the agent exceeds its allowed number of tool loops."""


class AgentConversation:
    """Stateful wrapper for an interactive conversation with the agent."""

    def __init__(
        self,
        agent: "EducationalAgent",
        *,
        verbose: bool,
        transcript: TranscriptWriter | None = None,
    ) -> None:
        self._agent = agent
        self._verbose = verbose
        self._transcript = transcript
        self._messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._has_primary_goal = False
        if verbose:
            system_preview = _truncate(SYSTEM_PROMPT)
            self._agent._console.print(f"[yellow]System prompt:[/yellow] {system_preview}")
            user_wrapper_preview = _truncate(USER_WRAPPER)
            self._agent._console.print(f"[yellow]User wrapper:[/yellow] {user_wrapper_preview}")

            tool_summaries = self._agent.registry.list_tools()
            if tool_summaries:
                self._agent._console.print("[yellow]Tools:[/yellow]")
                for tool in tool_summaries:
                    description = _truncate(tool.get("description", ""), limit=80)
                    self._agent._console.print(f"  - {tool['name']}: {description}")
            else:
                self._agent._console.print("[yellow]Tools:[/yellow] none registered")

    def ask(self, user_input: str, *, max_turns: int = 6) -> str:
        if not self._has_primary_goal:
            content = USER_WRAPPER.format(goal=user_input)
            self._has_primary_goal = True
        else:
            content = user_input

        if self._transcript:
            self._transcript.log_user(user_input)

        self._messages.append({"role": "user", "content": content})
        return self._agent._run_loop(
            self._messages,
            max_turns=max_turns,
            verbose=self._verbose,
            transcript=self._transcript,
        )

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

    def run(
        self,
        goal: str,
        *,
        max_turns: int = 15,
        verbose: bool = False,
        transcript: TranscriptWriter | None = None,
    ) -> str:
        """Run the agent loop until a final answer is produced."""
        conversation = self.start_conversation(verbose=verbose, transcript=transcript)
        return conversation.ask(goal, max_turns=max_turns)

    def start_conversation(
        self,
        *,
        verbose: bool = False,
        transcript: TranscriptWriter | None = None,
    ) -> AgentConversation:
        """Create a stateful conversation for interactive CLI usage."""
        return AgentConversation(self, verbose=verbose, transcript=transcript)

    def _run_loop(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_turns: int,
        verbose: bool,
        transcript: TranscriptWriter | None,
    ) -> str:
        tool_definitions = self._registry.as_openai_tools()
        for _ in range(max_turns):
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

            messages.append(
                {
                    "role": assistant_message.role,
                    "content": assistant_message.content or "",
                    **({"tool_calls": tool_calls_payload} if tool_calls_payload else {}),
                }
            )

            if transcript and assistant_message.content:
                transcript.log_agent(assistant_message.content)

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    if transcript:
                        transcript.log_tool_call(
                            tool_call.function.name,
                            tool_call.function.arguments or "",
                        )
                    result = self._handle_tool_call(
                        tool_call.id,
                        tool_call.function,
                        verbose=verbose,
                        transcript=transcript,
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
        transcript: TranscriptWriter | None = None,
    ) -> str:
        raw_arguments = function.arguments or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            if verbose:
                args_preview = _truncate(raw_arguments)
                self._console.print(f"[blue]Tool call[/blue] {function.name}: {args_preview}")
            raise
        else:
            if verbose:
                args_preview = _truncate(json.dumps(arguments, separators=(",", ":")))
                self._console.print(f"[blue]Tool call[/blue] {function.name}: {args_preview}")

        try:
            result = self._registry.invoke(function.name, arguments)
        except ToolInvocationError as exc:
            error_message = f"Tool {function.name} failed: {exc}"
            if verbose:
                self._console.print(f"[red]{error_message}[/red]")
            result = error_message

        if verbose:
            result_preview = _truncate(result)
            self._console.print(f"[magenta]Tool result[/magenta] {call_id}: {result_preview}")
        if transcript:
            transcript.log_tool_result(call_id, str(result))
        return result

    def list_tools(self) -> List[Dict[str, str]]:
        """Return tool metadata for CLI rendering."""
        return self._registry.list_tools()
