from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
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

        self._messages.append(
            {"role": "user", "content": [{"type": "text", "text": content}]}
        )
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
    """Simple educational agent that can reason with tools via the Anthropic Claude API."""

    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        client: Anthropic,
        registry: ToolRegistry,
        *,
        model: str,
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
        tool_definitions = self._registry.as_anthropic_tools()
        for _ in range(max_turns):
            system_prompt, request_messages = self._prepare_anthropic_messages(messages)
            request_kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": request_messages,
                "max_tokens": self.DEFAULT_MAX_TOKENS,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if tool_definitions:
                request_kwargs["tools"] = tool_definitions

            response = self._client.messages.create(**request_kwargs)

            assistant_blocks, tool_uses = self._parse_assistant_content(response.content)
            messages.append({"role": "assistant", "content": assistant_blocks})

            assistant_text = "\n".join(
                block["text"] for block in assistant_blocks if block["type"] == "text"
            )
            if transcript and assistant_text:
                transcript.log_agent(assistant_text)

            if tool_uses:
                tool_result_blocks = []
                for tool_use in tool_uses:
                    result = self._handle_tool_use(
                        tool_use,
                        verbose=verbose,
                        transcript=transcript,
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            final_answer = assistant_text.strip()
            return final_answer

        raise AgentLoopError(
            "Agent exceeded maximum number of turns without producing a final answer."
        )

    def _prepare_anthropic_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str | None, List[Dict[str, Any]]]:
        system_prompt: str | None = None
        request_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system_prompt = str(content)
                continue

            normalized_blocks = self._normalize_content_blocks(content)
            request_messages.append({"role": role, "content": normalized_blocks})

        return system_prompt, request_messages

    def _normalize_content_blocks(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            normalized: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    normalized.append(block)
                    continue
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    normalized.append({"type": "text", "text": getattr(block, "text", "")})
                elif block_type == "tool_use":
                    normalized.append(
                        {
                            "type": "tool_use",
                            "id": getattr(block, "id"),
                            "name": getattr(block, "name"),
                            "input": getattr(block, "input", {}),
                        }
                    )
                elif block_type == "tool_result":
                    normalized.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "tool_use_id"),
                            "content": getattr(block, "content", ""),
                        }
                    )
            if normalized:
                return normalized

        return [{"type": "text", "text": str(content)}]

    def _parse_assistant_content(
        self, content_blocks: List[Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        assistant_blocks: List[Dict[str, Any]] = []
        tool_uses: List[Dict[str, Any]] = []

        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_block = {"type": "text", "text": getattr(block, "text", "")}
                assistant_blocks.append(text_block)
            elif block_type == "tool_use":
                tool_block = {
                    "type": "tool_use",
                    "id": getattr(block, "id"),
                    "name": getattr(block, "name"),
                    "input": getattr(block, "input", {}) or {},
                }
                assistant_blocks.append(tool_block)
                tool_uses.append(tool_block)
            else:
                fallback_text = getattr(block, "text", None)
                if fallback_text is None and hasattr(block, "to_dict"):
                    fallback_text = json.dumps(block.to_dict())
                if fallback_text is None:
                    fallback_text = str(block)
                assistant_blocks.append({"type": "text", "text": str(fallback_text)})

        return assistant_blocks, tool_uses

    def _handle_tool_use(
        self,
        tool_use: Dict[str, Any],
        *,
        verbose: bool = False,
        transcript: TranscriptWriter | None = None,
    ) -> str:
        call_id = tool_use["id"]
        tool_name = tool_use["name"]
        arguments = tool_use.get("input", {}) or {}

        serialized_args = json.dumps(arguments, separators=(",", ":"))
        if verbose:
            self._console.print(f"[blue]Tool call[/blue] {tool_name}: {serialized_args}")
        if transcript:
            transcript.log_tool_call(tool_name, serialized_args)

        try:
            result = self._registry.invoke(tool_name, arguments)
        except ToolInvocationError as exc:
            error_message = f"Tool {tool_name} failed: {exc}"
            if verbose:
                self._console.print(f"[red]{error_message}[/red]")
            result = error_message

        if verbose:
            self._console.print(f"[magenta]Tool result[/magenta] {call_id}: {result}")
        if transcript:
            transcript.log_tool_result(call_id, str(result))
        return result

    def list_tools(self) -> List[Dict[str, str]]:
        """Return tool metadata for CLI rendering."""
        return self._registry.list_tools()
