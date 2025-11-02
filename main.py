from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, Iterable

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from src.agent.agent import EducationalAgent
from src.tools.base import ToolRegistry
from src.tools.mistakes_search import MistakesSearchTool
from src.tools.mistakes_store import MistakesStoreTool
from src.tools.user_name import UserNameTool
from src.utils.transcript import TranscriptWriter

console = Console()


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            UserNameTool(),
            MistakesStoreTool(),
            MistakesSearchTool(),
        ]
    )


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]OPENAI_API_KEY is not set. Export it before running the agent.[/red]"
        )
        sys.exit(1)


def render_tools(registry: ToolRegistry) -> None:
    table = Table(title="Registered Tools")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    for tool in registry.list_tools():
        table.add_row(tool["name"], tool["description"])
    console.print(table)


def validate_echo_tool(registry: ToolRegistry) -> None:
    payload = {"text": "tool called correctly"}
    console.print(f"Tool call: echo -> {payload}")
    result = registry.invoke("echo", payload)
    console.print(result)


def _get_commit_id() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _maybe_build_transcript(
    command: str,
    *,
    goal: str | None,
    model: str,
    verbose: bool,
    save_transcript: bool,
    extra: Dict[str, Any] | None = None,
) -> TranscriptWriter | None:
    if not save_transcript:
        return None

    metadata: Dict[str, Any] = {
        "command": command,
        "model": model,
        "flags": {"verbose": verbose, "save_transcript": save_transcript},
    }
    if goal is not None:
        metadata["goal"] = goal
    if extra:
        metadata.update({key: value for key, value in extra.items() if value is not None})

    commit_id = _get_commit_id()
    if commit_id:
        metadata["commit"] = commit_id

    return TranscriptWriter(metadata=metadata)


def run_agent(goal: str, *, verbose: bool, model: str, save_transcript: bool) -> None:
    ensure_api_key()
    client = OpenAI()
    registry = build_registry()
    transcript = _maybe_build_transcript(
        "run",
        goal=goal,
        model=model,
        verbose=verbose,
        save_transcript=save_transcript,
    )
    agent = EducationalAgent(client=client, registry=registry, model=model, console=console)
    final_answer = agent.run(goal, verbose=verbose, transcript=transcript)
    console.print(f"[bold green]Final answer:[/bold green] {final_answer}")


def interactive_agent(
    *,
    initial_goal: str | None,
    verbose: bool,
    model: str,
    save_transcript: bool,
) -> None:
    ensure_api_key()
    client = OpenAI()
    registry = build_registry()
    transcript = _maybe_build_transcript(
        "chat",
        goal=initial_goal if initial_goal is not None else "(none provided)",
        model=model,
        verbose=verbose,
        save_transcript=save_transcript,
        extra={"mode": "interactive"},
    )
    agent = EducationalAgent(client=client, registry=registry, model=model, console=console)
    conversation = agent.start_conversation(verbose=verbose, transcript=transcript)

    console.print("[bold cyan]Interactive session started. Press Ctrl+C or enter nothing to exit.[/bold cyan]")
    if initial_goal:
        answer = conversation.ask(initial_goal)
        console.print(f"[bold green]Agent:[/bold green] {answer}")

    try:
        while True:
            user_input = console.input("[bold blue]You> [/bold blue]").strip()
            if not user_input:
                console.print("[gray]Ending session.[/gray]")
                break
            answer = conversation.ask(user_input)
            console.print(f"[bold green]Agent:[/bold green] {answer}")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[gray]Session interrupted by user.[/gray]")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational agent CLI entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent with a goal.")
    run_parser.add_argument("--goal", required=True, help="User goal for the agent.")
    run_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for the agent (defaults to gpt-4o-mini).",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show intermediate tool calls and responses.",
    )
    run_parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Persist the conversation transcript to disk.",
    )

    subparsers.add_parser("validate", help="Validate that the Echo tool is callable.")
    subparsers.add_parser("tools", help="List registered tools.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive session with the agent.")
    chat_parser.add_argument(
        "--goal",
        help="Optional initial goal to kick off the conversation.",
    )
    chat_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for the agent (defaults to gpt-4o-mini).",
    )
    chat_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show intermediate tool calls and responses.",
    )
    chat_parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Persist the conversation transcript to disk.",
    )

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    registry = build_registry()

    if args.command == "run":
        run_agent(
            args.goal,
            verbose=args.verbose,
            model=args.model,
            save_transcript=args.save_transcript,
        )
    elif args.command == "validate":
        validate_echo_tool(registry)
    elif args.command == "tools":
        render_tools(registry)
    elif args.command == "chat":
        interactive_agent(
            initial_goal=args.goal,
            verbose=args.verbose,
            model=args.model,
            save_transcript=args.save_transcript,
        )
    else:  # pragma: no cover - handled by argparse
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
