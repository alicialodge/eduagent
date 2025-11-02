from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from src.agent.agent import EducationalAgent
from src.tools.base import ToolRegistry
from src.tools.mistakes_search import MistakesSearchTool
from src.tools.mistakes_store import MistakesStoreTool

console = Console()


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
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


def run_agent(goal: str, *, verbose: bool, model: str) -> None:
    ensure_api_key()
    client = OpenAI()
    registry = build_registry()
    agent = EducationalAgent(client=client, registry=registry, model=model, console=console)
    final_answer = agent.run(goal, verbose=verbose)
    console.print(f"[bold green]Final answer:[/bold green] {final_answer}")


def interactive_agent(*, initial_goal: str | None, verbose: bool, model: str) -> None:
    ensure_api_key()
    client = OpenAI()
    registry = build_registry()
    agent = EducationalAgent(client=client, registry=registry, model=model, console=console)
    conversation = agent.start_conversation(verbose=verbose)

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

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    registry = build_registry()

    if args.command == "run":
        run_agent(args.goal, verbose=args.verbose, model=args.model)
    elif args.command == "validate":
        validate_echo_tool(registry)
    elif args.command == "tools":
        render_tools(registry)
    elif args.command == "chat":
        interactive_agent(initial_goal=args.goal, verbose=args.verbose, model=args.model)
    else:  # pragma: no cover - handled by argparse
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
