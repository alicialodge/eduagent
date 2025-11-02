from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_readme() -> str:
    return dedent(
        """
        # Educational Agent Scaffolding

        A minimal CLI scaffolding for an educational coding tutor agent. The agent routes
        user goals through an OpenAI model and can call registered tools.

        ## Getting Started

        ```bash
        make install
        export OPENAI_API_KEY=sk-...
        python main.py run --goal "I want to learn Para/Por in Spanish"
        python main.py chat
        ```

        ## Repository Layout

        - `main.py` — CLI entry point for running, validating, and listing tools.
        - `src/agent/` — Agent coordination logic and prompt scaffolding.
        - `src/tools/` — Tool base class plus sample tools (echo, mistakes store/search).
        - `scripts/generate_readme.py` — Keeps this README in sync.

        ## Make Targets

        - `make run` — Quick smoke run of the agent (non-interactive).
        - `make validate` — Ensure the Echo tool can be invoked.
        - `make tools` — Display registered tools.
        - `make lint` — Lint the project using Ruff.
        - `make docs` — Rebuild this README.

        ## Next Steps

        - Flesh out persistence for mistakes tooling (Supabase + pgvector).
        - Add user identity and memory retrieval loops.
        - Replace the CLI with a minimal UI once the backend stabilises.
        """
    ).strip() + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme_path = project_root / "README.md"
    readme_path.write_text(render_readme(), encoding="utf-8")
    print(f"Regenerated {readme_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
