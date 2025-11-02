.PHONY: auth install run validate tools lint docs

auth:
	@echo "Opening a subshell with .env variables exported."
	@set -a; \
	if [ -f .env ]; then source .env; fi; \
	set +a; \
	exec $$SHELL

install:
	uv sync

run:
	python main.py run --goal "Say hello without tools."

validate:
	python main.py validate

tools:
	python main.py tools

lint:
	ruff check .

docs:
	python scripts/generate_readme.py
