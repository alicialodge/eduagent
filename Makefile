SHELL := /bin/bash

.PHONY: auth

auth:
	@echo "Opening a subshell with .env variables exported."
	@set -a; \
	if [ -f .env ]; then source .env; fi; \
	set +a; \
	exec $$SHELL
