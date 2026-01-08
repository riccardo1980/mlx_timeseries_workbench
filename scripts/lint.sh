#!/usr/bin/env bash

set -e
set -x


uv run mypy src tests
uv run ruff check src tests --fix
uv run ruff format src tests

# uv run mypy src tests
# uv run ruff check src tests
# uv run ruff format src tests
