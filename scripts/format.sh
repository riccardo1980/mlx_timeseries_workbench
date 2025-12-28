#!/usr/bin/env bash
set -x

uv run ruff check src tests --fix
uv run ruff format src tests