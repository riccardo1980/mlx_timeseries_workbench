#!/usr/bin/env bash
set -x

uv run coverage run -m pytest
uv run coverage report