#!/usr/bin/env bash
# GoLiGoL launcher for Linux / macOS: creates a virtual environment on first run.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m goligol "$@"
