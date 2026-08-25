#!/usr/bin/env python3
"""Launch Streamlit, verify its health endpoint, and shut it down."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8510)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={args.port}",
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/amp-finder-matplotlib")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health_url = f"http://127.0.0.1:{args.port}/_stcore/health"
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Streamlit exited early.\n{output}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    if response.status == 200 and "ok" in body.lower():
                        print(f"Streamlit health check passed at {health_url}")
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        raise TimeoutError(f"Streamlit did not become healthy within {args.timeout} seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
