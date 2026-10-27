"""Start both browser-backed bridge APIs from one console.

DeepSeek listens on 127.0.0.1:8000 and Qwen on 127.0.0.1:8001. The
supervisor collects the shared prompt mode and each backend's own RPM choice,
then passes the settings to child processes so prompts never race on stdin.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from startup_config import (
    choose_max_pending,
    choose_prompt_mode,
    choose_rpm,
    environment_for_child,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def start_server(name: str, script: str, port_env: str, port: int, env: dict[str, str]):
    child_env = dict(env)
    child_env[port_env] = str(port)
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=BASE,
        env=child_env,
        creationflags=flags,
    )
    print(f"[run_both] started {name}: {script} on port {port} (pid {proc.pid})")
    return proc


def _stop_processes(names: list[str], procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for name, proc in zip(names, procs):
        try:
            proc.wait(timeout=5)
            print(f"[run_both] {name} stopped.")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            print(f"[run_both] {name} killed after shutdown timeout.")


def main() -> None:
    prompt_mode = choose_prompt_mode()
    deepseek_rpm = choose_rpm("deepseek")
    qwen_rpm = choose_rpm("qwen")
    deepseek_pending = choose_max_pending("deepseek")
    qwen_pending = choose_max_pending("qwen")

    deepseek_env = environment_for_child(
        "deepseek", prompt_mode, deepseek_rpm, deepseek_pending
    )
    qwen_env = environment_for_child("qwen", prompt_mode, qwen_rpm, qwen_pending)

    print("\nStarting both bridges with independent provider queues:")
    print(f"  Prompt mode: {prompt_mode}")
    print(f"  DeepSeek: {deepseek_rpm} provider requests/min, {deepseek_pending} pending slots")
    print(f"  Qwen:     {qwen_rpm} provider requests/min, {qwen_pending} pending slots")

    names: list[str] = []
    procs: list[subprocess.Popen] = []
    try:
        deepseek = start_server(
            "DeepSeek", "server.py", "DEEPSEEK_PORT", 8000, deepseek_env
        )
        names.append("DeepSeek")
        procs.append(deepseek)
        qwen = start_server("Qwen", "qwen_server.py", "QWEN_PORT", 8001, qwen_env)
        names.append("Qwen")
        procs.append(qwen)

        print("\nOpenAI-compatible clients:")
        print("  DeepSeek: http://127.0.0.1:8000/v1  (model deepseek-chat)")
        print("  Qwen:     http://127.0.0.1:8001/v1  (model qwen-auto)")
        print("Claude Code / Anthropic Messages base URL (choose one backend per session):")
        print("  DeepSeek: http://127.0.0.1:8000")
        print("  Qwen:     http://127.0.0.1:8001")
        print("Queue stats: GET /queue/status on either server.")
        print("Both services remain loopback-only. Press Ctrl+C here to stop both.\n")

        while True:
            exited = [(name, proc.returncode) for name, proc in zip(names, procs)
                      if proc.poll() is not None]
            if exited:
                for name, code in exited:
                    print(f"[run_both] {name} exited with code {code}; stopping the other bridge.")
                _stop_processes(names, procs)
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run_both] shutting down both servers...")
        _stop_processes(names, procs)
    except Exception:
        _stop_processes(names, procs)
        raise


if __name__ == "__main__":
    main()
