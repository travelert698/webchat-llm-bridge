"""Interactive startup choices shared by the DeepSeek and Qwen servers.

The delay is a local cooldown after a completed provider response and after a
client generation response finishes. It is not a provider quota or an evasion
mechanism. Environment variables are supported for defaults and automation.
"""
from __future__ import annotations

import math
import os
import sys

DEFAULT_PROMPT_MODE = "full"
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_MAX_PENDING = 100
MIN_DELAY_SECONDS = 0.0
MAX_DELAY_SECONDS = 300.0
MIN_MAX_PENDING = 1
MAX_MAX_PENDING = 10_000


def _is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def _skip_prompts() -> bool:
    return os.environ.get("BRIDGE_STARTUP_CONFIGURED", "").strip() == "1"


def _read_choice(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return ""


def _environment_prompt_mode() -> str | None:
    env = os.environ.get("PROMPT_MODE", "").strip().lower()
    aliases = {"1": "full", "2": "user_only"}
    if env in ("full", "user_only"):
        return env
    if env in aliases:
        return aliases[env]
    if env:
        print(f"[config] Ignoring invalid PROMPT_MODE={env!r}; choose a mode below.")
    return None


def choose_prompt_mode() -> str:
    """Return ``full`` or ``user_only``; interactive terminals always ask."""
    env_mode = _environment_prompt_mode()
    if not _is_interactive() or _skip_prompts():
        mode = env_mode or DEFAULT_PROMPT_MODE
        if not _is_interactive() and env_mode is None:
            print(f"[config] Non-interactive input; using prompt mode {mode!r}.")
        return mode

    default_mode = env_mode or DEFAULT_PROMPT_MODE
    default_choice = "2" if default_mode == "user_only" else "1"
    print("\n=== Prompt mode selection ===")
    print("  [1] FULL passthrough (system + all messages)")
    print("  [2] USER message only")
    print("  Both modes work with Continue.dev and DeepSeek Harness.")
    while True:
        choice = _read_choice(f"Choose method 1 or 2 [default {default_choice}]: ")
        if choice == "":
            return default_mode
        if choice == "1":
            return "full"
        if choice == "2":
            return "user_only"
        print("Please enter 1 or 2 (or press Enter for the default).")


def _provider_delay_env_names(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        return (
            "DEEPSEEK_DELAY_SECONDS",
            "BRIDGE_DELAY_SECONDS",
            "REQUEST_DELAY_SECONDS",
        )
    if normalized in ("qwen", "qw"):
        return (
            "QWEN_DELAY_SECONDS",
            "BRIDGE_DELAY_SECONDS",
            "REQUEST_DELAY_SECONDS",
        )
    raise ValueError(f"Unknown bridge provider: {provider!r}")


def _format_seconds(seconds: float) -> str:
    return str(int(seconds)) if seconds.is_integer() else f"{seconds:g}"


def _environment_delay(provider: str) -> float | None:
    for name in _provider_delay_env_names(provider):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            seconds = float(value.strip())
            if math.isfinite(seconds) and MIN_DELAY_SECONDS <= seconds <= MAX_DELAY_SECONDS:
                return seconds
        except (TypeError, ValueError):
            pass
        print(
            f"[config] Ignoring invalid {name}={value!r}; expected a number from "
            f"{MIN_DELAY_SECONDS:g} to {MAX_DELAY_SECONDS:g} seconds."
        )
    return None


def choose_delay_seconds(provider: str) -> float:
    """Choose the cooldown in seconds between completed and next requests."""
    env_delay = _environment_delay(provider)
    default_delay = DEFAULT_DELAY_SECONDS if env_delay is None else env_delay
    interactive = _is_interactive() and not _skip_prompts()
    if not interactive:
        if env_delay is None:
            print(
                f"[config] Non-interactive input; using {default_delay:g}s cooldown "
                f"for {provider.title()}."
            )
        return default_delay

    print(f"\n=== {provider.title()} request cooldown ===")
    print("The delay begins after a provider response finishes, before another provider submission.")
    print("A second cooldown also prevents the next client generation from starting too soon.")
    while True:
        value = _read_choice(
            f"Seconds to wait before the next request [{_format_seconds(default_delay)}]: "
        )
        if value == "":
            return default_delay
        try:
            seconds = float(value)
        except ValueError:
            seconds = math.nan
        if math.isfinite(seconds) and MIN_DELAY_SECONDS <= seconds <= MAX_DELAY_SECONDS:
            return seconds
        print(f"Please enter a number from {MIN_DELAY_SECONDS:g} to {MAX_DELAY_SECONDS:g} seconds.")


def _provider_queue_env_names(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        return ("DEEPSEEK_MAX_QUEUE", "BRIDGE_MAX_QUEUE")
    if normalized in ("qwen", "qw"):
        return ("QWEN_MAX_QUEUE", "BRIDGE_MAX_QUEUE")
    raise ValueError(f"Unknown bridge provider: {provider!r}")


def choose_max_pending(provider: str) -> int:
    """Return the bounded count of requests allowed to wait in the FIFO queue."""
    for name in _provider_queue_env_names(provider):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value.strip())
            if MIN_MAX_PENDING <= size <= MAX_MAX_PENDING:
                return size
        except (TypeError, ValueError):
            pass
        print(
            f"[config] Ignoring invalid {name}={value!r}; expected "
            f"{MIN_MAX_PENDING}..{MAX_MAX_PENDING}."
        )
    return DEFAULT_MAX_PENDING


def choose_startup_config(provider: str) -> tuple[str, float, int]:
    """Ask the startup questions in the requested order and return settings."""
    mode = choose_prompt_mode()
    delay_seconds = choose_delay_seconds(provider)
    max_pending = choose_max_pending(provider)
    return mode, delay_seconds, max_pending


def environment_for_child(
    provider: str,
    prompt_mode: str,
    delay_seconds: float,
    max_pending: int,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create child env with the selected per-provider settings made explicit."""
    env = dict(os.environ if base is None else base)
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        delay_name, queue_name = "DEEPSEEK_DELAY_SECONDS", "DEEPSEEK_MAX_QUEUE"
    elif normalized in ("qwen", "qw"):
        delay_name, queue_name = "QWEN_DELAY_SECONDS", "QWEN_MAX_QUEUE"
    else:
        raise ValueError(f"Unknown bridge provider: {provider!r}")
    env["PROMPT_MODE"] = prompt_mode
    env[delay_name] = _format_seconds(float(delay_seconds))
    env[queue_name] = str(max_pending)
    env["BRIDGE_STARTUP_CONFIGURED"] = "1"
    return env
