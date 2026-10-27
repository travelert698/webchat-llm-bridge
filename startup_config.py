"""Interactive startup choices shared by the DeepSeek and Qwen servers.

The chosen RPM is a *local outbound provider-submission pace*, not a promise
about either provider's unpublished web-chat limits. Environment variables are
supported for defaults, non-interactive runs, and supervised child processes.
"""
from __future__ import annotations

import os
import sys

DEFAULT_PROMPT_MODE = "full"
DEFAULT_RPM = 5
DEFAULT_MAX_PENDING = 100
MIN_RPM = 1
MAX_RPM = 600
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


def _provider_env_names(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        return ("DEEPSEEK_RPM", "BRIDGE_RPM", "RATE_LIMIT_RPM")
    if normalized in ("qwen", "qw"):
        return ("QWEN_RPM", "BRIDGE_RPM", "RATE_LIMIT_RPM")
    raise ValueError(f"Unknown bridge provider: {provider!r}")


def _environment_rpm(provider: str) -> int | None:
    for name in _provider_env_names(provider):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            rpm = int(value.strip())
            if MIN_RPM <= rpm <= MAX_RPM:
                return rpm
        except (TypeError, ValueError):
            pass
        print(f"[config] Ignoring invalid {name}={value!r}; expected {MIN_RPM}..{MAX_RPM}.")
    return None


def choose_rpm(provider: str) -> int:
    """Select a conservative positive requests-per-minute value.

    The limit counts each outbound browser-backed model submission, including
    a tool-call repair attempt, rather than merely counting inbound HTTP calls.
    """
    env_rpm = _environment_rpm(provider)
    default_rpm = env_rpm or DEFAULT_RPM
    interactive = _is_interactive() and not _skip_prompts()
    if not interactive:
        if env_rpm is None:
            print(
                f"[config] Non-interactive input; using conservative local pace "
                f"of {default_rpm} provider requests/minute for {provider.title()}."
            )
        return default_rpm

    print(f"\n=== {provider.title()} provider request pacing ===")
    print("This is a local cap on model submissions, including tool-repair attempts.")
    print("It does not reveal or override the provider's own web-chat limits.")
    while True:
        value = _read_choice(
            f"Maximum provider requests per minute [{default_rpm}]: "
        )
        if value == "":
            return default_rpm
        try:
            rpm = int(value)
        except ValueError:
            rpm = 0
        if MIN_RPM <= rpm <= MAX_RPM:
            return rpm
        print(f"Please enter a whole number from {MIN_RPM} to {MAX_RPM}.")


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


def choose_startup_config(provider: str) -> tuple[str, int, int]:
    """Ask the startup questions in the requested order and return settings."""
    mode = choose_prompt_mode()
    rpm = choose_rpm(provider)
    max_pending = choose_max_pending(provider)
    return mode, rpm, max_pending


def environment_for_child(
    provider: str,
    prompt_mode: str,
    rpm: int,
    max_pending: int,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create child env with the selected per-provider settings made explicit."""
    env = dict(os.environ if base is None else base)
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        rpm_name, queue_name = "DEEPSEEK_RPM", "DEEPSEEK_MAX_QUEUE"
    elif normalized in ("qwen", "qw"):
        rpm_name, queue_name = "QWEN_RPM", "QWEN_MAX_QUEUE"
    else:
        raise ValueError(f"Unknown bridge provider: {provider!r}")
    env["PROMPT_MODE"] = prompt_mode
    env[rpm_name] = str(rpm)
    env[queue_name] = str(max_pending)
    env["BRIDGE_STARTUP_CONFIGURED"] = "1"
    return env
