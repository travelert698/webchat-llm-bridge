"""
run_both.py – Start BOTH bridges from one project folder.

  DeepSeek -> http://127.0.0.1:8000/v1   (server.py,  model: deepseek-chat)
  Qwen     -> http://127.0.0.1:8001/v1   (qwen_server.py, model: qwen-auto)

Each server runs in its own Python process / console window.
Press Ctrl+C here to stop both.

Note: browser profiles are separate (./browser_data for DeepSeek,
./qwen_browser_data for Qwen), so both can run at the same time.
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))


def start_server(name: str, script: str, port_env: str, port: int):
    env = dict(os.environ)
    env[port_env] = str(port)
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=BASE,
        env=env,
        creationflags=flags,
    )
    print(f"[run_both] started {name}: {script} on port {port} (pid {proc.pid})")
    return proc


def main():
    print("Starting both bridges...")
    procs = [
        start_server("DeepSeek", "server.py", "DEEPSEEK_PORT", 8000),
        start_server("Qwen", "qwen_server.py", "QWEN_PORT", 8001),
    ]
    print("\nBoth bridges are starting. Log in to each browser when it opens,")
    print("then point Continue at:")
    print("  DeepSeek: http://127.0.0.1:8000/v1  (model deepseek-chat)")
    print("  Qwen:     http://127.0.0.1:8001/v1  (model qwen-auto)")
    print("Press Ctrl+C here to stop both.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run_both] shutting down both servers...")
        for name, proc in zip(("DeepSeek", "Qwen"), procs):
            proc.terminate()
        for name, proc in zip(("DeepSeek", "Qwen"), procs):
            try:
                proc.wait(timeout=5)
                print(f"[run_both] {name} stopped.")
            except Exception:
                proc.kill()
                print(f"[run_both] {name} killed.")


if __name__ == "__main__":
    main()