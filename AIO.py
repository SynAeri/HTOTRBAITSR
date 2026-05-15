# Launcher: starts all three standalone Gradio apps as subprocesses and opens each in a new browser tab
import sys
import os
import time
import subprocess
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))

APPS = {
    "Attack Demo":        ("demo",               7860),
    "Experiment Results": ("experiment_results", 7861),
    "Apply Defence":      ("apply_defence",       7862),
}

PROCS = []

if __name__ == "__main__":
    print("Starting standalone apps...")
    for script, port in APPS.values():
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "standalone", f"{script}.py")],
            env={**os.environ, "GRADIO_SERVER_PORT": str(port), "GRADIO_NO_BROWSER": "1"},
            cwd=ROOT,
        )
        PROCS.append(proc)

    print("Apps launching on:")
    for name, (_, port) in APPS.items():
        print(f"  {name:25s} -> http://127.0.0.1:{port}")

    time.sleep(3)
    for _, port in APPS.values():
        webbrowser.open(f"http://127.0.0.1:{port}")
        time.sleep(0.3)

    print("\nPress Ctrl+C to stop all.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        for proc in PROCS:
            proc.terminate()
