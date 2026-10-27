"""Runs the project inside the .venv environment made by step1_setup.py.

The step*.py files call this, so you never need the terminal:
    launch.py demo        launch.py analyze --start ... --end ...        launch.py test
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main():
    if not PY.exists():
        print("!! The project isn't set up yet.")
        print("   Open step1_setup.py and click its green ▶ arrow first.")
        sys.exit(1)
    args = sys.argv[1:] or ["demo"]
    cmd = [str(PY), "-m", "pytest", "-v"] if args[:1] == ["test"] else [str(PY), "run.py", *args]
    result = subprocess.call(cmd, cwd=HERE)
    if result == 0 and args[:1] != ["test"]:
        print("\n✅ Done. Open the 'output' folder in the project panel on the left to see the charts.")
    sys.exit(result)


if __name__ == "__main__":
    main()
