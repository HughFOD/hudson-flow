"""One-time setup: creates a private Python environment (.venv) inside this
project and installs everything the project needs into it.

In PyCharm: open this file and click the green ▶ arrow next to
`if __name__ == "__main__":` at the bottom. Takes a few minutes the first time.
"""
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main():
    print(f"Using Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 10):
        print("\n!! This project needs Python 3.10 or newer.")
        print("   Install the latest Python from https://www.python.org/downloads/ ,")
        print("   then in PyCharm: Settings > Project > Python Interpreter > choose it,")
        print("   and run this setup again.")
        sys.exit(1)

    if not PY.exists():
        print("\nCreating environment in .venv ...")
        venv.create(VENV, with_pip=True)
    print("\nInstalling packages (this can take a few minutes) ...\n")
    subprocess.check_call([str(PY), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(PY), "-m", "pip", "install", "-r", str(HERE / "requirements.txt")])
    print("\n✅ Setup complete. Now open step2_run_demo.py and click its green ▶ arrow.")


if __name__ == "__main__":
    main()
