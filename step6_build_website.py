"""Step 6: build the project website from your latest real-data results.

In PyCharm: click the green ▶ arrow next to `if __name__ == "__main__":` below.
It writes the site into the `docs` folder: docs/index.html plus the charts.
Double-click docs/index.html in Finder to preview it in your browser.

To put it online for free, push the project to GitHub, then on GitHub:
Settings > Pages > Source: "Deploy from a branch" > Branch: main, folder: /docs > Save.
A minute later it is live at https://<your-username>.github.io/<repo-name>/
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if __name__ == "__main__":
    if not PY.exists():
        sys.exit("Run step1_setup.py first.")
    sys.exit(subprocess.call([str(PY), "-m", "hudson.website"], cwd=HERE))
