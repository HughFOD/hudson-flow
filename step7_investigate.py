"""Step 7: why doesn't the flow balance close? Tests the canal explanation
against the real canal season dates, finds periods where the gap jumps, and
compares each gauge with USGS hand measurements to see which one drifted.

Run step3 first. In PyCharm: click the green ▶ next to `if __name__ == "__main__":`.
Then re-run step6 to put the results on the website.
"""
import sys

from launch import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "investigate", "--start", "2022-01-01", "--end", "2025-12-31"]
    main()
