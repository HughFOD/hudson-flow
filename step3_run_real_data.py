"""Step 3: download real Hudson River data (2022-2025) and analyze it.

In PyCharm: open this file and click the green ▶ arrow next to
`if __name__ == "__main__":` below (or the ▶ at the top right).
The first run downloads a lot of data and can take 10+ minutes.
Results appear in the 'output' folder on the left.
"""
import sys

from launch import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "analyze", "--start", "2022-01-01", "--end", "2025-12-31"]
    main()
