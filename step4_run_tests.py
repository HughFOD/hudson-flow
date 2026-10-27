"""Step 4 (optional): run the automated checks. Takes about 1 minute.

In PyCharm: open this file and click the green ▶ arrow next to
`if __name__ == "__main__":` below (or the ▶ at the top right).
"""
import sys

from launch import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "test"]
    main()
