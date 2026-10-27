"""Step 2: run the whole project on fake data. Takes about 3 minutes.

In PyCharm: open this file and click the green ▶ arrow next to
`if __name__ == "__main__":` below (or the ▶ at the top right).
Results appear in the 'output/demo' folder on the left.
"""
import sys

from launch import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "demo"]
    main()
