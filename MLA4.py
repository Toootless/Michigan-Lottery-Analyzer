"""
MLA4 - Michigan Lottery Analyzer (renamed entrypoint)

This entrypoint executes the legacy Streamlit app defined in
MichiganLotteryAnalyzer.py. The original file did not expose a main()
function, so we execute its top-level code via runpy.
"""

import runpy
import sys


def main():
    try:
        # Execute the legacy app module as if it were run directly
        runpy.run_module("MichiganLotteryAnalyzer", run_name="__main__")
    except Exception as e:
        print("Failed to execute MichiganLotteryAnalyzer:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
