"""
Unified Test Runner for all 4 Phases of Mini Chess (`tests/`).
Usage:
    python tests/run_all_tests.py          # Runs all Phase 1-4 tests (including 100-game benchmarks)
    python tests/run_all_tests.py --phase 1
    python tests/run_all_tests.py --phase 2
    python tests/run_all_tests.py --phase 3
    python tests/run_all_tests.py --phase 4
"""

from __future__ import annotations
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mini Chess Test Suite (Phases 1-4)")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only a specific phase test suite (1, 2, 3, or 4)",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    if args.phase is not None:
        pattern = f"test_phase{args.phase}.py"
    else:
        pattern = "test_phase*.py"

    discovered = loader.discover(start_dir=tests_dir, pattern=pattern)
    suite.addTests(discovered)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

