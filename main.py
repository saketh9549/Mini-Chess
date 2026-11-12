"""
Main Entry Point for Gardner's 5x5 Mini Chess AI Project.

Usage:
    python main.py                  # Launch Desktop GUI Application (Default)
    python main.py --gui            # Launch Desktop GUI Application
    python main.py --cli            # Launch Interactive ASCII Terminal CLI
    python main.py --perft          # Run Phase 1 Perft Verification (d=1..5)
    python main.py --test           # Run Full Test Suite (Phases 1..4)
    python main.py --ablation       # Run Formal Search Ablation Study
    python main.py --tournament     # Run 4-Tier Round-Robin Tournament Matrix
    python main.py --tune           # Run Heuristic Weight Optimizer (GA/SA/HC)
"""

from __future__ import annotations
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gardner's 5x5 Mini Chess — AI Engine, GUI App & Benchmark Suite"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true", help="Launch Desktop GUI App (default)")
    group.add_argument("--cli", action="store_true", help="Launch Interactive Terminal CLI")
    group.add_argument("--perft", action="store_true", help="Run Perft verification (depths 1..5)")
    group.add_argument("--test", action="store_true", help="Run all Phase 1-4 unit & benchmark tests")
    group.add_argument("--ablation", action="store_true", help="Run formal search ablation study")
    group.add_argument("--tournament", action="store_true", help="Run full 4-tier tournament matrix")
    group.add_argument("--tune", action="store_true", help="Run heuristic parameter tuning")

    parser.add_argument("--depth", type=int, default=4, help="Search depth for Perft or Ablation")
    parser.add_argument("--games", type=int, default=40, help="Games per matchup for --tournament")
    args = parser.parse_args()

    if args.cli:
        from app.cli import main as cli_main
        cli_main()
    elif args.perft:
        from engine.perft import run_perft_suite
        run_perft_suite(max_depth=max(args.depth, 5))
    elif args.test:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir="tests", pattern="test_phase*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)
    elif args.ablation:
        from benchmarks.run_tournaments import run_formal_ablation_table
        run_formal_ablation_table(depth=args.depth)
    elif args.tournament:
        from benchmarks.run_tournaments import run_full_round_robin
        run_full_round_robin(games_per_pair=args.games)
    elif args.tune:
        from benchmarks.tune import tune_hill_climbing
        tune_hill_climbing(iterations=12, games_per_eval=8, depth=2)
    else:
        from app.gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

