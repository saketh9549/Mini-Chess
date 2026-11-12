"""
Phase 1: Perft (Performance Test / Move Path Enumeration)
Recursive leaf-node counter to verify move generation and make/unmake integrity.
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.game_state import GameState

GARDNER_STARTPOS_PERFT: Dict[int, int] = {
    0: 1,
    1: 7,
    2: 53,
    3: 506,
    4: 4775,
    5: 52512,
}


def perft(state: GameState, depth: int) -> int:
    """Recursively count leaf nodes at `depth` from `state`."""
    if depth == 0:
        return 1

    moves = state.generate_legal_moves()
    if depth == 1:
        return len(moves)

    nodes = 0
    for move in moves:
        state.make_move(move)
        nodes += perft(state, depth - 1)
        state.unmake_move(move)
    return nodes


def perft_divide(state: GameState, depth: int) -> Tuple[int, Dict[str, int]]:
    """Run perft with root move breakdown."""
    if depth <= 0:
        return 1, {}

    breakdown: Dict[str, int] = {}
    total = 0
    for move in state.generate_legal_moves():
        state.make_move(move)
        count = perft(state, depth - 1)
        state.unmake_move(move)
        breakdown[move.uci()] = count
        total += count
    return total, breakdown


def run_perft_suite(max_depth: int = 4, debug_hash: bool = False) -> bool:
    """Run Perft verification from the initial Gardner 5x5 position for depths 1..max_depth."""
    state = GameState(debug=debug_hash)
    print("=" * 64)
    print("GARDNER 5x5 MINICHESS - PERFT VERIFICATION")
    print("=" * 64)
    print(state.render_ascii())
    print("-" * 64)
    print(f"{'Depth':<8} {'Nodes':<12} {'Expected':<12} {'Status':<10} {'Time (s)':<10} {'NPS':<12}")
    print("-" * 64)

    all_passed = True
    for d in range(1, max_depth + 1):
        t0 = time.perf_counter()
        nodes = perft(state, d)
        elapsed = time.perf_counter() - t0
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        expected = GARDNER_STARTPOS_PERFT.get(d)
        if expected is not None:
            passed = nodes == expected
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            exp_str = str(expected)
        else:
            status = "INFO"
            exp_str = "N/A"
        print(f"{d:<8} {nodes:<12} {exp_str:<12} {status:<10} {elapsed:<10.4f} {nps:<12}")

    print("-" * 64)
    return all_passed


if __name__ == "__main__":
    max_d = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ok = run_perft_suite(max_depth=max_d, debug_hash=True)
    if not ok:
        sys.exit(1)

