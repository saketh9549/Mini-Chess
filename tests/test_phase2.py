"""
Phase 2 Validation & Benchmark Suite (Step 2.5):
- Unit tests for Agent abstraction and RandomAgent
- Unit tests for Heuristic Evaluation (symmetry, rank-flipping, PST bonuses, perspective)
- Unit tests for GreedyAgent (mate-in-1 detection, material capture, avoiding stalemate)
- Sanity Test: RandomBot vs. RandomBot (50 games, all terminate cleanly)
- Benchmark Target: GreedyBot vs. RandomBot (100 games, >= 95% win rate)
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from benchmarks.arena import adjudicate_state, run_tournament
from engine.eval import FLIP_SQ, evaluate, evaluate_white, flip_square
from engine.game_state import (
    BLACK,
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    ROOK,
    WHITE,
    GameState,
    str_to_sq,
)


class TestPhase2(unittest.TestCase):

    def test_eval_starting_position_symmetry(self):
        state = GameState()
        self.assertEqual(evaluate_white(state, use_pst=True), 0)
        self.assertEqual(evaluate_white(state, use_pst=False), 0)
        self.assertEqual(evaluate(state, use_pst=True), 0)

        state.turn = BLACK
        self.assertEqual(evaluate(state, use_pst=True), 0)

    def test_rank_flipping_formula(self):
        self.assertEqual(flip_square(str_to_sq("a1")), str_to_sq("a5"))
        self.assertEqual(flip_square(str_to_sq("e1")), str_to_sq("e5"))
        self.assertEqual(flip_square(str_to_sq("c3")), str_to_sq("c3"))
        self.assertEqual(flip_square(str_to_sq("b2")), str_to_sq("b4"))
        for sq in range(25):
            self.assertEqual(FLIP_SQ[FLIP_SQ[sq]], sq)

    def test_eval_perspective_and_pst_bonuses(self):
        board = [EMPTY] * 25
        board[str_to_sq("e1")] = KING
        board[str_to_sq("e5")] = -KING
        board[str_to_sq("c3")] = KNIGHT
        board[str_to_sq("a5")] = -KNIGHT

        state_w = GameState(board=board, turn=WHITE)
        state_b = GameState(board=board, turn=BLACK)

        self.assertEqual(evaluate(state_w, use_pst=False), 0)
        self.assertEqual(evaluate(state_w, use_pst=True), 50)
        self.assertEqual(evaluate(state_b, use_pst=True), -50)

    def test_greedy_finds_mate_in_one(self):
        board = [EMPTY] * 25
        board[str_to_sq("a5")] = -KING
        board[str_to_sq("b3")] = KING
        board[str_to_sq("a3")] = PAWN
        board[str_to_sq("e4")] = ROOK

        state = GameState(board=board, turn=WHITE)
        greedy = GreedyAgent(seed=1)
        move = greedy.select_move(state)
        self.assertIsNotNone(move)
        state.make_move(move)
        verdict = adjudicate_state(state)
        self.assertEqual(verdict, (WHITE, "checkmate"))

    def test_random_vs_random_50_games_sanity(self):
        r1 = RandomAgent(seed=11, name="RandomBot-1")
        r2 = RandomAgent(seed=22, name="RandomBot-2")
        stats = run_tournament(
            r1, r2, num_games=50, opening_plies=(2, 4), seed=42, verbose=True
        )
        self.assertEqual(stats.total_games, 50)
        self.assertEqual(stats.wins + stats.draws + stats.losses, 50)
        for g in stats.games:
            self.assertLessEqual(g.plies, 300)

    def test_greedy_vs_random_100_games_benchmark(self):
        greedy = GreedyAgent(seed=42, name="GreedyBot")
        rand_bot = RandomAgent(seed=99, name="RandomBot")
        stats = run_tournament(
            greedy, rand_bot, num_games=100, opening_plies=2, seed=42, verbose=True
        )
        self.assertEqual(stats.total_games, 100)
        self.assertGreaterEqual(
            stats.win_rate,
            0.95,
            f"Expected GreedyBot win rate >= 95%, got {stats.win_rate * 100:.1f}%",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

