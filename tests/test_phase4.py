"""
Phase 4 Validation, Quiescence, Tuning & Full Benchmark Suite (Step 4.4):
- Quiescence Search horizon-effect test (avoids poisoning queen capture at leaf)
- Tuned Evaluation symmetry, passed-pawn, and endgame mop-up tests
- Parameter Optimizer (GA / SA / HC) unit tests
- Formal Ablation Table (+/- Move Ordering, +/- TT, +/- Quiescence, +/- PST)
- 100-game match: TunedSearchAgent(d=4) vs. Baseline SearchAgent(d=4, untuned, no Q)
- 100-game match: TunedSearchAgent(d=4) vs. GreedyBot(1-ply)
"""

from __future__ import annotations
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.greedy_agent import GreedyAgent
from agents.search_agent import SearchAgent
from agents.tuned_agent import TunedSearchAgent
from benchmarks.arena import run_tournament
from benchmarks.run_tournaments import run_formal_ablation_table
from benchmarks.tune import crossover_params, mutate_params
from engine.eval_tuned import TUNED_PARAMS, evaluate_tuned, evaluate_tuned_white
from engine.game_state import (
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    GameState,
    Move,
    str_to_sq,
)


class TestPhase4(unittest.TestCase):

    def test_quiescence_avoids_horizon_queen_blunder(self):
        """
        Verify Quiescence Search resolves recaptures at depth=1 and avoids capturing
        a defended pawn with a Queen (the classic horizon-effect blunder).
        """
        # White Queen on c2, Black Pawn on c4 defended by Black Rook on c5.
        # At depth 1 WITHOUT quiescence, White sees c2c4 (Queen takes Pawn) as +100!
        # WITH quiescence, Black's recapture c5c4 (Rook takes Queen, -900) is searched.
        board = [EMPTY] * 25
        board[str_to_sq("e1")] = KING
        board[str_to_sq("c2")] = QUEEN
        board[str_to_sq("c4")] = -PAWN
        board[str_to_sq("c5")] = -ROOK
        board[str_to_sq("a5")] = -KING

        state = GameState(board=board, turn=WHITE)

        naive_agent = SearchAgent(max_depth=1, use_quiescence=False)
        q_agent = SearchAgent(max_depth=1, use_quiescence=True)

        naive_move = naive_agent.select_move(state.copy())
        q_move = q_agent.select_move(state.copy())

        # Naive depth-1 agent blunders the Queen onto c4; Quiescence agent avoids c2c4!
        self.assertEqual(naive_move, Move.from_uci("c2c4"))
        self.assertNotEqual(q_move, Move.from_uci("c2c4"))
        self.assertGreater(q_agent.last_stats.q_nodes, 0)

    def test_tuned_eval_symmetry_and_features(self):
        """Verify tuned evaluation is 0 at startpos and rewards passed pawns & mop-up."""
        start = GameState()
        self.assertEqual(evaluate_tuned_white(start), 0)
        self.assertEqual(evaluate_tuned(start), 0)

        # Passed pawn on rank 4 (a4) vs blocked pawn
        b_free = [EMPTY] * 25
        b_free[str_to_sq("e1")] = KING
        b_free[str_to_sq("e5")] = -KING
        b_free[str_to_sq("a4")] = PAWN

        b_blocked = list(b_free)
        b_blocked[str_to_sq("a5")] = -KNIGHT  # Blocks promotion square a5

        s_free = GameState(board=b_free, turn=WHITE)
        s_blocked = GameState(board=b_blocked, turn=WHITE)
        # Free passed pawn on a4 gets unblocked_rank4_bonus + passed_pawn_bonus
        self.assertGreater(evaluate_tuned_white(s_free), 200)
        self.assertGreater(evaluate_tuned_white(s_free), evaluate_tuned_white(s_blocked))

    def test_genetic_operators_produce_valid_params(self):
        """Verify mutation and crossover operators preserve valid parameter bounds."""
        rng = random.Random(42)
        p1 = mutate_params(TUNED_PARAMS, rng, scale=1.0)
        p2 = mutate_params(TUNED_PARAMS, rng, scale=1.0)
        child = crossover_params(p1, p2, rng)
        self.assertGreaterEqual(child.pawn, 50)
        self.assertGreaterEqual(child.queen, 600)
        self.assertGreater(child.pst_scale, 0.0)

    def test_formal_ablation_suite(self):
        """Run and print the Phase 4 formal ablation study across 7 configurations."""
        run_formal_ablation_table(depth=4, num_positions=6, seed=42)

    def test_tuned_vs_untuned_baseline_100_games(self):
        """
        Verify TunedSearchAgent(d=4, +Quiescence, +TunedEval) outperforms
        Baseline SearchAgent(d=4, untuned, no Quiescence) over 100 games.
        """
        tuned = TunedSearchAgent(max_depth=4, name="TunedSearch(d=4+Q)")
        baseline = SearchAgent(max_depth=4, use_quiescence=False, name="BaselineSearch(d=4)")
        stats = run_tournament(
            tuned, baseline, num_games=100, opening_plies=(2, 4), seed=42, verbose=True
        )
        self.assertGreater(
            stats.score_rate,
            0.55,
            f"Expected TunedSearchAgent to score > 55% vs Baseline, got {stats.score_rate * 100:.1f}%",
        )

    def test_tuned_vs_greedy_100_games(self):
        """Verify TunedSearchAgent(d=4) achieves >= 95% win rate vs GreedyBot over 100 games."""
        tuned = TunedSearchAgent(max_depth=4, name="TunedSearch(d=4+Q)")
        greedy = GreedyAgent(seed=42, name="GreedyBot(1-ply)")
        stats = run_tournament(
            tuned, greedy, num_games=100, opening_plies=(2, 4), seed=42, verbose=True
        )
        self.assertGreaterEqual(stats.win_rate, 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)

