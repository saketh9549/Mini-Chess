"""
Phase 3 Validation, Ablation Study & Benchmark Suite (Step 3.6):
- Mandatory Regression Test (rules.txt 6.5): 20 diverse positions at depths 1..4
  verifying Plain Negamax == Alpha-Beta (unordered, ordered, and with TT)
- Mate-distance scoring & TT ply-adjustment verification (rules.txt 5.3, 5.6)
- Iterative Deepening & mid-depth timeout safety test (rules.txt 6.6)
- Search Ablation Study at Depth 4 (Nodes, Effective Branching Factor b*, Time, TT Hit Rate)
- 100-game Arena Benchmark: SearchAgent(d=4) vs. GreedyBot (>= 95% win rate)
- 100-game Arena Benchmark: SearchAgent(d=4) vs. RandomBot (100% win rate)
"""

from __future__ import annotations
from pathlib import Path
import random
import sys
import unittest
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.search_agent import SearchAgent
from benchmarks.arena import generate_random_opening, run_tournament
from engine.eval import MATE_SCORE
from engine.game_state import (
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    GameState,
    str_to_sq,
)
from engine.move_order import KillerTable
from engine.search import search_alphabeta_root, search_negamax_root
from engine.transposition_table import TranspositionTable, score_from_tt, score_to_tt


def build_20_test_positions(seed: int = 2026) -> List[GameState]:
    """Build 20 diverse opening, midgame, tactical, and promotion positions."""
    positions: List[GameState] = [GameState()]
    rng = random.Random(seed)

    for i in range(16):
        plies = (i % 6) + 1
        st, _ = generate_random_opening(rng, opening_plies=plies)
        positions.append(st)

    b18 = [EMPTY] * 25
    b18[str_to_sq("e1")] = KING
    b18[str_to_sq("e2")] = ROOK
    b18[str_to_sq("c3")] = KNIGHT
    b18[str_to_sq("d4")] = -QUEEN
    b18[str_to_sq("e5")] = -ROOK
    b18[str_to_sq("a5")] = -KING
    positions.append(GameState(board=b18, turn=WHITE))

    b19 = [EMPTY] * 25
    b19[str_to_sq("e1")] = KING
    b19[str_to_sq("a4")] = PAWN
    b19[str_to_sq("b5")] = -ROOK
    b19[str_to_sq("d2")] = -PAWN
    b19[str_to_sq("e5")] = -KING
    positions.append(GameState(board=b19, turn=WHITE))

    b20 = [EMPTY] * 25
    b20[str_to_sq("a5")] = -KING
    b20[str_to_sq("c4")] = KING
    b20[str_to_sq("e3")] = QUEEN
    b20[str_to_sq("d1")] = ROOK
    positions.append(GameState(board=b20, turn=WHITE))

    return positions


def run_ablation_study(depth: int = 4) -> None:
    """Measure and print search efficiency across the 4 search variants at `depth`."""
    state = GameState()

    s_negamax = search_negamax_root(state.copy(), depth=depth)
    s_ab_unord = search_alphabeta_root(
        state.copy(), depth=depth, use_move_ordering=False, tt=None, killers=None
    )
    s_ab_ord = search_alphabeta_root(
        state.copy(), depth=depth, use_move_ordering=True, tt=None, killers=KillerTable()
    )

    agent_full = SearchAgent(
        max_depth=depth,
        mode="alphabeta",
        use_move_ordering=True,
        use_tt=True,
        use_iterative_deepening=True,
    )
    agent_full.select_move(state.copy())
    s_full = agent_full.last_stats

    variants = [
        ("1. Plain Negamax (Minimax)", s_negamax),
        ("2. Alpha-Beta (Unordered)", s_ab_unord),
        ("3. Alpha-Beta + Move Ordering", s_ab_ord),
        ("4. Alpha-Beta + Ordering + TT (IDS)", s_full),
    ]

    print("\n" + "=" * 86)
    print(f"PHASE 3 ABLATION STUDY (Starting Position, Fixed Depth = {depth})")
    print("=" * 86)
    print(
        f"{'Search Variant':<36} {'Nodes':>8} {'b*':>7} {'Time (s)':>10} "
        f"{'TT Hit%':>9} {'Score':>7} {'Best':>6}"
    )
    print("-" * 86)
    for label, st in variants:
        tt_pct = f"{st.tt_hit_rate * 100:.1f}%" if st.tt_probes > 0 else "N/A"
        best_u = st.best_move.uci() if st.best_move else "None"
        print(
            f"{label:<36} {st.nodes:>8} {st.effective_branching_factor:>7.2f} "
            f"{st.elapsed_sec:>10.4f} {tt_pct:>9} {st.score:>7} {best_u:>6}"
        )
    print("=" * 86)


class TestPhase3(unittest.TestCase):

    def test_tt_mate_score_ply_normalization(self):
        root_score = MATE_SCORE - 3
        stored = score_to_tt(root_score, ply=2)
        self.assertEqual(stored, MATE_SCORE - 1)
        retrieved_at_4 = score_from_tt(stored, ply=4)
        self.assertEqual(retrieved_at_4, MATE_SCORE - 5)

        neg_root_score = -(MATE_SCORE - 4)
        stored_neg = score_to_tt(neg_root_score, ply=1)
        self.assertEqual(stored_neg, -(MATE_SCORE - 3))
        self.assertEqual(score_from_tt(stored_neg, ply=1), neg_root_score)

    def test_mandatory_minimax_equals_alphabeta_regression(self):
        positions = build_20_test_positions(seed=2026)
        self.assertEqual(len(positions), 20)

        for idx, pos in enumerate(positions):
            for depth in (1, 2, 3, 4):
                st_minimax = search_negamax_root(pos.copy(), depth=depth)
                st_ab_unord = search_alphabeta_root(
                    pos.copy(), depth=depth, use_move_ordering=False, tt=None, killers=None
                )
                st_ab_ord = search_alphabeta_root(
                    pos.copy(),
                    depth=depth,
                    use_move_ordering=True,
                    tt=None,
                    killers=KillerTable(),
                )
                st_ab_tt = search_alphabeta_root(
                    pos.copy(),
                    depth=depth,
                    use_move_ordering=True,
                    tt=TranspositionTable(),
                    killers=KillerTable(),
                )

                self.assertEqual(st_minimax.score, st_ab_unord.score)
                self.assertEqual(st_minimax.best_move, st_ab_unord.best_move)
                self.assertEqual(st_minimax.score, st_ab_ord.score)
                self.assertEqual(st_minimax.score, st_ab_tt.score)

                if depth >= 2:
                    self.assertLessEqual(st_ab_ord.nodes, st_minimax.nodes)

    def test_ablation_node_reduction_at_depth_4(self):
        run_ablation_study(depth=4)
        state = GameState()
        s_minimax = search_negamax_root(state.copy(), depth=4)
        s_ab_unord = search_alphabeta_root(state.copy(), depth=4, use_move_ordering=False)
        s_ab_ord = search_alphabeta_root(
            state.copy(), depth=4, use_move_ordering=True, killers=KillerTable()
        )
        self.assertEqual(s_minimax.score, s_ab_unord.score)
        self.assertEqual(s_minimax.score, s_ab_ord.score)
        self.assertLess(s_ab_unord.nodes, s_minimax.nodes)
        self.assertLess(s_ab_ord.nodes, s_ab_unord.nodes)

    def test_iterative_deepening_timeout_safety(self):
        state = GameState(debug=True)
        orig_board = list(state.board)
        orig_hash = state.zobrist_hash

        agent = SearchAgent(
            max_depth=10,
            time_limit_sec=0.01,
            check_interval=64,
        )
        move = agent.select_move(state)
        self.assertIsNotNone(move)
        self.assertIn(move, state.generate_legal_moves())
        self.assertEqual(state.board, orig_board)
        self.assertEqual(state.zobrist_hash, orig_hash)
        self.assertEqual(len(state.history), 0)
        self.assertTrue(agent.last_stats.timed_out)

    def test_search_agent_vs_greedy_100_games(self):
        search_bot = SearchAgent(max_depth=4, name="SearchAgent(d=4)")
        greedy_bot = GreedyAgent(seed=42, name="GreedyBot(1-ply)")
        stats = run_tournament(
            search_bot,
            greedy_bot,
            num_games=100,
            opening_plies=(2, 4),
            seed=42,
            verbose=True,
        )
        self.assertGreaterEqual(stats.win_rate, 0.95)

    def test_search_agent_vs_random_100_games(self):
        search_bot = SearchAgent(max_depth=4, name="SearchAgent(d=4)")
        random_bot = RandomAgent(seed=99, name="RandomBot")
        stats = run_tournament(
            search_bot,
            random_bot,
            num_games=100,
            opening_plies=(2, 4),
            seed=42,
            verbose=True,
        )
        self.assertGreaterEqual(stats.win_rate, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)

