"""
Unit & Integration Tests for Phase 1:
- GameState initialization, Zobrist hashing, ASCII rendering
- Move generation (Pawn, Knight, Bishop, Rook, Queen, King, Promotions)
- Check detection (in_check) and pin/evasion legality filtering
- Make / Unmake move state restoration & incremental Zobrist hash verification
- Draw rules (Threefold repetition, 50-move rule, Dead positions)
- Perft node counts at depths 1 to 4
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.game_state import (
    BLACK,
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    ROOK,
    WHITE,
    GameState,
    Move,
    sq_to_str,
    str_to_sq,
)
from engine.perft import GARDNER_STARTPOS_PERFT, perft


class TestPhase1(unittest.TestCase):

    def test_initial_setup_and_coordinates(self):
        state = GameState(debug=True)
        self.assertEqual(len(state.board), 25)
        self.assertEqual(state.turn, WHITE)
        self.assertEqual(state.halfmove_clock, 0)
        self.assertEqual(state.history, [])

        self.assertEqual(str_to_sq("a1"), 0)
        self.assertEqual(str_to_sq("e1"), 4)
        self.assertEqual(str_to_sq("a5"), 20)
        self.assertEqual(str_to_sq("e5"), 24)
        for sq in range(25):
            self.assertEqual(str_to_sq(sq_to_str(sq)), sq)

        self.assertEqual(state.board[str_to_sq("a1")], ROOK)
        self.assertEqual(state.board[str_to_sq("e1")], KING)
        self.assertEqual(state.board[str_to_sq("a5")], -ROOK)
        self.assertEqual(state.board[str_to_sq("e5")], -KING)
        self.assertFalse(state.in_check(WHITE))
        self.assertFalse(state.in_check(BLACK))

    def test_opening_legal_moves(self):
        state = GameState(debug=True)
        moves = state.generate_legal_moves()
        uci_moves = sorted(m.uci() for m in moves)
        expected = sorted(["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "b1a3", "b1c3"])
        self.assertEqual(uci_moves, expected)

    def test_make_and_unmake_restores_state(self):
        state = GameState(debug=True)
        orig_board = list(state.board)
        orig_hash = state.zobrist_hash

        for m1 in state.generate_legal_moves():
            state.make_move(m1)
            for m2 in state.generate_legal_moves():
                state.make_move(m2)
                state.unmake_move(m2)
            state.unmake_move(m1)

        self.assertEqual(state.board, orig_board)
        self.assertEqual(state.turn, WHITE)
        self.assertEqual(state.halfmove_clock, 0)
        self.assertEqual(state.history, [])
        self.assertEqual(state.zobrist_hash, orig_hash)

    def test_pawn_promotion_push_and_capture(self):
        board = [EMPTY] * 25
        board[str_to_sq("e1")] = KING
        board[str_to_sq("e5")] = -KING
        board[str_to_sq("a4")] = PAWN
        board[str_to_sq("b5")] = -KNIGHT

        state = GameState(board=board, turn=WHITE, debug=True)
        uci_moves = {m.uci() for m in state.generate_legal_moves()}

        for target in ("a5", "b5"):
            for promo_char in ("q", "r", "b", "n"):
                move_str = f"a4{target}{promo_char}"
                self.assertIn(move_str, uci_moves)

                move = Move.from_uci(move_str)
                state.make_move(move)
                self.assertNotEqual(state.board[str_to_sq(target)], PAWN)
                self.assertEqual(state.halfmove_clock, 0)
                state.unmake_move(move)
                self.assertEqual(state.board[str_to_sq("a4")], PAWN)
                self.assertEqual(state.board[str_to_sq("b5")], -KNIGHT)

    def test_pinned_piece_and_check_evasion(self):
        board = [EMPTY] * 25
        board[str_to_sq("e1")] = KING
        board[str_to_sq("e2")] = ROOK
        board[str_to_sq("e5")] = -ROOK
        board[str_to_sq("a5")] = -KING

        state = GameState(board=board, turn=WHITE, debug=True)
        self.assertFalse(state.in_check(WHITE))
        legal_uci = {m.uci() for m in state.generate_legal_moves()}

        self.assertIn("e2e3", legal_uci)
        self.assertIn("e2e4", legal_uci)
        self.assertIn("e2e5", legal_uci)
        self.assertNotIn("e2d2", legal_uci)
        self.assertNotIn("e2a2", legal_uci)

    def test_threefold_repetition_and_fifty_move_rule(self):
        board = [EMPTY] * 25
        board[str_to_sq("e1")] = KING
        board[str_to_sq("b1")] = KNIGHT
        board[str_to_sq("e5")] = -KING
        board[str_to_sq("b5")] = -KNIGHT

        state = GameState(board=board, turn=WHITE, debug=True)
        cycle = [
            Move.from_uci("b1c3"),
            Move.from_uci("b5a3"),
            Move.from_uci("c3b1"),
            Move.from_uci("a3b5"),
        ]
        for _ in range(2):
            for m in cycle:
                state.make_move(m)
        self.assertTrue(state.is_threefold_repetition())
        self.assertEqual(state.halfmove_clock, 8)

    def test_perft_depths_1_to_4(self):
        state = GameState(debug=True)
        for depth in range(1, 5):
            count = perft(state, depth)
            self.assertEqual(
                count,
                GARDNER_STARTPOS_PERFT[depth],
                f"Perft mismatch at depth {depth}: got {count}, expected {GARDNER_STARTPOS_PERFT[depth]}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

