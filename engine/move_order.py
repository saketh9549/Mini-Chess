"""
Phase 3 (Step 3.3): Move Ordering Heuristics
Orders legal moves to maximize Alpha-Beta pruning efficiency:
  1. Transposition Table / Principal Variation Move (PV)
  2. Captures (MVV-LVA: Victim_Value * 10 - Attacker_Value) & Pawn Promotions
  3. Killer Moves (2 quiet beta-cutoff moves per ply)
  4. Quiet Moves ordered by Piece-Square Table (PST) positional gain
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from engine.eval import FLIP_SQ, MATERIAL_WEIGHTS, PST_TABLES
from engine.game_state import EMPTY, WHITE, GameState, Move

MAX_PLY = 128

PV_MOVE_BONUS = 1_000_000
CAPTURE_BASE_BONUS = 100_000
PROMOTION_BASE_BONUS = 90_000
KILLER_1_BONUS = 80_000
KILLER_2_BONUS = 79_000


class KillerTable:
    """Stores up to 2 quiet moves per search ply that caused a beta cutoff."""

    def __init__(self, max_ply: int = MAX_PLY) -> None:
        self.max_ply = max_ply
        self._killers: List[List[Optional[Move]]] = [[None, None] for _ in range(max_ply)]

    def clear(self) -> None:
        for slot in self._killers:
            slot[0] = None
            slot[1] = None

    def record(self, ply: int, move: Move) -> None:
        if ply >= self.max_ply or move.promotion != 0:
            return
        slot = self._killers[ply]
        if slot[0] != move:
            slot[1] = slot[0]
            slot[0] = move

    def get(self, ply: int) -> Tuple[Optional[Move], Optional[Move]]:
        if ply < self.max_ply:
            s = self._killers[ply]
            return s[0], s[1]
        return None, None


def mvv_lva_score(victim_type: int, attacker_type: int) -> int:
    """Most Valuable Victim - Least Valuable Attacker: Victim_Value * 10 - Attacker_Value."""
    return MATERIAL_WEIGHTS[victim_type] * 10 - MATERIAL_WEIGHTS[attacker_type]


def score_move(
    state: GameState,
    move: Move,
    tt_move: Optional[Move] = None,
    killer_1: Optional[Move] = None,
    killer_2: Optional[Move] = None,
) -> int:
    """Assign a heuristic ordering score to `move` in `state`."""
    if tt_move is not None and move == tt_move:
        return PV_MOVE_BONUS

    board = state.board
    from_sq = move.from_sq
    to_sq = move.to_sq
    promo = move.promotion

    attacker_piece = board[from_sq]
    attacker_type = abs(attacker_piece)
    captured_piece = board[to_sq]

    if captured_piece != EMPTY:
        victim_type = abs(captured_piece)
        promo_bonus = MATERIAL_WEIGHTS[promo] * 10 if promo else 0
        return CAPTURE_BASE_BONUS + mvv_lva_score(victim_type, attacker_type) + promo_bonus

    if promo != 0:
        return PROMOTION_BASE_BONUS + MATERIAL_WEIGHTS[promo] * 10

    if killer_1 is not None and move == killer_1:
        return KILLER_1_BONUS
    if killer_2 is not None and move == killer_2:
        return KILLER_2_BONUS

    pst = PST_TABLES[attacker_type]
    if state.turn == WHITE:
        return pst[to_sq] - pst[from_sq]
    else:
        return pst[FLIP_SQ[to_sq]] - pst[FLIP_SQ[from_sq]]


def order_moves(
    state: GameState,
    moves: List[Move],
    tt_move: Optional[Move] = None,
    killers: Optional[KillerTable] = None,
    ply: int = 0,
) -> List[Move]:
    """Return `moves` sorted in descending order of heuristic move-ordering priority."""
    if len(moves) <= 1:
        return moves

    k1, k2 = killers.get(ply) if killers is not None else (None, None)
    return sorted(
        moves,
        key=lambda m: score_move(state, m, tt_move=tt_move, killer_1=k1, killer_2=k2),
        reverse=True,
    )

