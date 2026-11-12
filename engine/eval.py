"""
Phase 2 (Step 2.3): Heuristic Position Evaluation Function
Implements centipawn material balance and 5x5 Piece-Square Tables (PST)
following rules.txt Section 5 (Negamax / side-to-move convention).
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
from engine.game_state import (
    BOARD_SIZE,
    EMPTY,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    GameState,
)

MATE_SCORE: int = 100000
DRAW_SCORE: int = 0

MATERIAL_WEIGHTS: Dict[int, int] = {
    EMPTY: 0,
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 300,
    ROOK: 500,
    QUEEN: 900,
    KING: 0,
}


def flip_square(sq: int) -> int:
    """
    Mirror square index vertically for Black's perspective:
        black_sq = (4 - (sq // 5)) * 5 + (sq % 5)
    """
    return (4 - (sq // 5)) * 5 + (sq % 5)


FLIP_SQ: Tuple[int, ...] = tuple(flip_square(sq) for sq in range(BOARD_SIZE))

# 5x5 Piece-Square Tables (from White's perspective: index 0..4 is Rank 1, 20..24 is Rank 5)
PAWN_PST: Tuple[int, ...] = (
      0,   0,   0,   0,   0,
      5,  10,  15,  10,   5,
     15,  25,  35,  25,  15,
     40,  50,  60,  50,  40,
      0,   0,   0,   0,   0,
)

KNIGHT_PST: Tuple[int, ...] = (
    -20, -10, -10, -10, -20,
    -10,  10,  15,  10, -10,
    -10,  15,  30,  15, -10,
    -10,  15,  20,  15, -10,
    -20, -10, -10, -10, -20,
)

BISHOP_PST: Tuple[int, ...] = (
    -15,  -5, -10,  -5, -15,
     -5,  15,  10,  15,  -5,
    -10,  10,  25,  10, -10,
     -5,  15,  10,  15,  -5,
    -15,  -5, -10,  -5, -15,
)

ROOK_PST: Tuple[int, ...] = (
      0,   5,  10,   5,   0,
      0,   5,  10,   5,   0,
      5,  10,  15,  10,   5,
     15,  20,  25,  20,  15,
     10,  15,  20,  15,  10,
)

QUEEN_PST: Tuple[int, ...] = (
    -10,  -5,   0,  -5, -10,
     -5,  10,  15,  10,  -5,
      0,  15,  25,  15,   0,
     -5,  15,  20,  15,  -5,
    -10,  -5,   5,  -5, -10,
)

KING_PST: Tuple[int, ...] = (
     15,  10,   0,  10,  20,
      0,  -5, -15,  -5,   0,
    -15, -25, -35, -25, -15,
    -25, -35, -45, -35, -25,
    -35, -45, -50, -45, -35,
)

PST_TABLES: Dict[int, Tuple[int, ...]] = {
    EMPTY: (0,) * BOARD_SIZE,
    PAWN: PAWN_PST,
    KNIGHT: KNIGHT_PST,
    BISHOP: BISHOP_PST,
    ROOK: ROOK_PST,
    QUEEN: QUEEN_PST,
    KING: KING_PST,
}


def _build_combined_table(
    weights: Dict[int, int],
    pst: Dict[int, Sequence[int]],
    use_pst: bool = True,
) -> Tuple[Tuple[int, ...], ...]:
    table = []
    for p in range(-6, 7):
        if p == 0:
            table.append((0,) * BOARD_SIZE)
        elif p > 0:
            mat = weights[p]
            p_table = pst[p]
            table.append(
                tuple(mat + (p_table[sq] if use_pst else 0) for sq in range(BOARD_SIZE))
            )
        else:
            pt = -p
            mat = weights[pt]
            p_table = pst[pt]
            table.append(
                tuple(-(mat + (p_table[FLIP_SQ[sq]] if use_pst else 0)) for sq in range(BOARD_SIZE))
            )
    return tuple(table)


DEFAULT_COMBINED_TABLE = _build_combined_table(MATERIAL_WEIGHTS, PST_TABLES, use_pst=True)
MATERIAL_ONLY_TABLE = _build_combined_table(MATERIAL_WEIGHTS, PST_TABLES, use_pst=False)


def evaluate_white(
    state: GameState,
    use_pst: bool = True,
    custom_weights: Optional[Dict[int, int]] = None,
    custom_pst: Optional[Dict[int, Sequence[int]]] = None,
) -> int:
    """Compute static evaluation from White's perspective."""
    board = state.board
    if custom_weights is None and custom_pst is None:
        lookup = DEFAULT_COMBINED_TABLE if use_pst else MATERIAL_ONLY_TABLE
        score = 0
        for sq in range(BOARD_SIZE):
            piece = board[sq]
            if piece != EMPTY:
                score += lookup[piece + 6][sq]
        return score

    weights = custom_weights if custom_weights is not None else MATERIAL_WEIGHTS
    pst = custom_pst if custom_pst is not None else PST_TABLES
    score = 0
    for sq, piece in enumerate(board):
        if piece > 0:
            score += weights[piece] + (pst[piece][sq] if use_pst else 0)
        elif piece < 0:
            pt = -piece
            score -= weights[pt] + (pst[pt][FLIP_SQ[sq]] if use_pst else 0)
    return score


def evaluate(
    state: GameState,
    use_pst: bool = True,
    custom_weights: Optional[Dict[int, int]] = None,
    custom_pst: Optional[Dict[int, Sequence[int]]] = None,
) -> int:
    """Compute static heuristic evaluation from the side-to-move perspective (Negamax)."""
    white_score = evaluate_white(
        state,
        use_pst=use_pst,
        custom_weights=custom_weights,
        custom_pst=custom_pst,
    )
    return state.turn * white_score

