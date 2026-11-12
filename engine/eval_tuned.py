"""
Phase 4 (Step 4.3): Tuned Evaluation Function for 5x5 Gardner Minichess
Combines optimized material weights, scaled 5x5 Piece-Square Tables (PST),
passed-pawn / promotion-threat bonuses, and endgame king-tropism (mop-up)
evaluation obtained via evolutionary/local-search parameter tuning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from engine.eval import FLIP_SQ, PST_TABLES
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


@dataclass
class EvalParams:
    """Parameter vector (chromosome / state) for heuristic evaluation tuning."""
    pawn: int = 115
    knight: int = 310
    bishop: int = 290
    rook: int = 525
    queen: int = 950
    pst_scale: float = 1.20
    passed_pawn_bonus: int = 35
    unblocked_rank4_bonus: int = 45
    mopup_weight: int = 12

    def to_vector(self) -> List[float]:
        return [
            float(self.pawn),
            float(self.knight),
            float(self.bishop),
            float(self.rook),
            float(self.queen),
            float(self.pst_scale),
            float(self.passed_pawn_bonus),
            float(self.unblocked_rank4_bonus),
            float(self.mopup_weight),
        ]

    @classmethod
    def from_vector(cls, vec: Sequence[float]) -> "EvalParams":
        return cls(
            pawn=max(50, int(round(vec[0]))),
            knight=max(150, int(round(vec[1]))),
            bishop=max(150, int(round(vec[2]))),
            rook=max(300, int(round(vec[3]))),
            queen=max(600, int(round(vec[4]))),
            pst_scale=max(0.2, round(float(vec[5]), 3)),
            passed_pawn_bonus=max(0, int(round(vec[6]))),
            unblocked_rank4_bonus=max(0, int(round(vec[7]))),
            mopup_weight=max(0, int(round(vec[8]))),
        )

    def material_dict(self) -> Dict[int, int]:
        return {
            EMPTY: 0,
            PAWN: self.pawn,
            KNIGHT: self.knight,
            BISHOP: self.bishop,
            ROOK: self.rook,
            QUEEN: self.queen,
            KING: 0,
        }


TUNED_PARAMS = EvalParams(
    pawn=115,
    knight=310,
    bishop=290,
    rook=525,
    queen=950,
    pst_scale=1.20,
    passed_pawn_bonus=35,
    unblocked_rank4_bonus=45,
    mopup_weight=12,
)

TUNED_WEIGHTS: Dict[int, int] = TUNED_PARAMS.material_dict()

CENTER_MANHATTAN_DIST: Tuple[int, ...] = tuple(
    abs((sq // 5) - 2) + abs((sq % 5) - 2) for sq in range(BOARD_SIZE)
)


def build_fast_lookup(params: EvalParams) -> Tuple[Tuple[int, ...], ...]:
    """Precompute [piece + 6][sq] -> signed (tuned_material + round(pst_scale * pst)) table."""
    weights = params.material_dict()
    scale = params.pst_scale
    table = []
    for p in range(-6, 7):
        if p == 0:
            table.append((0,) * BOARD_SIZE)
        elif p > 0:
            mat = weights[p]
            p_table = PST_TABLES[p]
            table.append(
                tuple(mat + int(round(scale * p_table[sq])) for sq in range(BOARD_SIZE))
            )
        else:
            pt = -p
            mat = weights[pt]
            p_table = PST_TABLES[pt]
            table.append(
                tuple(-(mat + int(round(scale * p_table[FLIP_SQ[sq]]))) for sq in range(BOARD_SIZE))
            )
    return tuple(table)


_DEFAULT_TUNED_LOOKUP = build_fast_lookup(TUNED_PARAMS)


def evaluate_tuned_white(
    state: GameState,
    params: Optional[EvalParams] = None,
    lookup: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> int:
    """
    Compute tuned evaluation from White's perspective:
      1. Tuned Material + Scaled 5x5 PST
      2. Passed Pawn & Unblocked Rank-4/Rank-2 Promotion Threat Bonuses
      3. Endgame Mop-Up / King Tropism
    """
    if params is None:
        params = TUNED_PARAMS
        if lookup is None:
            lookup = _DEFAULT_TUNED_LOOKUP
    elif lookup is None:
        lookup = build_fast_lookup(params)

    board = state.board
    score = 0
    w_king_sq = -1
    b_king_sq = -1

    w_pawns = []
    b_pawns = []

    for sq in range(BOARD_SIZE):
        piece = board[sq]
        if piece != EMPTY:
            score += lookup[piece + 6][sq]
            if piece == KING:
                w_king_sq = sq
            elif piece == -KING:
                b_king_sq = sq
            elif piece == PAWN:
                w_pawns.append(sq)
            elif piece == -PAWN:
                b_pawns.append(sq)

    passed_bonus = params.passed_pawn_bonus
    rank4_bonus = params.unblocked_rank4_bonus

    for sq in w_pawns:
        r, f = divmod(sq, 5)
        if r == 3 and board[sq + 5] == EMPTY:
            score += rank4_bonus
        is_passed = True
        for b_sq in b_pawns:
            br, bf = divmod(b_sq, 5)
            if br > r and abs(bf - f) <= 1:
                is_passed = False
                break
        if is_passed:
            score += passed_bonus * r

    for sq in b_pawns:
        r, f = divmod(sq, 5)
        if r == 1 and board[sq - 5] == EMPTY:
            score -= rank4_bonus
        is_passed = True
        for w_sq in w_pawns:
            wr, wf = divmod(w_sq, 5)
            if wr < r and abs(wf - f) <= 1:
                is_passed = False
                break
        if is_passed:
            score -= passed_bonus * (4 - r)

    mopup = params.mopup_weight
    if mopup > 0 and w_king_sq != -1 and b_king_sq != -1 and abs(score) >= 200:
        wr, wf = divmod(w_king_sq, 5)
        br, bf = divmod(b_king_sq, 5)
        king_chebyshev = max(abs(wr - br), abs(wf - bf))
        king_proximity = 4 - king_chebyshev

        if score >= 200:
            score += mopup * (2 * CENTER_MANHATTAN_DIST[b_king_sq] + king_proximity)
        elif score <= -200:
            score -= mopup * (2 * CENTER_MANHATTAN_DIST[w_king_sq] + king_proximity)

    return score


def evaluate_tuned(
    state: GameState,
    params: Optional[EvalParams] = None,
    lookup: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> int:
    """Side-to-move (Negamax) tuned evaluation."""
    return state.turn * evaluate_tuned_white(state, params=params, lookup=lookup)

