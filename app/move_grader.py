"""
Fast Chess.com-style 3-Tier Move Quality Grader (Great, Good, Mistake).

Evaluates each played move asynchronously against the engine's best tactical move
at an equal search horizon (default depth=2 + Quiescence) so gameplay remains
completely smooth with zero UI blocking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from engine.eval import DRAW_SCORE, MATE_SCORE
from engine.eval_tuned import evaluate_tuned
from engine.game_state import EMPTY, GameState, Move
from engine.move_order import KillerTable
from engine.search import (
    INFINITY,
    SearchContext,
    alphabeta,
    search_alphabeta_root,
)
from engine.transposition_table import TranspositionTable

MATE_CUTOFF: int = MATE_SCORE - 1000


@dataclass(frozen=True)
class MoveGrade:
    label: str  # Strictly one of: "Great", "Good", "Mistake"
    cp_loss: int
    best_move_uci: Optional[str]
    played_score: int
    best_score: int


def _score_child_move(
    state: GameState,
    move: Move,
    sub_depth: int,
    tt: TranspositionTable,
    killers: KillerTable,
    lower_bound: int = -INFINITY,
) -> int:
    """Evaluate `move` from `state` to `sub_depth` + Quiescence (mover's perspective)."""
    state.make_move(move)
    try:
        if (
            state.zobrist_hash in state.history
            or state.is_fifty_move_draw()
            or state.is_dead_position()
            or state.is_hard_cap_draw()
        ):
            return DRAW_SCORE

        opp_moves = state.generate_legal_moves()
        if not opp_moves:
            return (MATE_SCORE - 1) if state.in_check() else DRAW_SCORE

        ctx = SearchContext(
            use_pst=True,
            use_quiescence=True,
            max_q_depth=6,
            eval_fn=evaluate_tuned,
        )
        beta = INFINITY
        alpha = max(-INFINITY, lower_bound)
        return -alphabeta(
            state,
            sub_depth,
            -beta,
            -alpha,
            1,
            ctx,
            use_move_ordering=True,
            tt=tt,
            killers=killers,
        )
    finally:
        state.unmake_move(move)


def grade_move(
    state_before: GameState,
    played_move: Move,
    search_depth: int = 3,
) -> MoveGrade:
    """
    Classify `played_move` from `state_before` strictly into:
      - 'Great'   : Delivers/finds forced mate, or plays a uniquely decisive/tactical top move
      - 'Good'    : Solid move with low centipawn loss (< 110 cp)
      - 'Mistake' : Drops >= 110 cp, misses a forced mate, or allows a forced mate
    """
    st = state_before.copy()
    legal_moves = st.generate_legal_moves()
    if not legal_moves:
        return MoveGrade("Good", 0, None, 0, 0)

    is_capture = st.board[played_move.to_sq] != EMPTY
    is_promo = played_move.promotion is not None and played_move.promotion != 0

    # Immediate checkmate check
    st.make_move(played_move)
    gives_check = st.in_check()
    opp_has_moves = bool(st.generate_legal_moves())
    st.unmake_move(played_move)

    if not opp_has_moves and gives_check:
        mate_val = MATE_SCORE - 1
        return MoveGrade("Great", 0, played_move.uci(), mate_val, mate_val)

    if len(legal_moves) == 1:
        return MoveGrade("Good", 0, played_move.uci(), 0, 0)

    tt = TranspositionTable(max_entries=16384)
    killers = KillerTable()
    sub_depth = max(1, search_depth - 1)

    root_stats = search_alphabeta_root(
        st,
        depth=search_depth,
        use_move_ordering=True,
        tt=tt,
        killers=killers,
        use_pst=True,
        use_quiescence=True,
        max_q_depth=6,
        eval_fn=evaluate_tuned,
    )
    best_move = root_stats.best_move
    best_score = root_stats.score
    best_uci = best_move.uci() if best_move is not None else played_move.uci()

    if best_move is not None and played_move == best_move:
        played_score = best_score
    else:
        played_score = _score_child_move(st, played_move, sub_depth, tt, killers)

    cp_loss = max(0, best_score - played_score)

    # 1. Forced mate checks
    if best_score > MATE_CUTOFF and played_score <= MATE_CUTOFF:
        return MoveGrade("Mistake", max(cp_loss, 500), best_uci, played_score, best_score)
    if played_score < -MATE_CUTOFF and best_score >= -MATE_CUTOFF:
        return MoveGrade("Mistake", max(cp_loss, 500), best_uci, played_score, best_score)
    if played_score > MATE_CUTOFF:
        return MoveGrade("Great", 0, best_uci, played_score, best_score)

    # 2. Significant centipawn loss -> Mistake
    if cp_loss >= 110:
        return MoveGrade("Mistake", cp_loss, best_uci, played_score, best_score)

    # 3. Top-tier move: distinguish 'Great' (uniquely strong / tactical find) vs 'Good'
    if cp_loss <= 20:
        second_best = -INFINITY
        for m in legal_moves:
            if m == played_move:
                continue
            sc = _score_child_move(st, m, sub_depth, tt, killers, lower_bound=second_best)
            if sc > second_best:
                second_best = sc

        uniqueness_gap = played_score - second_best
        is_tactical = is_capture or is_promo or gives_check
        if uniqueness_gap >= 85 or (is_tactical and uniqueness_gap >= 45):
            return MoveGrade("Great", cp_loss, best_uci, played_score, best_score)

    return MoveGrade("Good", cp_loss, best_uci, played_score, best_score)
