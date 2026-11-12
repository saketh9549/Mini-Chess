"""
Phase 3 & Phase 4: Adversarial Search Engine with Quiescence Search
Implements:
  - Plain Negamax Minimax with fixed depth, mate-distance scoring, and repetition detection
  - Alpha-Beta Pruning with Move Ordering (PV, MVV-LVA, Killers, PST) and Zobrist TT
  - Quiescence Search at depth 0 (Stand-pat, capture/promotion search, Delta pruning)
  - Custom evaluation function hook (for Phase 4 Tuned Evaluation)
  - Node-budgeted periodic clock checks (rules.txt Section 6.6)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional
from engine.eval import DRAW_SCORE, MATE_SCORE, MATERIAL_WEIGHTS, evaluate
from engine.game_state import EMPTY, QUEEN, GameState, Move
from engine.move_order import KillerTable, order_moves
from engine.transposition_table import EXACT, LOWERBOUND, UPPERBOUND, TranspositionTable

INFINITY: int = 10_000_000
DELTA_MARGIN: int = 150


class SearchTimeout(Exception):
    """Raised when the wall-clock time limit expires during mid-depth search."""
    pass


@dataclass
class SearchStats:
    """Instrumentation metrics recorded during a search call (rules.txt Section 7.6)."""
    depth: int = 0
    nodes: int = 0
    q_nodes: int = 0
    elapsed_sec: float = 0.0
    score: int = 0
    best_move: Optional[Move] = None
    pv: List[str] = field(default_factory=list)
    tt_probes: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    timed_out: bool = False

    @property
    def effective_branching_factor(self) -> float:
        """Compute b* = Nodes^(1 / depth)."""
        if self.depth <= 0 or self.nodes <= 0:
            return 0.0
        return float(self.nodes) ** (1.0 / self.depth)

    @property
    def nps(self) -> int:
        return int(self.nodes / self.elapsed_sec) if self.elapsed_sec > 0 else 0

    @property
    def tt_hit_rate(self) -> float:
        return (self.tt_hits / self.tt_probes) if self.tt_probes > 0 else 0.0


class SearchContext:
    """Mutable context shared across recursive calls within a single search invocation."""

    def __init__(
        self,
        time_limit_sec: Optional[float] = None,
        check_interval: int = 2048,
        use_pst: bool = True,
        use_quiescence: bool = False,
        max_q_depth: int = 6,
        eval_fn: Optional[Callable[[GameState], int]] = None,
    ) -> None:
        self.start_time: float = time.perf_counter()
        self.time_limit_sec: Optional[float] = time_limit_sec
        self.check_interval: int = check_interval
        self.use_pst: bool = use_pst
        self.use_quiescence: bool = use_quiescence
        self.max_q_depth: int = max_q_depth
        self._custom_eval_fn: Optional[Callable[[GameState], int]] = eval_fn
        self.nodes: int = 0
        self.q_nodes: int = 0

    def evaluate_leaf(self, state: GameState) -> int:
        if self._custom_eval_fn is not None:
            return self._custom_eval_fn(state)
        return evaluate(state, use_pst=self.use_pst)

    def step_node(self, is_quiescence: bool = False) -> None:
        self.nodes += 1
        if is_quiescence:
            self.q_nodes += 1
        if (
            self.time_limit_sec is not None
            and (self.nodes & (self.check_interval - 1)) == 0
        ):
            if (time.perf_counter() - self.start_time) >= self.time_limit_sec:
                raise SearchTimeout()


def generate_tactical_moves(state: GameState, legal_moves: Optional[List[Move]] = None) -> List[Move]:
    """Filter legal moves to only captures and queen promotions, ordered by MVV-LVA."""
    if legal_moves is None:
        legal_moves = state.generate_legal_moves()
    board = state.board
    tactical = [
        m for m in legal_moves
        if board[m.to_sq] != EMPTY or m.promotion == QUEEN
    ]
    return order_moves(state, tactical)


def quiescence(
    state: GameState,
    alpha: int,
    beta: int,
    ply: int,
    q_depth: int,
    ctx: SearchContext,
) -> int:
    """Quiescence search to resolve tactical captures/promotions beyond depth 0."""
    ctx.step_node(is_quiescence=True)

    if ply > 0 and (
        state.zobrist_hash in state.history
        or state.is_fifty_move_draw()
        or state.is_dead_position()
        or state.is_hard_cap_draw()
    ):
        return DRAW_SCORE

    legal_moves = state.generate_legal_moves()
    if not legal_moves:
        if state.in_check():
            return -(MATE_SCORE - ply)
        return DRAW_SCORE

    in_chk = state.in_check()
    if not in_chk:
        stand_pat = ctx.evaluate_leaf(state)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        stand_pat = -INFINITY

    if q_depth >= ctx.max_q_depth:
        return stand_pat if not in_chk else ctx.evaluate_leaf(state)

    if in_chk:
        candidates = order_moves(state, legal_moves)
    else:
        candidates = generate_tactical_moves(state, legal_moves)

    board = state.board
    for move in candidates:
        if not in_chk:
            captured_val = MATERIAL_WEIGHTS[abs(board[move.to_sq])]
            promo_val = (MATERIAL_WEIGHTS[move.promotion] - 100) if move.promotion else 0
            if stand_pat + captured_val + promo_val + DELTA_MARGIN < alpha:
                continue

        state.make_move(move)
        score = -quiescence(state, -beta, -alpha, ply + 1, q_depth + 1, ctx)
        state.unmake_move(move)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(
    state: GameState,
    depth: int,
    ply: int,
    ctx: SearchContext,
) -> int:
    """Plain recursive Negamax search without pruning."""
    ctx.step_node()

    if ply > 0 and (
        state.zobrist_hash in state.history
        or state.is_fifty_move_draw()
        or state.is_dead_position()
        or state.is_hard_cap_draw()
    ):
        return DRAW_SCORE

    legal_moves = state.generate_legal_moves()
    if not legal_moves:
        if state.in_check():
            return -(MATE_SCORE - ply)
        return DRAW_SCORE

    if depth <= 0:
        return ctx.evaluate_leaf(state)

    max_eval = -INFINITY
    for move in legal_moves:
        state.make_move(move)
        score = -negamax(state, depth - 1, ply + 1, ctx)
        state.unmake_move(move)
        if score > max_eval:
            max_eval = score

    return max_eval


def search_negamax_root(
    state: GameState,
    depth: int,
    time_limit_sec: Optional[float] = None,
    use_pst: bool = True,
    eval_fn: Optional[Callable[[GameState], int]] = None,
) -> SearchStats:
    """Run Plain Negamax at the root position to fixed `depth` and return SearchStats."""
    ctx = SearchContext(time_limit_sec=time_limit_sec, use_pst=use_pst, eval_fn=eval_fn)
    legal_moves = state.generate_legal_moves()
    if not legal_moves:
        score = -(MATE_SCORE) if state.in_check() else DRAW_SCORE
        return SearchStats(depth=depth, nodes=1, score=score, best_move=None)

    best_score = -INFINITY
    best_move: Optional[Move] = legal_moves[0]
    ctx.step_node()

    for move in legal_moves:
        state.make_move(move)
        score = -negamax(state, depth - 1, 1, ctx)
        state.unmake_move(move)
        if score > best_score:
            best_score = score
            best_move = move

    elapsed = time.perf_counter() - ctx.start_time
    return SearchStats(
        depth=depth,
        nodes=ctx.nodes,
        elapsed_sec=elapsed,
        score=best_score,
        best_move=best_move,
        pv=[best_move.uci()] if best_move else [],
    )


def alphabeta(
    state: GameState,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    ctx: SearchContext,
    use_move_ordering: bool = True,
    tt: Optional[TranspositionTable] = None,
    killers: Optional[KillerTable] = None,
) -> int:
    """Negamax Alpha-Beta search with Move Ordering, Zobrist TT, and Quiescence Search."""
    if depth <= 0 and ctx.use_quiescence:
        return quiescence(state, alpha, beta, ply, 0, ctx)

    ctx.step_node()

    if ply > 0 and (
        state.zobrist_hash in state.history
        or state.is_fifty_move_draw()
        or state.is_dead_position()
        or state.is_hard_cap_draw()
    ):
        return DRAW_SCORE

    orig_alpha = alpha
    z_key = state.zobrist_hash
    tt_move: Optional[Move] = None

    if tt is not None and ply > 0:
        cutoff_score, tt_move = tt.probe_cutoff(z_key, depth, alpha, beta, ply)
        if cutoff_score is not None:
            return cutoff_score

    legal_moves = state.generate_legal_moves()
    if not legal_moves:
        if state.in_check():
            return -(MATE_SCORE - ply)
        return DRAW_SCORE

    if depth <= 0:
        return ctx.evaluate_leaf(state)

    if use_move_ordering:
        moves = order_moves(state, legal_moves, tt_move=tt_move, killers=killers, ply=ply)
    else:
        moves = legal_moves

    best_score = -INFINITY
    best_move: Optional[Move] = moves[0]

    for move in moves:
        is_capture = state.board[move.to_sq] != EMPTY
        state.make_move(move)
        score = -alphabeta(
            state,
            depth - 1,
            -beta,
            -alpha,
            ply + 1,
            ctx,
            use_move_ordering=use_move_ordering,
            tt=tt,
            killers=killers,
        )
        state.unmake_move(move)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            if killers is not None and not is_capture and move.promotion == 0:
                killers.record(ply, move)
            break

    if tt is not None:
        if best_score <= orig_alpha:
            flag = UPPERBOUND
        elif best_score >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(z_key, depth, best_score, flag, best_move, ply)

    return best_score


def search_alphabeta_root(
    state: GameState,
    depth: int,
    use_move_ordering: bool = True,
    tt: Optional[TranspositionTable] = None,
    killers: Optional[KillerTable] = None,
    pv_move: Optional[Move] = None,
    time_limit_sec: Optional[float] = None,
    ctx: Optional[SearchContext] = None,
    use_pst: bool = True,
    use_quiescence: bool = False,
    max_q_depth: int = 6,
    eval_fn: Optional[Callable[[GameState], int]] = None,
) -> SearchStats:
    """Run Alpha-Beta search from the root position to fixed `depth`."""
    if ctx is None:
        ctx = SearchContext(
            time_limit_sec=time_limit_sec,
            use_pst=use_pst,
            use_quiescence=use_quiescence,
            max_q_depth=max_q_depth,
            eval_fn=eval_fn,
        )

    probes_before = tt.probes if tt is not None else 0
    hits_before = tt.hits if tt is not None else 0
    cutoffs_before = tt.cutoffs if tt is not None else 0

    legal_moves = state.generate_legal_moves()
    if not legal_moves:
        score = -(MATE_SCORE) if state.in_check() else DRAW_SCORE
        return SearchStats(depth=depth, nodes=1, score=score, best_move=None)

    z_key = state.zobrist_hash
    tt_root_move = pv_move
    if tt_root_move is None and tt is not None:
        entry = tt.probe_entry(z_key)
        if entry is not None and entry.best_move is not None:
            tt_root_move = entry.best_move

    if use_move_ordering:
        moves = order_moves(state, legal_moves, tt_move=tt_root_move, killers=killers, ply=0)
    elif tt_root_move is not None and tt_root_move in legal_moves:
        moves = [tt_root_move] + [m for m in legal_moves if m != tt_root_move]
    else:
        moves = legal_moves

    alpha = -INFINITY
    beta = INFINITY
    best_score = -INFINITY
    best_move: Optional[Move] = moves[0]

    ctx.step_node()

    for move in moves:
        state.make_move(move)
        score = -alphabeta(
            state,
            depth - 1,
            -beta,
            -alpha,
            1,
            ctx,
            use_move_ordering=use_move_ordering,
            tt=tt,
            killers=killers,
        )
        state.unmake_move(move)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    if tt is not None:
        tt.store(z_key, depth, best_score, EXACT, best_move, ply=0)

    elapsed = time.perf_counter() - ctx.start_time
    pv_line = extract_pv(state, best_move, tt, max_len=depth)

    return SearchStats(
        depth=depth,
        nodes=ctx.nodes,
        q_nodes=ctx.q_nodes,
        elapsed_sec=elapsed,
        score=best_score,
        best_move=best_move,
        pv=pv_line,
        tt_probes=(tt.probes - probes_before) if tt is not None else 0,
        tt_hits=(tt.hits - hits_before) if tt is not None else 0,
        tt_cutoffs=(tt.cutoffs - cutoffs_before) if tt is not None else 0,
    )


def extract_pv(
    state: GameState,
    first_move: Optional[Move],
    tt: Optional[TranspositionTable],
    max_len: int = 4,
) -> List[str]:
    """Extract the Principal Variation (PV) move sequence from the Transposition Table."""
    if first_move is None:
        return []
    if tt is None or max_len <= 1:
        return [first_move.uci()]

    pv: List[str] = []
    made_moves: List[Move] = []
    seen_hashes = {state.zobrist_hash}

    curr_move: Optional[Move] = first_move
    for _ in range(max_len):
        if curr_move is None:
            break
        legal = state.generate_legal_moves()
        if curr_move not in legal:
            break
        pv.append(curr_move.uci())
        state.make_move(curr_move)
        made_moves.append(curr_move)

        if state.zobrist_hash in seen_hashes:
            break
        seen_hashes.add(state.zobrist_hash)

        entry = tt._table.get(state.zobrist_hash)
        curr_move = entry.best_move if entry is not None else None

    for m in reversed(made_moves):
        state.unmake_move(m)

    return pv

