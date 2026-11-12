"""
Phase 3 & Phase 4: Iterative Deepening Search Agent & Time Manager
Wraps Negamax / Alpha-Beta search in an Iterative Deepening loop (depths 1..D)
with PV move seeding, Zobrist TT persistence, Killer moves, optional Quiescence
Search, custom evaluation hooks, and node-budgeted clock checks.
"""

from __future__ import annotations
import time
from typing import Callable, List, Optional
from agents.base_agent import Agent
from engine.game_state import GameState, Move
from engine.move_order import KillerTable
from engine.search import (
    SearchContext,
    SearchStats,
    SearchTimeout,
    search_alphabeta_root,
    search_negamax_root,
)
from engine.transposition_table import TranspositionTable


class SearchAgent(Agent):
    """
    Configurable multi-ply search agent supporting:
      - Plain Negamax ('negamax') or Alpha-Beta ('alphabeta')
      - Move Ordering (PV, MVV-LVA, Promotions, Killers, PST)
      - Zobrist Transposition Table (with ply-adjusted mate scores)
      - Optional Quiescence Search at depth 0 (use_quiescence=True/False)
      - Optional custom evaluation function (eval_fn)
      - Iterative Deepening (1..max_depth) with node-counted time management
    """

    def __init__(
        self,
        max_depth: int = 4,
        mode: str = "alphabeta",
        use_move_ordering: bool = True,
        use_tt: bool = True,
        use_iterative_deepening: bool = True,
        use_pst: bool = True,
        use_quiescence: bool = False,
        max_q_depth: int = 6,
        eval_fn: Optional[Callable[[GameState], int]] = None,
        time_limit_sec: Optional[float] = None,
        check_interval: int = 2048,
        tt_size: int = 262144,
        name: Optional[str] = None,
    ) -> None:
        if name is None:
            q_tag = "+Q" if use_quiescence else ""
            name = f"SearchAgent(d={max_depth}{q_tag})"
        super().__init__(name=name)
        self.max_depth = max_depth
        self.mode = mode.lower()
        self.use_move_ordering = use_move_ordering
        self.use_tt = use_tt
        self.use_iterative_deepening = use_iterative_deepening
        self.use_pst = use_pst
        self.use_quiescence = use_quiescence
        self.max_q_depth = max_q_depth
        self.eval_fn = eval_fn
        self.time_limit_sec = time_limit_sec
        self.check_interval = check_interval

        self.tt: Optional[TranspositionTable] = (
            TranspositionTable(max_entries=tt_size) if use_tt else None
        )
        self.killers: Optional[KillerTable] = (
            KillerTable() if use_move_ordering else None
        )

        self.last_stats: Optional[SearchStats] = None
        self.depth_history: List[SearchStats] = []

    def reset(self) -> None:
        if self.tt is not None:
            self.tt.clear()
        if self.killers is not None:
            self.killers.clear()
        self.last_stats = None
        self.depth_history.clear()

    def select_move(self, state: GameState) -> Optional[Move]:
        legal_moves = state.generate_legal_moves()
        if not legal_moves:
            return None

        if self.killers is not None:
            self.killers.clear()
        self.depth_history.clear()

        if self.mode == "negamax":
            stats = search_negamax_root(
                state,
                depth=self.max_depth,
                time_limit_sec=self.time_limit_sec,
                use_pst=self.use_pst,
                eval_fn=self.eval_fn,
            )
            self.last_stats = stats
            self.depth_history.append(stats)
            return stats.best_move if stats.best_move is not None else legal_moves[0]

        if not self.use_iterative_deepening:
            stats = search_alphabeta_root(
                state,
                depth=self.max_depth,
                use_move_ordering=self.use_move_ordering,
                tt=self.tt,
                killers=self.killers,
                pv_move=None,
                time_limit_sec=self.time_limit_sec,
                use_pst=self.use_pst,
                use_quiescence=self.use_quiescence,
                max_q_depth=self.max_q_depth,
                eval_fn=self.eval_fn,
            )
            self.last_stats = stats
            self.depth_history.append(stats)
            return stats.best_move if stats.best_move is not None else legal_moves[0]

        ctx = SearchContext(
            time_limit_sec=self.time_limit_sec,
            check_interval=self.check_interval,
            use_pst=self.use_pst,
            use_quiescence=self.use_quiescence,
            max_q_depth=self.max_q_depth,
            eval_fn=self.eval_fn,
        )
        best_move: Move = legal_moves[0]
        best_stats: Optional[SearchStats] = None
        base_undo_len = len(state._undo_stack)

        probes_start = self.tt.probes if self.tt is not None else 0
        hits_start = self.tt.hits if self.tt is not None else 0
        cutoffs_start = self.tt.cutoffs if self.tt is not None else 0

        for d in range(1, self.max_depth + 1):
            try:
                iter_stats = search_alphabeta_root(
                    state,
                    depth=d,
                    use_move_ordering=self.use_move_ordering,
                    tt=self.tt,
                    killers=self.killers,
                    pv_move=best_move,
                    ctx=ctx,
                    use_pst=self.use_pst,
                    use_quiescence=self.use_quiescence,
                    max_q_depth=self.max_q_depth,
                    eval_fn=self.eval_fn,
                )
                if iter_stats.best_move is not None:
                    best_move = iter_stats.best_move
                best_stats = iter_stats
                self.depth_history.append(iter_stats)
            except SearchTimeout:
                while len(state._undo_stack) > base_undo_len:
                    state.unmake_move()
                if best_stats is not None:
                    best_stats.timed_out = True
                break

        total_elapsed = time.perf_counter() - ctx.start_time
        if best_stats is not None:
            best_stats.nodes = ctx.nodes
            best_stats.q_nodes = ctx.q_nodes
            best_stats.elapsed_sec = total_elapsed
            if self.tt is not None:
                best_stats.tt_probes = self.tt.probes - probes_start
                best_stats.tt_hits = self.tt.hits - hits_start
                best_stats.tt_cutoffs = self.tt.cutoffs - cutoffs_start
            self.last_stats = best_stats
        else:
            self.last_stats = SearchStats(
                depth=0,
                nodes=ctx.nodes,
                q_nodes=ctx.q_nodes,
                elapsed_sec=total_elapsed,
                best_move=best_move,
                timed_out=True,
            )

        return best_move

