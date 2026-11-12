"""
Phase 4 (Step 4.3): TunedSearchAgent Integration
Combines Iterative Deepening Alpha-Beta Search, Move Ordering, Zobrist TT,
Quiescence Search (Step 4.1), and the Tuned Heuristic Evaluation (Step 4.2/4.3).
"""

from __future__ import annotations
from typing import Optional
from agents.search_agent import SearchAgent
from engine.eval_tuned import EvalParams, TUNED_PARAMS, build_fast_lookup, evaluate_tuned
from engine.game_state import GameState


class TunedSearchAgent(SearchAgent):
    """
    Phase 4 Flagship Engine:
      - Fixed depth (default d=4) or node-budgeted time control
      - Quiescence Search enabled at depth 0 (resolves tactical captures & promotions)
      - Tuned 5x5 evaluation function (material + scaled PST + passed pawns + mop-up)
      - Zobrist Transposition Table + MVV-LVA / Killer Move Ordering + Iterative Deepening
    """

    def __init__(
        self,
        max_depth: int = 4,
        params: Optional[EvalParams] = None,
        use_quiescence: bool = True,
        max_q_depth: int = 6,
        use_move_ordering: bool = True,
        use_tt: bool = True,
        use_iterative_deepening: bool = True,
        time_limit_sec: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        self.params: EvalParams = params if params is not None else TUNED_PARAMS
        self._lookup = build_fast_lookup(self.params)

        def _bound_eval(state: GameState) -> int:
            return evaluate_tuned(state, params=self.params, lookup=self._lookup)

        if name is None:
            name = f"TunedSearchAgent(d={max_depth})"

        super().__init__(
            max_depth=max_depth,
            mode="alphabeta",
            use_move_ordering=use_move_ordering,
            use_tt=use_tt,
            use_iterative_deepening=use_iterative_deepening,
            use_pst=True,
            use_quiescence=use_quiescence,
            max_q_depth=max_q_depth,
            eval_fn=_bound_eval,
            time_limit_sec=time_limit_sec,
            name=name,
        )

