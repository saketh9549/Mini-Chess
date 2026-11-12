"""
Phase 2 (Step 2.4): Baseline Agent 2 - 1-Ply GreedyBot
Evaluates all legal successor states at depth 1 and chooses the move
that maximizes the heuristic score (immediately selecting checkmate if found).
"""

from __future__ import annotations
import random
from typing import Optional
from agents.base_agent import Agent
from engine.eval import DRAW_SCORE, MATERIAL_WEIGHTS, evaluate
from engine.game_state import EMPTY, GameState, Move


class GreedyAgent(Agent):
    """
    1-ply greedy lookahead agent:
    - Generates all legal moves
    - Immediately plays any move that delivers checkmate
    - Evaluates non-terminal child states via -evaluate(state)
    - Avoids accidental stalemates/repetitions when ahead by scoring draws as 0
    - Breaks ties with safe destination, safe check, capture preference, and mobility
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        use_pst: bool = True,
        name: str = "GreedyBot",
    ) -> None:
        super().__init__(name=name)
        self.seed = seed
        self.use_pst = use_pst
        self.rng = random.Random(seed)

    def select_move(self, state: GameState) -> Optional[Move]:
        legal_moves = state.generate_legal_moves()
        if not legal_moves:
            return None

        best_move: Optional[Move] = None
        best_key = None

        for move in legal_moves:
            captured_piece = abs(state.board[move.to_sq])
            capture_bonus = MATERIAL_WEIGHTS[captured_piece] if captured_piece != EMPTY else 0

            state.make_move(move)

            opp_moves = state.generate_legal_moves()
            if not opp_moves:
                if state.in_check():
                    state.unmake_move(move)
                    return move
                else:
                    score = DRAW_SCORE
            elif (
                state.zobrist_hash in state.history
                or state.is_fifty_move_draw()
                or state.is_dead_position()
                or state.is_hard_cap_draw()
            ):
                score = DRAW_SCORE
            else:
                score = -evaluate(state, use_pst=self.use_pst)

            is_safe_dest = 0 if state.is_square_attacked(move.to_sq, state.turn) else 1
            gives_safe_check = 1 if (state.in_check() and is_safe_dest) else 0
            opp_mobility_penalty = -len(opp_moves)
            state.unmake_move(move)

            candidate_key = (
                score,
                is_safe_dest,
                gives_safe_check,
                capture_bonus,
                opp_mobility_penalty,
                self.rng.random(),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_move = move

        return best_move

