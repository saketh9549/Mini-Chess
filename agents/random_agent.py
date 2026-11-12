"""
Phase 2 (Step 2.1): Baseline Agent 1 - RandomBot
Selects uniformly at random among all legal moves.
"""

from __future__ import annotations
import random
from typing import Optional
from agents.base_agent import Agent
from engine.game_state import GameState, Move


class RandomAgent(Agent):
    """Zero-intelligence baseline agent that chooses uniformly at random."""

    def __init__(self, seed: Optional[int] = None, name: str = "RandomBot") -> None:
        super().__init__(name=name)
        self.seed = seed
        self.rng = random.Random(seed)

    def select_move(self, state: GameState) -> Optional[Move]:
        legal_moves = state.generate_legal_moves()
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)

