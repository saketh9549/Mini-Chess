"""
Phase 2 (Step 2.1): Base Agent Abstraction
Provides a unified interface for baseline agents (Random, Greedy) and
search-based agents (Minimax, Alpha-Beta, Iterative Deepening, Tuned).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from engine.game_state import GameState, Move


class Agent(ABC):
    """Abstract base class for all Mini Chess agents."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def select_move(self, state: GameState) -> Optional[Move]:
        """
        Choose and return a legal move for `state.turn` in the given `state`,
        or return None if no legal moves exist.
        Must leave `state` unmodified upon return.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook called by the Arena before the start of a new game."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

