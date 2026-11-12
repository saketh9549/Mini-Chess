"""
Phase 3 (Step 3.4): Zobrist Transposition Table (TT)
Caches evaluated positions using 64-bit Zobrist keys with ply-adjusted mate scores
(rules.txt Sections 5.6, 6.2, 7.6).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from engine.eval import MATE_SCORE
from engine.game_state import Move

EXACT: int = 0
LOWERBOUND: int = 1
UPPERBOUND: int = 2

MATE_THRESHOLD: int = MATE_SCORE - 1000


def score_to_tt(score: int, ply: int) -> int:
    """Convert a root-relative mate score into a ply-independent TT score (rules.txt 5.6)."""
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    """Convert a stored TT score back to a root-ply-adjusted score (rules.txt 5.6)."""
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


@dataclass
class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "best_move")
    key: int
    depth: int
    score: int
    flag: int
    best_move: Optional[Move]


class TranspositionTable:
    """Fixed-capacity Zobrist Transposition Table with depth-preferred replacement."""

    def __init__(self, max_entries: int = 262144) -> None:
        self.max_entries = max_entries
        self._table: Dict[int, TTEntry] = {}
        self.probes: int = 0
        self.hits: int = 0
        self.cutoffs: int = 0
        self.stores: int = 0

    def clear(self) -> None:
        self._table.clear()
        self.probes = 0
        self.hits = 0
        self.cutoffs = 0
        self.stores = 0

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.probes) if self.probes > 0 else 0.0

    @property
    def cutoff_rate(self) -> float:
        return (self.cutoffs / self.probes) if self.probes > 0 else 0.0

    def probe_entry(self, key: int) -> Optional[TTEntry]:
        self.probes += 1
        entry = self._table.get(key)
        if entry is not None and entry.key == key:
            self.hits += 1
            return entry
        return None

    def probe_cutoff(
        self,
        key: int,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
    ) -> Tuple[Optional[int], Optional[Move]]:
        entry = self.probe_entry(key)
        if entry is None:
            return None, None

        best_move = entry.best_move
        if entry.depth >= depth:
            tt_score = score_from_tt(entry.score, ply)
            if entry.flag == EXACT:
                self.cutoffs += 1
                return tt_score, best_move
            elif entry.flag == LOWERBOUND and tt_score >= beta:
                self.cutoffs += 1
                return tt_score, best_move
            elif entry.flag == UPPERBOUND and tt_score <= alpha:
                self.cutoffs += 1
                return tt_score, best_move

        return None, best_move

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: int,
        best_move: Optional[Move],
        ply: int,
    ) -> None:
        existing = self._table.get(key)
        if existing is not None:
            if existing.depth > depth and flag != EXACT:
                if existing.best_move is None and best_move is not None:
                    existing.best_move = best_move
                return
        elif len(self._table) >= self.max_entries:
            oldest_key = next(iter(self._table))
            del self._table[oldest_key]

        adj_score = score_to_tt(score, ply)
        self._table[key] = TTEntry(
            key=key,
            depth=depth,
            score=adj_score,
            flag=flag,
            best_move=best_move,
        )
        self.stores += 1

    def __len__(self) -> int:
        return len(self._table)

