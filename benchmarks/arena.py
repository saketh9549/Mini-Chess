"""
Phase 2 (Step 2.2): Automated Head-to-Head Match Runner & Tournament Arena
Implements color-swapped paired matchups, seeded randomized openings (2-4 plies),
complete game-end adjudication (rules.txt Sections 3, 4, 7), and statistical reporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base_agent import Agent
from engine.game_state import BLACK, WHITE, GameState, Move


@dataclass
class GameResult:
    white_name: str
    black_name: str
    white_score: float
    black_score: float
    winner: int                 # +1 (White), -1 (Black), or 0 (Draw)
    reason: str
    plies: int
    opening_uci: List[str] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class TournamentStats:
    agent1_name: str
    agent2_name: str
    total_games: int
    seed: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    white_w: int = 0
    white_d: int = 0
    white_l: int = 0
    black_w: int = 0
    black_d: int = 0
    black_l: int = 0
    total_plies: int = 0
    total_time_sec: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)
    games: List[GameResult] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        return self.total_score / self.total_games if self.total_games > 0 else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def avg_plies(self) -> float:
        return self.total_plies / self.total_games if self.total_games > 0 else 0.0

    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        n = len(self.scores)
        if n <= 1:
            r = self.score_rate
            return (r, r)
        mean = self.score_rate
        variance = sum((x - mean) ** 2 for x in self.scores) / (n - 1)
        std_err = math.sqrt(variance / n)
        margin = 1.96 * std_err
        return (max(0.0, mean - margin), min(1.0, mean + margin))

    def summary(self) -> str:
        ci_low, ci_high = self.confidence_interval_95
        mate_count = self.reasons.get("checkmate", 0)
        mate_rate = (mate_count / self.total_games * 100.0) if self.total_games else 0.0

        lines = [
            "=" * 68,
            f"TOURNAMENT REPORT: {self.agent1_name} vs. {self.agent2_name}",
            "=" * 68,
            f"Total Games      : {self.total_games} ({self.total_games // 2} color-swapped pairs, seed={self.seed})",
            f"{self.agent1_name:<16} : +{self.wins} ={self.draws} -{self.losses} "
            f"(Score: {self.total_score:.1f} / {self.total_games}, {self.score_rate * 100:.1f}%)",
            f"Win Rate         : {self.win_rate * 100:.1f}%  |  95% Score CI: [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]",
            f"  As White       : +{self.white_w} ={self.white_d} -{self.white_l}",
            f"  As Black       : +{self.black_w} ={self.black_d} -{self.black_l}",
            f"Average Length   : {self.avg_plies:.1f} plies  |  Total Time: {self.total_time_sec:.3f}s",
            f"Checkmate Rate   : {mate_rate:.1f}% ({mate_count}/{self.total_games})",
            "-" * 68,
            "Termination Breakdown:",
        ]
        for reason, count in sorted(self.reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = (count / self.total_games * 100.0) if self.total_games else 0.0
            lines.append(f"  - {reason:<24}: {count:>4} ({pct:>5.1f}%)")
        lines.append("=" * 68)
        return "\n".join(lines)


def adjudicate_state(state: GameState, legal_moves: Optional[List[Move]] = None) -> Optional[Tuple[int, str]]:
    """Check if `state` is terminal according to rules.txt Sections 3 & 4."""
    if legal_moves is None:
        legal_moves = state.generate_legal_moves()

    if not legal_moves:
        if state.in_check():
            return (-state.turn, "checkmate")
        return (0, "stalemate")

    if state.is_threefold_repetition():
        return (0, "threefold_repetition")

    if state.is_fifty_move_draw():
        return (0, "fifty_move_rule")

    if state.is_dead_position():
        return (0, "dead_position")

    if state.is_hard_cap_draw():
        return (0, "hard_cap_300_plies")

    return None


def play_game(
    white_agent: Agent,
    black_agent: Agent,
    initial_state: Optional[GameState] = None,
    opening_uci: Optional[List[str]] = None,
) -> GameResult:
    """Play a single complete game between `white_agent` and `black_agent`."""
    state = initial_state.copy() if initial_state is not None else GameState()
    white_agent.reset()
    black_agent.reset()

    t0 = time.perf_counter()

    while True:
        legal_moves = state.generate_legal_moves()
        verdict = adjudicate_state(state, legal_moves)
        if verdict is not None:
            winner, reason = verdict
            break

        active_agent = white_agent if state.turn == WHITE else black_agent
        chosen_move = active_agent.select_move(state)

        if chosen_move is None or chosen_move not in legal_moves:
            raise ValueError(
                f"Agent {active_agent.name!r} returned illegal move {chosen_move!r} "
                f"on ply {len(state.history)}."
            )

        state.make_move(chosen_move)

    elapsed = time.perf_counter() - t0
    if winner == WHITE:
        w_score, b_score = 1.0, 0.0
    elif winner == BLACK:
        w_score, b_score = 0.0, 1.0
    else:
        w_score, b_score = 0.5, 0.5

    return GameResult(
        white_name=white_agent.name,
        black_name=black_agent.name,
        white_score=w_score,
        black_score=b_score,
        winner=winner,
        reason=reason,
        plies=len(state.history),
        opening_uci=list(opening_uci) if opening_uci else [],
        duration_sec=elapsed,
    )


def generate_random_opening(
    rng: random.Random,
    opening_plies: Union[int, Tuple[int, int]] = (2, 4),
) -> Tuple[GameState, List[str]]:
    """Generate a single non-terminal opening position by playing `k` random plies."""
    if isinstance(opening_plies, tuple):
        min_p, max_p = opening_plies
    else:
        min_p = max_p = int(opening_plies)

    while True:
        k = rng.randint(min_p, max_p)
        state = GameState()
        uci_list: List[str] = []
        valid = True
        for _ in range(k):
            moves = state.generate_legal_moves()
            if not moves:
                valid = False
                break
            m = rng.choice(moves)
            state.make_move(m)
            uci_list.append(m.uci())
            if adjudicate_state(state) is not None:
                valid = False
                break
        if valid:
            return state, uci_list


def run_tournament(
    agent1: Agent,
    agent2: Agent,
    num_games: int = 100,
    opening_plies: Union[int, Tuple[int, int]] = 2,
    seed: int = 42,
    verbose: bool = True,
) -> TournamentStats:
    """Run a head-to-head tournament of `num_games` games in color-swapped pairs."""
    if num_games % 2 != 0:
        raise ValueError(f"num_games must be even for color-swapped pairs (got {num_games}).")

    rng = random.Random(seed)
    num_pairs = num_games // 2

    stats = TournamentStats(
        agent1_name=agent1.name,
        agent2_name=agent2.name,
        total_games=num_games,
        seed=seed,
    )

    for _ in range(num_pairs):
        opening_state, opening_uci = generate_random_opening(rng, opening_plies=opening_plies)

        res_a = play_game(agent1, agent2, initial_state=opening_state, opening_uci=opening_uci)
        _record_result(stats, res_a, agent1_is_white=True)

        res_b = play_game(agent2, agent1, initial_state=opening_state, opening_uci=opening_uci)
        _record_result(stats, res_b, agent1_is_white=False)

    if verbose:
        print(stats.summary())

    return stats


def _record_result(stats: TournamentStats, res: GameResult, agent1_is_white: bool) -> None:
    stats.games.append(res)
    stats.total_plies += res.plies
    stats.total_time_sec += res.duration_sec
    stats.reasons[res.reason] = stats.reasons.get(res.reason, 0) + 1

    if agent1_is_white:
        score = res.white_score
        if res.winner == WHITE:
            stats.wins += 1
            stats.white_w += 1
        elif res.winner == BLACK:
            stats.losses += 1
            stats.white_l += 1
        else:
            stats.draws += 1
            stats.white_d += 1
    else:
        score = res.black_score
        if res.winner == BLACK:
            stats.wins += 1
            stats.black_w += 1
        elif res.winner == WHITE:
            stats.losses += 1
            stats.black_l += 1
        else:
            stats.draws += 1
            stats.black_d += 1

    stats.scores.append(score)


if __name__ == "__main__":
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent

    print("Running Sanity Check: RandomBot vs. RandomBot (50 games)...")
    r1 = RandomAgent(seed=101, name="RandomBot-A")
    r2 = RandomAgent(seed=202, name="RandomBot-B")
    run_tournament(r1, r2, num_games=50, opening_plies=(2, 4), seed=42, verbose=True)

    print("\nRunning Phase 2 Benchmark: GreedyBot vs. RandomBot (100 games)...")
    greedy = GreedyAgent(seed=303, name="GreedyBot")
    rand_bot = RandomAgent(seed=404, name="RandomBot")
    run_tournament(greedy, rand_bot, num_games=100, opening_plies=2, seed=42, verbose=True)

