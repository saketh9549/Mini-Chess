"""
Phase 4 (Step 4.4): Full Four-Tier Tournament Matrix & Formal Ablation Suite
Implements the benchmark and ablation protocol specified in rules.txt Section 7:
  - 4-Tier Round-Robin Tournament (100 games / 50 color-swapped pairs per matchup):
      Tier 1: RandomBot
      Tier 2: GreedyBot (1-ply)
      Tier 3: Baseline SearchAgent (d=4, untuned weights, no quiescence)
      Tier 4: TunedSearchAgent (d=4, tuned weights + quiescence search)
  - Formal Search & Evaluation Ablation Table:
      +/- Move Ordering, +/- Transposition Table, +/- Quiescence Search, +/- PST
"""

from __future__ import annotations
import argparse
from pathlib import Path
import random
import sys
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.search_agent import SearchAgent
from agents.tuned_agent import TunedSearchAgent
from benchmarks.arena import TournamentStats, generate_random_opening, run_tournament


def run_formal_ablation_table(depth: int = 4, num_positions: int = 10, seed: int = 42) -> None:
    """
    Evaluate search efficiency across `num_positions` diverse opening/midgame positions
    at fixed `depth` for each ablation configuration (rules.txt Sections 7.6 & 7.7).
    """
    rng = random.Random(seed)
    positions = [generate_random_opening(rng, opening_plies=(2, 4))[0] for _ in range(num_positions)]

    configs: List[Tuple[str, SearchAgent]] = [
        (
            "1. Plain Negamax (Minimax)",
            SearchAgent(max_depth=depth, mode="negamax", use_move_ordering=False, use_tt=False, use_iterative_deepening=False),
        ),
        (
            "2. Alpha-Beta (No Ordering, No TT)",
            SearchAgent(max_depth=depth, mode="alphabeta", use_move_ordering=False, use_tt=False, use_iterative_deepening=False),
        ),
        (
            "3. Alpha-Beta + Move Ordering (No TT)",
            SearchAgent(max_depth=depth, mode="alphabeta", use_move_ordering=True, use_tt=False, use_iterative_deepening=False),
        ),
        (
            "4. Alpha-Beta + TT (No Move Ordering)",
            SearchAgent(max_depth=depth, mode="alphabeta", use_move_ordering=False, use_tt=True, use_iterative_deepening=True),
        ),
        (
            "5. AB + Ordering + TT (Material Only, -PST)",
            SearchAgent(max_depth=depth, mode="alphabeta", use_move_ordering=True, use_tt=True, use_iterative_deepening=True, use_pst=False),
        ),
        (
            "6. AB + Ordering + TT + PST (Baseline d=4)",
            SearchAgent(max_depth=depth, mode="alphabeta", use_move_ordering=True, use_tt=True, use_iterative_deepening=True, use_pst=True),
        ),
        (
            "7. TunedSearchAgent (+Quiescence +TunedEval)",
            TunedSearchAgent(max_depth=depth, use_quiescence=True),
        ),
    ]

    print("\n" + "=" * 96)
    print(f"FORMAL ABLATION TABLE ({num_positions} Tactical/Opening Positions, Fixed Depth = {depth})")
    print("=" * 96)
    print(
        f"{'Configuration':<44} {'Avg Nodes':>10} {'Avg Q-Nodes':>12} "
        f"{'Avg b*':>8} {'Avg Time(s)':>12} {'TT Hit%':>9}"
    )
    print("-" * 96)

    for label, agent in configs:
        total_nodes = 0
        total_q_nodes = 0
        total_b_star = 0.0
        total_time = 0.0
        total_probes = 0
        total_hits = 0

        for pos in positions:
            agent.reset()
            agent.select_move(pos.copy())
            st = agent.last_stats
            total_nodes += st.nodes
            total_q_nodes += st.q_nodes
            total_b_star += st.effective_branching_factor
            total_time += st.elapsed_sec
            total_probes += st.tt_probes
            total_hits += st.tt_hits

        avg_n = total_nodes // num_positions
        avg_qn = total_q_nodes // num_positions
        avg_b = total_b_star / num_positions
        avg_t = total_time / num_positions
        tt_str = f"{(total_hits / total_probes) * 100:.1f}%" if total_probes > 0 else "N/A"

        print(
            f"{label:<44} {avg_n:>10} {avg_qn:>12} "
            f"{avg_b:>8.2f} {avg_t:>12.4f} {tt_str:>9}"
        )
    print("=" * 96)


def run_four_tier_round_robin(num_games: int = 100, depth: int = 4, seed: int = 42) -> List[TournamentStats]:
    """Run the 4-tier round-robin tournament matrix (rules.txt Section 7.4 & 7.5)."""
    matchups = [
        (
            TunedSearchAgent(max_depth=depth, name=f"TunedSearch(d={depth}+Q)"),
            SearchAgent(max_depth=depth, use_quiescence=False, name=f"BaselineSearch(d={depth})"),
        ),
        (
            TunedSearchAgent(max_depth=depth, name=f"TunedSearch(d={depth}+Q)"),
            GreedyAgent(seed=42, name="GreedyBot(1-ply)"),
        ),
        (
            TunedSearchAgent(max_depth=depth, name=f"TunedSearch(d={depth}+Q)"),
            RandomAgent(seed=99, name="RandomBot"),
        ),
        (
            SearchAgent(max_depth=depth, use_quiescence=False, name=f"BaselineSearch(d={depth})"),
            GreedyAgent(seed=42, name="GreedyBot(1-ply)"),
        ),
        (
            SearchAgent(max_depth=depth, use_quiescence=False, name=f"BaselineSearch(d={depth})"),
            RandomAgent(seed=99, name="RandomBot"),
        ),
        (
            GreedyAgent(seed=42, name="GreedyBot(1-ply)"),
            RandomAgent(seed=99, name="RandomBot"),
        ),
    ]

    all_stats: List[TournamentStats] = []
    for a1, a2 in matchups:
        stats = run_tournament(
            a1,
            a2,
            num_games=num_games,
            opening_plies=(2, 4),
            seed=seed,
            verbose=True,
        )
        all_stats.append(stats)

    print("\n" + "=" * 102)
    print(f"FOUR-TIER ROUND-ROBIN SUMMARY MATRIX ({num_games} Games / Matchup, Color-Swapped, Seed={seed})")
    print("=" * 102)
    print(
        f"{'Matchup (Agent 1 vs. Agent 2)':<44} {'W-D-L':<13} "
        f"{'Win%':>7} {'Score%':>8} {'95% Score CI':<17} {'AvgPly':>7} {'Mate%':>7}"
    )
    print("-" * 102)
    for st in all_stats:
        pair_name = f"{st.agent1_name} vs {st.agent2_name}"
        wdl = f"+{st.wins}={st.draws}-{st.losses}"
        ci_l, ci_h = st.confidence_interval_95
        ci_str = f"[{ci_l * 100:.1f}%, {ci_h * 100:.1f}%]"
        mate_pct = st.reasons.get("checkmate", 0) / st.total_games * 100.0
        print(
            f"{pair_name:<44} {wdl:<13} "
            f"{st.win_rate * 100:>6.1f}% {st.score_rate * 100:>7.1f}% "
            f"{ci_str:<17} {st.avg_plies:>7.1f} {mate_pct:>6.1f}%"
        )
    print("=" * 102)
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 Ablation & 4-Tier Tournament Suite")
    parser.add_argument("--games", type=int, default=100, help="Games per matchup (even number)")
    parser.add_argument("--depth", type=int, default=4, help="Fixed search depth for search agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for openings")
    args = parser.parse_args()

    run_formal_ablation_table(depth=args.depth)
    run_four_tier_round_robin(num_games=args.games, depth=args.depth, seed=args.seed)

