"""
Automated Comparison Graph Generator for the 5 Mini Chess AI Algorithms.
Generates 6 publication-quality PNG charts in `benchmarks/plots/`:
  1. graph1_overall_win_and_score_rates.png  (5-Algorithm Win Rate & Score Rate Bar Chart)
  2. graph2_head_to_head_heatmap.png         (5x5 Round-Robin Score Rate Matrix)
  3. graph3_nodes_vs_depth_log.png           (Nodes Expanded vs. Depth d=1..4, Log Scale)
  4. graph4_time_vs_depth.png                (Execution Time ms vs. Depth d=1..4)
  5. graph5_branching_factor_comparison.png  (Effective Branching Factor b* at Depth 4)
  6. graph6_game_length_and_checkmate.png    (Avg Game Length Plies & Checkmate % per Matchup)

Usage:
    python benchmarks/plot_graphs.py
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_SHORT = [
    "RandomBot\n(d=0)",
    "GreedyBot\n(1-Ply)",
    "Plain Minimax\n(Negamax d=3)",
    "Baseline AB\n(d=4 +Order+TT)",
    "TunedSearch\n(d=4 +Quiescence)",
]


def plot_1_win_and_score_rates() -> None:
    win_rates = [0.0, 25.1, 51.9, 69.1, 90.4]
    score_rates = [0.6, 26.1, 54.1, 73.5, 93.6]

    x = np.arange(len(ALGO_SHORT))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    b1 = ax.bar(x - width / 2, win_rates, width, label="Win Rate (%)", color="#3b82f6")
    b2 = ax.bar(x + width / 2, score_rates, width, label="Overall Score Rate (%)", color="#10b981")

    ax.set_title("Overall Tournament Performance Across the 5 AI Algorithms (400 Games / Bot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_SHORT, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph1_overall_win_and_score_rates.png")
    plt.close(fig)


def plot_2_head_to_head_matrix() -> None:
    labels = [
        "TunedSearch (d=4+Q)",
        "Baseline AB (d=4)",
        "Plain Minimax (d=3)",
        "GreedyBot (1-Ply)",
        "RandomBot",
    ]
    # Row i vs Col j Score Rate (%) for Row i
    matrix = np.array([
        [50.0, 79.0, 96.0, 99.0, 100.0],
        [21.0, 50.0, 75.5, 98.0, 100.0],
        [4.0,  24.5, 50.0, 97.0, 100.0],
        [1.0,   2.0,  3.0, 50.0,  99.0],
        [0.0,   0.0,  0.0,  1.0,  50.0],
    ])

    fig, ax = plt.subplots(figsize=(8.5, 6.8), dpi=150)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title("5×5 Round-Robin Head-to-Head Score Rate Matrix (Row vs. Column %)", fontsize=12.5, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = "—" if i == j else f"{matrix[i, j]:.1f}%"
            color = "white" if matrix[i, j] > 60 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Score Rate (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph2_head_to_head_heatmap.png")
    plt.close(fig)


def plot_3_nodes_vs_depth() -> None:
    depths = [1, 2, 3, 4]
    minimax_nodes = [11.8, 112.8, 1260.3, 13970.3]
    ab_unord_nodes = [11.8, 77.3, 488.9, 2153.9]
    ab_order_nodes = [11.8, 31.7, 166.8, 536.4]
    baseline_ab_nodes = [11.8, 45.1, 212.0, 666.2]
    tuned_tree_nodes = [1.0, 12.8, 50.2, 211.9]
    tuned_total_nodes = [271.3, 789.9, 1592.5, 3507.3]

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    ax.plot(depths, minimax_nodes, "o-", linewidth=2.2, label="Plain Minimax (Negamax)", color="#ef4444")
    ax.plot(depths, ab_unord_nodes, "s--", linewidth=2.0, label="Alpha-Beta (Unordered)", color="#f59e0b")
    ax.plot(depths, baseline_ab_nodes, "^-", linewidth=2.2, label="Baseline SearchAgent (AB + Order + TT)", color="#3b82f6")
    ax.plot(depths, ab_order_nodes, "d-", linewidth=2.0, label="Alpha-Beta + Move Ordering (Fixed)", color="#8b5cf6")
    ax.plot(depths, tuned_total_nodes, "P-", linewidth=2.2, label="TunedSearchAgent (Total incl. Quiescence)", color="#10b981")
    ax.plot(depths, tuned_tree_nodes, "P:", linewidth=1.8, label="TunedSearchAgent (Main Tree d≤4 Nodes Only)", color="#059669")

    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xticklabels([f"Depth {d}" for d in depths], fontsize=10)
    ax.set_ylabel("Average Nodes Expanded (Log Scale)", fontsize=11)
    ax.set_title("Search Tree Node Expansion Across Depths (d = 1 to 4)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph3_nodes_vs_depth_log.png")
    plt.close(fig)


def plot_4_time_vs_depth() -> None:
    depths = [1, 2, 3, 4]
    minimax_ms = [0.55, 5.09, 60.24, 1005.20]
    ab_unord_ms = [0.47, 3.17, 23.48, 146.94]
    ab_order_ms = [0.51, 1.33, 8.11, 40.95]
    baseline_ms = [0.57, 1.91, 10.22, 48.82]
    tuned_ms = [15.51, 43.85, 90.66, 303.77]

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    ax.plot(depths, minimax_ms, "o-", linewidth=2.2, label="Plain Minimax (Negamax)", color="#ef4444")
    ax.plot(depths, ab_unord_ms, "s--", linewidth=2.0, label="Alpha-Beta (Unordered)", color="#f59e0b")
    ax.plot(depths, tuned_ms, "P-", linewidth=2.2, label="TunedSearchAgent (+Quiescence)", color="#10b981")
    ax.plot(depths, baseline_ms, "^-", linewidth=2.2, label="Baseline SearchAgent (AB + Order + TT)", color="#3b82f6")
    ax.plot(depths, ab_order_ms, "d-", linewidth=2.0, label="Alpha-Beta + Move Ordering", color="#8b5cf6")

    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xticklabels([f"Depth {d}" for d in depths], fontsize=10)
    ax.set_ylabel("Average Time per Move in ms (Log Scale)", fontsize=11)
    ax.set_title("Execution Time per Move vs. Search Depth (d = 1 to 4)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph4_time_vs_depth.png")
    plt.close(fig)


def plot_5_branching_factor() -> None:
    configs = [
        "Plain Minimax\n(Negamax)",
        "Alpha-Beta\n(Unordered)",
        "Alpha-Beta\n+ TT Only",
        "Baseline AB\n(+Order+TT+IDS)",
        "Alpha-Beta\n+ Move Order",
        "TunedSearch\n(Main Tree d=4)",
    ]
    b_stars = [10.72, 6.76, 6.12, 5.01, 4.69, 3.82]
    colors = ["#ef4444", "#f59e0b", "#eab308", "#3b82f6", "#8b5cf6", "#10b981"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    bars = ax.bar(configs, b_stars, color=colors, width=0.52)
    ax.axhline(y=10.72 ** 0.5, color="#111827", linestyle="--", linewidth=1.5, label="Theoretical Optimal √b ≈ 3.27")

    ax.set_ylabel("Effective Branching Factor (b* = N^(1/4))", fontsize=11)
    ax.set_title("Effective Branching Factor (b*) Reduction at Depth 4", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 12.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"b* = {h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph5_branching_factor_comparison.png")
    plt.close(fig)


def plot_6_game_length_and_checkmates() -> None:
    matchups = [
        "Tuned(d=4+Q)\nvs Random",
        "Baseline(d=4)\nvs Random",
        "Minimax(d=3)\nvs Random",
        "Tuned(d=4+Q)\nvs Greedy",
        "Greedy(1-Ply)\nvs Random",
        "Minimax(d=3)\nvs Greedy",
        "Baseline(d=4)\nvs Greedy",
        "Tuned(d=4+Q)\nvs Minimax(d=3)",
        "Tuned(d=4+Q)\nvs Baseline(d=4)",
        "Baseline(d=4)\nvs Minimax(d=3)",
    ]
    avg_plies = [16.8, 17.7, 17.9, 20.4, 21.6, 24.9, 26.1, 40.5, 42.9, 49.0]
    mate_rates = [100.0, 100.0, 100.0, 98.0, 98.0, 100.0, 96.0, 92.0, 84.0, 85.0]

    x = np.arange(len(matchups))
    fig, ax1 = plt.subplots(figsize=(11.5, 5.5), dpi=150)

    bars = ax1.bar(x, avg_plies, width=0.5, color="#6366f1", label="Avg Game Length (Plies)")
    ax1.set_ylabel("Average Game Length (Plies)", color="#4338ca", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(matchups, rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylim(0, 60)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(x, mate_rates, "o-", color="#10b981", linewidth=2.2, markersize=7, label="Checkmate Rate (%)")
    ax2.set_ylabel("Checkmate Termination Rate (%)", color="#059669", fontsize=11, fontweight="bold")
    ax2.set_ylim(50, 105)

    plt.title("Average Game Length (Plies) & Checkmate Rate Across All 10 Matchups", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "graph6_game_length_and_checkmate.png")
    plt.close(fig)


def main() -> None:
    plot_1_win_and_score_rates()
    plot_2_head_to_head_matrix()
    plot_3_nodes_vs_depth()
    plot_4_time_vs_depth()
    plot_5_branching_factor()
    plot_6_game_length_and_checkmates()
    print(f"All 6 comparison graphs successfully saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

