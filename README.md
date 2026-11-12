# Gardner's 5×5 Mini Chess — AI Engine, Desktop Studio & Benchmark Suite

A complete 4-Phase Artificial Intelligence project implementing **Gardner's 5×5 Minichess** from scratch in Python (zero external dependencies required to run the engine and GUI).

---

## 1. Project Structure

```text
Mini Chess/
├── main.py                              # Unified Entry Point (GUI default, --cli, --test, --tournament, etc.)
├── README.md                            # Quick-Start Guide, Architecture & Commands
├── PROJECT_REPORT.md                    # Complete Technical Report, Ablations & 5-Algorithm Data
├── rules.txt                            # Formal Rules & Coursework Specification
│
├── engine/                              # Core Game State, Search & Evaluation Engine
│   ├── __init__.py
│   ├── game_state.py                    # 1D board[25], 64-bit Zobrist hashing, make/unmake, legal move generator
│   ├── perft.py                         # Perft & Perft-Divide tree verification (depths 1..5)
│   ├── eval.py                          # Baseline centipawn material + 5×5 Piece-Square Tables (PST)
│   ├── eval_tuned.py                    # Tuned weights, passed pawns, rank-4 threats & endgame mop-up
│   ├── move_order.py                    # PV/TT > MVV-LVA captures > Promotions > Killer moves > PST delta
│   ├── transposition_table.py           # 64-bit Zobrist TT with ply-relative mate score normalization
│   └── search.py                        # Plain Negamax, Alpha-Beta pruning & Quiescence Search
│
├── agents/                              # 5 AI Algorithms / Pluggable Agent Hierarchy
│   ├── __init__.py
│   ├── base_agent.py                    # Abstract Agent interface
│   ├── random_agent.py                  # Algo 1: Seeded uniform-random baseline (RandomBot)
│   ├── greedy_agent.py                  # Algo 2: 1-ply static evaluation lookahead (GreedyBot)
│   ├── search_agent.py                  # Algo 3 & 4: Plain Minimax (Negamax) & Alpha-Beta SearchAgent
│   └── tuned_agent.py                   # Algo 5: Flagship TunedSearchAgent (d=4 + Quiescence + TunedEval)
│
├── app/                                 # User Interfaces (Desktop GUI & Terminal CLI)
│   ├── __init__.py
│   ├── gui.py                           # Native Tkinter Desktop GUI App (Step-by-step AI, Eval Bar, Telemetry)
│   └── cli.py                           # Interactive ASCII Terminal CLI
│
├── benchmarks/                          # Automated Match Arena, Tournaments, Tuning & Graph Data
│   ├── __init__.py
│   ├── arena.py                         # Color-swapped match runner, opening book & 95% CI reporting
│   ├── run_tournaments.py               # Round-robin tournament matrix & 7-config ablation study
│   ├── tune.py                          # Genetic Algorithm (GA), Simulated Annealing (SA) & Hill Climbing
│   ├── plot_graphs.py                   # Generates 6 comparison PNG graphs in benchmarks/plots/
│   └── data/                            # Ready-to-plot CSV datasets for all 5 algorithms
│       ├── 1_five_algorithms_summary.csv
│       ├── 2_round_robin_tournament_matrix.csv
│       ├── 3_depth_scaling_d1_to_d4.csv
│       └── 4_search_ablation_depth4.csv
│
└── tests/                               # Phase-by-Phase Automated Unit & Regression Test Suites
    ├── __init__.py
    ├── run_all_tests.py                 # Unified test runner (all phases or --phase 1..4)
    ├── test_phase1.py                   # Phase 1: Board, Perft(1..4), Zobrist & Make/Unmake tests
    ├── test_phase2.py                   # Phase 2: Eval symmetry, PST flip & Greedy vs Random (100 games)
    ├── test_phase3.py                   # Phase 3: Minimax == AlphaBeta regression, TT mate & Search vs Greedy
    └── test_phase4.py                   # Phase 4: Quiescence horizon test, Ablation & Tuned vs Baseline
```

---

## 2. Quick-Start Commands

### Launch the Desktop GUI App (Default)
```powershell
python main.py
# or directly:
python app/gui.py
```
**GUI Features**:
* **Play vs. Any AI or Watch AI vs. AI**: Choose `Human` or any of the 5 AI algorithms for White and Black.
* **Step-by-Step AI Visualization**:
  * **Directional Move Arrow & Step Banner**: Shows the exact move (`Step 5: White (Tuned) played ♞ Knight b1 → c3`).
  * **`AI Step Delay` Slider (`0.2s`–`2.0s`)**: Controls AI vs. AI playback speed so moves are easy to follow.
  * **`⏭ Step AI Move`, `◀ Prev Step`, `Next Step ▶`, `Pause AI`**: Step through AI vs. AI one move at a time or replay history.
* **Explicit Winner & Loser Banner**: Announces the exact winner, loser, and termination reason when a match finishes.

### Launch the Interactive Terminal CLI
```powershell
python main.py --cli
```

### Run Unit, Regression & Benchmark Tests (Phases 1–4)
```powershell
# Run all tests across Phases 1-4:
python main.py --test

# Or run a specific phase:
python tests/run_all_tests.py --phase 1
python tests/run_all_tests.py --phase 2
python tests/run_all_tests.py --phase 3
python tests/run_all_tests.py --phase 4
```

### Run Perft, Ablations, Tournaments & Graph Plotting
```powershell
# Phase 1 Perft verification (depths 1..5):
python main.py --perft

# Phase 3 & 4 Formal Search Ablation Table:
python main.py --ablation

# Full Round-Robin Tournament Matrix:
python main.py --tournament --games 100

# Heuristic Parameter Tuning (GA / SA / Hill Climbing):
python benchmarks/tune.py --method ga --generations 5 --pop-size 8 --games 6

# Generate all 6 Comparison Charts (PNGs saved to benchmarks/plots/):
python benchmarks/plot_graphs.py
```

---

## 3. Summary of the 5 AI Algorithms

| # | Algorithm | Search Depth | Key Techniques | Overall Win Rate | Overall Score Rate | Avg Nodes / Move | Effective $b^*$ |
| :-: | :--- | :-: | :--- | :-: | :-: | :-: | :-: |
| **1** | **RandomBot** | `0` | Uniform random legal move selection | `0.0%` | `0.6%` | `1.0` | `1.00` |
| **2** | **GreedyBot** | `1` | 1-ply static evaluation lookahead | `25.1%` | `26.1%` | `11.8` | `11.80` |
| **3** | **Plain Minimax (Negamax)** | `3` | Exhaustive depth-limited Negamax tree | `51.9%` | `54.1%` | `1,260.3` | `10.65` |
| **4** | **Baseline SearchAgent** | `4` | $\alpha$-$\beta$ + MVV-LVA + Killers + Zobrist TT + IDS | `69.1%` | `73.5%` | `666.2` | `5.01` |
| **5** | **TunedSearchAgent** | `4` | $\alpha$-$\beta$ + Ordering + TT + **Quiescence** + **Tuned Eval** | **`90.4%`** | **`93.6%`** | `3,507.3` *(211.9 tree)* | **`3.82`** *(tree)* |

For full experimental tables, ablation analyses, and 5×5 round-robin matrices, see [PROJECT_REPORT.md](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/PROJECT_REPORT.md).

