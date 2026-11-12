# Mini Chess AI Project — Technical Report & Benchmark Summary
**Variant**: Gardner's 5×5 Minichess | **Convention**: Negamax (Side-to-Move)

---

## 1. System Architecture & Directory Map

```text
Mini Chess/
├── main.py                              # Unified Entry Point (GUI default, --cli, --test, --tournament, etc.)
├── README.md                            # Quick-Start Guide & Project Overview
├── PROJECT_REPORT.md                    # Technical Report & Complete Benchmark Summary
├── rules.txt                            # Formal specification of Gardner's 5x5 Minichess
├── engine/                              # Core Game State, Search & Evaluation Engine
│   ├── __init__.py
│   ├── game_state.py                    # 1D board[25], Zobrist hashing, make/unmake, legal move generator
│   ├── perft.py                         # Perft & Perft-Divide tree verification (depths 1..5)
│   ├── eval.py                          # Baseline centipawn material + 5x5 Piece-Square Tables (PST)
│   ├── eval_tuned.py                    # Tuned weights, passed pawns, rank-4 threats & endgame mop-up
│   ├── move_order.py                    # PV/TT > MVV-LVA captures > Promotions > Killer moves > PST delta
│   ├── transposition_table.py           # 64-bit Zobrist TT with ply-relative mate score normalization
│   └── search.py                        # Plain Negamax, Alpha-Beta pruning & Quiescence Search
├── agents/                              # Pluggable AI Agent Hierarchy (5 Algorithms)
│   ├── __init__.py
│   ├── base_agent.py                    # Abstract Agent interface
│   ├── random_agent.py                  # Algo 1: Seeded uniform-random baseline (RandomBot)
│   ├── greedy_agent.py                  # Algo 2: 1-ply static evaluation lookahead (GreedyBot)
│   ├── search_agent.py                  # Algo 3 & 4: Plain Negamax & Iterative Deepening Alpha-Beta + TT
│   └── tuned_agent.py                   # Algo 5: Flagship TunedSearchAgent (d=4 + Quiescence + TunedEval)
├── app/                                 # User Interfaces (Desktop GUI & Terminal CLI)
│   ├── __init__.py
│   ├── gui.py                           # Native Tkinter Desktop GUI App (Step-by-step AI, Eval Gauge, Telemetry)
│   └── cli.py                           # Interactive ASCII Terminal CLI
├── benchmarks/                          # Automated Match Arena, Tournaments, Tuning & Graph Datasets
│   ├── __init__.py
│   ├── arena.py                         # Color-swapped match runner, opening book & statistical reporting
│   ├── run_tournaments.py               # Round-robin tournament matrix & 7-config ablation study
│   ├── tune.py                          # Genetic Algorithm (GA), Simulated Annealing (SA) & Hill Climbing
│   ├── plot_graphs.py                   # Automated 6-chart matplotlib PNG generator
│   └── data/                            # CSV datasets for comparison graphs
│       ├── 1_five_algorithms_summary.csv
│       ├── 2_round_robin_tournament_matrix.csv
│       ├── 3_depth_scaling_d1_to_d4.csv
│       └── 4_search_ablation_depth4.csv
└── tests/                               # Phase-by-Phase Automated Unit & Regression Test Suites
    ├── __init__.py
    ├── run_all_tests.py                 # Unified test runner (all phases or --phase 1..4)
    ├── test_phase1.py                   # Phase 1: Board, Perft(1..4), Zobrist & Make/Unmake tests
    ├── test_phase2.py                   # Phase 2: Eval symmetry, PST flip & Greedy vs Random (100 games)
    ├── test_phase3.py                   # Phase 3: Minimax == AlphaBeta regression, TT mate & Search vs Greedy
    └── test_phase4.py                   # Phase 4: Quiescence horizon test, Ablation & Tuned vs Baseline
```

| Directory / Module | Files | Responsibility |
| :---: | :--- | :--- |
| **`engine/`** | [game_state.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/game_state.py), [perft.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/perft.py), [eval.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/eval.py), [eval_tuned.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/eval_tuned.py), [move_order.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/move_order.py), [transposition_table.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/transposition_table.py), [search.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/search.py) | 1D `board[25]`, Zobrist hashing, legal move generator, Perft, static & tuned evaluation, MVV-LVA/Killer move ordering, TT caching, Plain Negamax, $\alpha$-$\beta$ pruning, and Quiescence search. |
| **`agents/`** | [base_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/base_agent.py), [random_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/random_agent.py), [greedy_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/greedy_agent.py), [search_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/search_agent.py), [tuned_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/tuned_agent.py) | Unified `Agent` interface and all 5 AI algorithms (`RandomAgent`, `GreedyAgent`, `Plain Negamax`, `Baseline SearchAgent`, `TunedSearchAgent`). |
| **`app/`** | [gui.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/app/gui.py), [cli.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/app/cli.py) | Interactive Desktop GUI App (`tkinter`) with step-by-step AI playback, directional move arrow, live evaluation gauge, AI telemetry, and Winner/Loser banner, plus ASCII CLI. |
| **`benchmarks/`** | [arena.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/arena.py), [run_tournaments.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/run_tournaments.py), [tune.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/tune.py), [plot_graphs.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/plot_graphs.py) | Color-swapped tournament runner, round-robin matrix, 7-configuration ablation table, GA/SA/HC weight optimizers, CSV datasets, and graph plotter. |
| **`tests/`** | [run_all_tests.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/tests/run_all_tests.py), [test_phase1.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/tests/test_phase1.py), [test_phase2.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/tests/test_phase2.py), [test_phase3.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/tests/test_phase3.py), [test_phase4.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/tests/test_phase4.py) | Automated unit, regression, and 100-game tournament verification suites for Phases 1–4. |

---

## 2. Phase 1: Perft Move Generation Verification (Rules §6.4)

Validated from the canonical 5×5 Gardner starting position against an independent 2D-grid reference engine with incremental Zobrist hash verification on every `make_move` and `unmake_move`:

| Depth | Leaf Nodes | Status |
| :---: | :---: | :---: |
| **1** | `7` | Verified (`100%` match) |
| **2** | `53` | Verified (`100%` match) |
| **3** | `506` | Verified (`100%` match) |
| **4** | `4,775` | Verified (`100%` match) |
| **5** | `52,512` | Verified (`100%` match) |

---

## 3. Search Design Note: Path Repetition & Transposition Table Interaction (Rules §4.5 & §5.6)

1. **Single-Repeat Draw Detection on Search Path (§4.5)**:
   Inside `negamax`, `alphabeta`, and `quiescence`, any node with `ply > 0` whose Zobrist key appears in `state.history` (or triggers the 50-move rule / dead-position rule) immediately returns `DRAW_SCORE = 0` **before** probing or storing into the Transposition Table. Checking path repetition prior to TT lookup and not polluting the TT with path-dependent `0` scores ensures that TT entries remain pure functions of `(board, turn)` and guarantees 100% root-score equivalence with plain Minimax (`test_mandatory_minimax_equals_alphabeta_regression`).
2. **Ply-Relative Mate Score Normalization (§5.6)**:
   Checkmate at ply $p$ from the root is scored as $-(\text{MATE\_SCORE} - p)$ where $\text{MATE\_SCORE} = 100,000$. When storing a mate score ($|s| > \text{MATE\_SCORE} - 1000$) in the TT at ply $p$, `score_to_tt` adjusts it by $\pm p$ into a node-relative mate distance; `score_from_tt` readjusts by $\mp p_{\text{new}}$ on retrieval.

---

## 4. Overall Comparison of the 5 AI Algorithms
**Dataset CSV**: [1_five_algorithms_summary.csv](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/data/1_five_algorithms_summary.csv)

| # | Algorithm | Depth | Pruning & Optimizations | Evaluation Function | Overall Win Rate (%) | Overall Score Rate (%) | Avg Nodes / Move | Effective $b^*$ | Avg Time / Move (ms) |
| :-: | :--- | :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **1** | **RandomBot** | `0` | None (Uniform Random) | None | **`0.0%`** | **`0.6%`** | `1.0` | `1.00` | `0.02 ms` |
| **2** | **GreedyBot (1-Ply)** | `1` | 1-Ply Lookahead | Baseline Material + PST | **`25.1%`** | **`26.1%`** | `11.8` | `11.80` | `0.48 ms` |
| **3** | **Plain Minimax (Negamax)** | `3` | None (Exhaustive Tree) | Baseline Material + PST | **`51.9%`** | **`54.1%`** | `1,260.3` | `10.65` | `60.24 ms` |
| **4** | **Baseline SearchAgent ($\alpha$-$\beta$)** | `4` | $\alpha$-$\beta$ + MVV-LVA + Killers + TT + IDS | Baseline Material + PST | **`69.1%`** | **`73.5%`** | `666.2` | `5.01` | `48.82 ms` |
| **5** | **TunedSearchAgent ($\alpha$-$\beta$ + Q)** | `4` | $\alpha$-$\beta$ + Ordering + TT + **Quiescence** | **Tuned Weights + Passed Pawn + Mop-Up** | **`90.4%`** | **`93.6%`** | `3,507.3` *(211.9 tree + 3,295.4 Q)* | **`3.82`** *(tree)* / `7.11` *(total)* | `206.70 ms` |

---

## 5. Depth Scaling Study ($d = 1, 2, 3, 4$) Across Search Algorithms
**Dataset CSV**: [3_depth_scaling_d1_to_d4.csv](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/data/3_depth_scaling_d1_to_d4.csv)

| Depth ($d$) | Search Algorithm | Avg Total Nodes | Avg Fixed-Depth Tree Nodes | Avg Quiescence Nodes | Effective $b^*$ | Avg Time / Move (ms) | TT Hit Rate (%) | Tree Node Reduction vs. Minimax (%) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | Plain Minimax (Negamax) | `11.8` | `11.8` | `0.0` | `11.80` | `0.55 ms` | `0.0%` | `0.0%` |
| **1** | Alpha-Beta (Unordered) | `11.8` | `11.8` | `0.0` | `11.80` | `0.47 ms` | `0.0%` | `0.0%` |
| **1** | Alpha-Beta + Move Ordering | `11.8` | `11.8` | `0.0` | `11.80` | `0.51 ms` | `0.0%` | `0.0%` |
| **1** | Baseline SearchAgent (AB+Order+TT+IDS) | `11.8` | `11.8` | `0.0` | `11.80` | `0.57 ms` | `0.0%` | `0.0%` |
| **1** | TunedSearchAgent (+Quiescence) | `271.3` | `1.0` | `270.3` | `11.80` | `15.51 ms` | `0.0%` | `91.5%` |
| **2** | Plain Minimax (Negamax) | `112.8` | `112.8` | `0.0` | `10.53` | `5.09 ms` | `0.0%` | `0.0%` |
| **2** | Alpha-Beta (Unordered) | `77.3` | `77.3` | `0.0` | `8.72` | `3.17 ms` | `0.0%` | `31.5%` |
| **2** | Alpha-Beta + Move Ordering | `31.7` | `31.7` | `0.0` | `5.61` | `1.33 ms` | `0.0%` | `71.9%` |
| **2** | Baseline SearchAgent (AB+Order+TT+IDS) | `45.1` | `45.1` | `0.0` | `6.68` | `1.91 ms` | `0.0%` | `60.0%` |
| **2** | TunedSearchAgent (+Quiescence) | `789.9` | `12.8` | `777.1` | `3.58` *(tree)* | `43.85 ms` | `0.0%` | `88.7%` |
| **3** | Plain Minimax (Negamax) | `1,260.3` | `1,260.3` | `0.0` | `10.65` | `60.24 ms` | `0.0%` | `0.0%` |
| **3** | Alpha-Beta (Unordered) | `488.9` | `488.9` | `0.0` | `7.77` | `23.48 ms` | `0.0%` | `61.2%` |
| **3** | Alpha-Beta + Move Ordering | `166.8` | `166.8` | `0.0` | `5.42` | `8.11 ms` | `0.0%` | `86.8%` |
| **3** | Baseline SearchAgent (AB+Order+TT+IDS) | `212.0` | `212.0` | `0.0` | `5.89` | `10.22 ms` | `5.2%` | `83.2%` |
| **3** | TunedSearchAgent (+Quiescence) | `1,592.5` | `50.2` | `1,542.3` | `3.69` *(tree)* | `90.66 ms` | `22.9%` | `96.0%` |
| **4** | Plain Minimax (Negamax) | `13,970.3` | `13,970.3` | `0.0` | `10.72` | `1005.20 ms` | `0.0%` | `0.0%` |
| **4** | Alpha-Beta (Unordered) | `2,153.9` | `2,153.9` | `0.0` | `6.76` | `146.94 ms` | `0.0%` | `84.6%` |
| **4** | Alpha-Beta + Move Ordering | `536.4` | `536.4` | `0.0` | **`4.69`** | `40.95 ms` | `0.0%` | **`96.2%`** |
| **4** | Baseline SearchAgent (AB+Order+TT+IDS) | `666.2` | `666.2` | `0.0` | `5.01` | `48.82 ms` | `8.1%` | `95.2%` |
| **4** | TunedSearchAgent (+Quiescence) | `3,507.3` | **`211.9`** | `3,295.4` | **`3.82`** *(tree)* | `303.77 ms` | **`25.3%`** | **`98.5%`** |

---

## 6. Phase 3 & 4 Formal Search & Evaluation Ablation Study at Depth $d = 4$ (Rules §7.6 & §7.7)
**Dataset CSV**: [4_search_ablation_depth4.csv](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/data/4_search_ablation_depth4.csv)

Measured across 10 diverse opening/tactical positions (2–4 random plies, `seed=42`) at fixed depth $d = 4$:

| # | Search / Evaluation Configuration | Avg Total Nodes | Avg Q-Nodes | Avg Tree Nodes ($d \le 4$) | Avg $b^* = N^{1/4}$ | Avg Time (s) | TT Hit Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Plain Negamax (Minimax)** | `13,970` | `0` | `13,970` | `10.72` | `0.7300s` | N/A |
| **2** | **$\alpha$-$\beta$ (No Move Ordering, No TT)** | `2,153` | `0` | `2,153` | `6.76` | `0.0993s` | N/A |
| **3** | **$\alpha$-$\beta$ + Move Ordering (No TT)** | `536` | `0` | `536` | **`4.69`** | `0.0255s` | N/A |
| **4** | **$\alpha$-$\beta$ + TT (No Move Ordering)** | `1,467` | `0` | `1,467` | `6.12` | `0.0752s` | `5.0%` |
| **5** | **$\alpha$-$\beta$ + Ordering + TT (Material Only, `-PST`)** | `629` | `0` | `629` | `4.93` | `0.0674s` | `8.6%` |
| **6** | **$\alpha$-$\beta$ + Ordering + TT + PST (`Baseline d=4`)** | `666` *(IDS 1..4)* | `0` | `666` | `5.01` | `0.0354s` | `8.1%` |
| **7** | **`TunedSearchAgent` (`+Quiescence`, `+TunedEval`)** | `3,507` | `3,295` | **`212`** | `7.11` *(incl. Q)* | `0.2067s` | **`25.3%`** |

---

## 7. Heuristic Tuning & Parameter Comparison (Rules §5.4 & Step 4.2)

| Parameter | Untuned Baseline (`engine/eval.py`) | Tuned Engine (`engine/eval_tuned.py`) | Domain Rationale on 5×5 Board |
| :--- | :---: | :---: | :--- |
| **Pawn ($w_P$)** | `100` | **`115`** | Pawns start on rank 2 and promote in only 3 steps. |
| **Knight ($w_N$)** | `300` | **`310`** | Knights jump over the congested 5×5 center early. |
| **Bishop ($w_B$)** | `300` | **`290`** | Longest diagonal is only 5 squares (shorter than 8×8). |
| **Rook ($w_R$)** | `500` | **`525`** | Open 5-square files/ranks are rapidly decisive. |
| **Queen ($w_Q$)** | `900` | **`950`** | Dominates both orthos and diagonals simultaneously. |
| **PST Scale** | `1.00` | **`1.20`** | Stronger centralization (`c3`) and king shelter incentive. |
| **Passed Pawn Bonus** | `0` | **`35 × rank`** | Unopposed pawns promote quickly if not blockaded. |
| **Rank-4 Threat Bonus** | `0` | **`45`** | Unblocked pawn 1 step from promotion forces immediate response. |
| **Endgame Mop-Up** | `0` | **`12`** | Drives enemy king to corner when ahead ($\ge 200$ cp), eliminating 50-move draws. |

---

## 8. Five-Algorithm Round-Robin Tournament Matrix (All 10 Matchups, Color-Swapped, Seed=42)
**Dataset CSV**: [2_round_robin_tournament_matrix.csv](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/data/2_round_robin_tournament_matrix.csv)

| Matchup (Agent 1 vs. Agent 2) | W – D – L (Agent 1) | Win Rate | Score Rate | 95% Score CI | Avg Length | Checkmate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TunedSearch(d=4+Q)` vs. `BaselineSearch(d=4)`** | **`+71 =16 -13`** | **`71.0%`** | **`79.0%`** | `[72.0%, 86.0%]` | `42.9` plies | `84.0%` |
| **`TunedSearch(d=4+Q)` vs. `PlainMinimax(d=3)`** | **`+92 =8 -0`** | **`92.0%`** | **`96.0%`** | `[93.3%, 98.7%]` | `40.5` plies | `92.0%` |
| **`TunedSearch(d=4+Q)` vs. `GreedyBot(1-ply)`** | **`+98 =2 -0`** | **`98.0%`** | **`99.0%`** | `[97.6%, 100.0%]` | `20.4` plies | `98.0%` |
| **`TunedSearch(d=4+Q)` vs. `RandomBot`** | **`+100 =0 -0`** | **`100.0%`** | **`100.0%`** | `[100.0%, 100.0%]` | `16.8` plies | `100.0%` |
| **`BaselineSearch(d=4)` vs. `PlainMinimax(d=3)`** | **`+68 =15 -17`** | **`68.0%`** | **`75.5%`** | `[68.1%, 82.9%]` | `49.0` plies | `85.0%` |
| **`BaselineSearch(d=4)` vs. `GreedyBot(1-ply)`** | **`+96 =4 -0`** | **`96.0%`** | **`98.0%`** | `[96.1%, 99.9%]` | `26.1` plies | `96.0%` |
| **`BaselineSearch(d=4)` vs. `RandomBot`** | **`+100 =0 -0`** | **`100.0%`** | **`100.0%`** | `[100.0%, 100.0%]` | `17.7` plies | `100.0%` |
| **`PlainMinimax(d=3)` vs. `GreedyBot(1-ply)`** | **`+97 =0 -3`** | **`97.0%`** | **`97.0%`** | `[93.7%, 100.0%]` | `24.9` plies | `100.0%` |
| **`PlainMinimax(d=3)` vs. `RandomBot`** | **`+100 =0 -0`** | **`100.0%`** | **`100.0%`** | `[100.0%, 100.0%]` | `17.9` plies | `100.0%` |
| **`GreedyBot(1-ply)` vs. `RandomBot`** | **`+98 =2 -0`** | **`98.0%`** | **`99.0%`** | `[97.6%, 100.0%]` | `21.6` plies | `98.0%` |
