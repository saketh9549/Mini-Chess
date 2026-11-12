# Mini Chess AI Project — Technical Report & Benchmark Summary
**Variant**: Gardner's 5×5 Minichess | **Convention**: Negamax (Side-to-Move)

---

## 1. System Architecture & Directory Map

```text
Mini Chess/
├── main.py                      # Unified Entry Point (GUI default, --cli, --test, --tournament, etc.)
├── rules.txt                    # Formal specification of Gardner's 5x5 Minichess
├── PROJECT_REPORT.md            # Technical Report & Benchmark Summary
├── engine/                      # Core Game State, Search & Evaluation Engine
│   ├── __init__.py
│   ├── game_state.py            # 1D board[25], Zobrist hashing, make/unmake, legal move generator
│   ├── perft.py                 # Perft & Perft-Divide tree verification (depths 1..5)
│   ├── eval.py                  # Baseline centipawn material + 5x5 Piece-Square Tables (PST)
│   ├── eval_tuned.py            # Tuned weights, passed pawns, rank-4 threats & endgame mop-up
│   ├── move_order.py            # PV/TT > MVV-LVA captures > Promotions > Killer moves > PST delta
│   ├── transposition_table.py   # 64-bit Zobrist TT with ply-relative mate score normalization
│   └── search.py                # Plain Negamax, Alpha-Beta pruning & Quiescence Search
├── agents/                      # Pluggable AI Agent Hierarchy
│   ├── __init__.py
│   ├── base_agent.py            # Abstract Agent interface
│   ├── random_agent.py          # Tier 1: Seeded uniform-random baseline (RandomBot)
│   ├── greedy_agent.py          # Tier 2: 1-ply static evaluation lookahead (GreedyBot)
│   ├── search_agent.py          # Tier 3: Iterative Deepening Alpha-Beta + TT + Move Ordering
│   └── tuned_agent.py           # Tier 4: Flagship TunedSearchAgent (d=4 + Quiescence + TunedEval)
├── app/                         # User Interfaces (Desktop GUI & Terminal CLI)
│   ├── __init__.py
│   ├── gui.py                   # Native Tkinter Desktop GUI App (Board, Eval Gauge, AI Telemetry)
│   └── cli.py                   # Interactive ASCII Terminal CLI
├── benchmarks/                  # Automated Match Arena, Tournaments & Parameter Tuning
│   ├── __init__.py
│   ├── arena.py                 # Color-swapped match runner, opening book & statistical reporting
│   ├── run_tournaments.py       # 4-tier round-robin tournament matrix & 7-config ablation study
│   └── tune.py                  # Genetic Algorithm (GA), Simulated Annealing (SA) & Hill Climbing
└── tests/                       # Phase-by-Phase Automated Unit & Regression Test Suites
    ├── __init__.py
    ├── run_all_tests.py         # Unified test runner (all phases or --phase 1..4)
    ├── test_phase1.py           # Phase 1: Board, Perft(1..4), Zobrist & Make/Unmake tests
    ├── test_phase2.py           # Phase 2: Eval symmetry,PST flip & Greedy vs Random (100 games)
    ├── test_phase3.py           # Phase 3: Minimax == AlphaBeta regression, TT mate & Search vs Greedy
    └── test_phase4.py           # Phase 4: Quiescence horizon test, Ablation & Tuned vs Baseline
```

| Directory / Module | Files | Responsibility |
| :---: | :--- | :--- |
| **`engine/`** | [game_state.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/game_state.py), [perft.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/perft.py), [eval.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/eval.py), [eval_tuned.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/eval_tuned.py), [move_order.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/move_order.py), [transposition_table.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/transposition_table.py), [search.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/engine/search.py) | 1D `board[25]`, Zobrist hashing, legal move generator, Perft, static & tuned evaluation, MVV-LVA/Killer move ordering, TT caching, Plain Negamax, $\alpha$-$\beta$ pruning, and Quiescence search. |
| **`agents/`** | [base_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/base_agent.py), [random_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/random_agent.py), [greedy_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/greedy_agent.py), [search_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/search_agent.py), [tuned_agent.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/agents/tuned_agent.py) | Unified `Agent` interface and all 4 bot tiers (`RandomAgent`, `GreedyAgent`, `SearchAgent`, `TunedSearchAgent`). |
| **`app/`** | [gui.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/app/gui.py), [cli.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/app/cli.py) | Interactive Desktop GUI App (`tkinter`) with live evaluation gauge and AI telemetry, plus interactive ASCII CLI. |
| **`benchmarks/`** | [arena.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/arena.py), [run_tournaments.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/run_tournaments.py), [tune.py](file:///c:/Users/Lenovo/OneDrive%20-%20Shiv%20Nadar%20Institution%20of%20Eminence/saketh/shiv%20nadar%20university/cse/semester%205/Artificial%20intelligence/Mini%20Chess/benchmarks/tune.py) | Color-swapped tournament runner, 4-tier round-robin matrix, 7-configuration ablation table, and GA/SA/HC weight optimizers. |
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

## 4. Phase 3 & 4 Formal Search & Evaluation Ablation Study (Rules §7.6 & §7.7)

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

### Key Ablation Findings:
- **$\alpha$-$\beta$ Pruning + Move Ordering**: Reduces average nodes at depth 4 from `13,970` to `536` (**`96.2%` node reduction**), driving the effective branching factor from $b^* = 10.72$ down to $b^* = 4.69$ (close to the theoretical optimum $\sqrt{10.72} \approx 3.27$).
- **Quiescence Search**: Expands tactical capture/promotion sequences at leaf nodes (`3,295` Q-nodes on average) while shrinking the main fixed-depth tree to just `212` nodes and boosting TT hit rate to `25.3%`.

---

## 5. Heuristic Tuning & Parameter Comparison (Rules §5.4 & Step 4.2)

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

## 6. Four-Tier Tournament Matrix (100 Games / Matchup, Color-Swapped, Seed=42)

Every matchup consists of **100 games** (50 color-swapped pairs from identical seeded 2–4 ply randomized openings):

| Matchup (Agent 1 vs. Agent 2) | W – D – L (Agent 1) | As White | As Black | Win Rate | Score Rate | 95% Score CI | Avg Length | Checkmate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TunedSearch(d=4+Q)` vs. `BaselineSearch(d=4)`** | **`+71 =16 -13`** | `+39 =5 -6` | `+32 =11 -7` | **`71.0%`** | **`79.0%`** | `[72.0%, 86.0%]` | `42.9` plies | `84.0%` |
| **`TunedSearch(d=4+Q)` vs. `GreedyBot(1-ply)`** | **`+98 =2 -0`** | `+50 =0 -0` | `+48 =2 -0` | **`98.0%`** | **`99.0%`** | `[97.6%, 100.0%]` | `20.4` plies | `98.0%` |
| **`TunedSearch(d=4+Q)` vs. `RandomBot`** | **`+100 =0 -0`** | `+50 =0 -0` | `+50 =0 -0` | **`100.0%`** | **`100.0%`** | `[100.0%, 100.0%]` | `16.8` plies | `100.0%` |
| **`BaselineSearch(d=4)` vs. `GreedyBot(1-ply)`** | **`+96 =4 -0`** | `+47 =3 -0` | `+49 =1 -0` | **`96.0%`** | **`98.0%`** | `[96.1%, 99.9%]` | `26.1` plies | `96.0%` |
| **`BaselineSearch(d=4)` vs. `RandomBot`** | **`+100 =0 -0`** | `+50 =0 -0` | `+50 =0 -0` | **`100.0%`** | **`100.0%`** | `[100.0%, 100.0%]` | `17.7` plies | `100.0%` |
| **`GreedyBot(1-ply)` vs. `RandomBot`** | **`+98 =2 -0`** | `+48 =2 -0` | `+50 =0 -0` | **`98.0%`** | **`99.0%`** | `[97.6%, 100.0%]` | `21.6` plies | `98.0%` |
