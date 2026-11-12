"""
Phase 1: Interactive CLI for Gardner's 5x5 Minichess
Renders the ASCII board state and allows playing test moves manually,
inspecting legal moves, undoing moves, and running Perft diagnostics.
"""

from __future__ import annotations
from pathlib import Path
import sys
import time
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.game_state import GameState, Move, WHITE
from engine.perft import perft_divide, run_perft_suite


def get_game_status(state: GameState) -> Tuple[bool, str]:
    """Evaluate terminal / non-terminal status of `state`."""
    legal_moves = state.generate_legal_moves()
    in_chk = state.in_check()

    if not legal_moves:
        if in_chk:
            winner = "Black (-1)" if state.turn == WHITE else "White (+1)"
            return True, f"CHECKMATE! {winner} wins."
        return True, "STALEMATE! Draw (no legal moves)."

    if state.is_threefold_repetition():
        return True, "DRAW by Threefold Repetition (rules.txt 4.1)."
    if state.is_fifty_move_draw():
        return True, "DRAW by Fifty-Move Rule (100 plies, rules.txt 4.2)."
    if state.is_dead_position():
        return True, "DRAW by Dead Position / Insufficient Material (rules.txt 4.3)."
    if state.is_hard_cap_draw():
        return True, "DRAW by 300-Ply Hard Cap (rules.txt 4.4)."

    if in_chk:
        side = "White" if state.turn == WHITE else "Black"
        return False, f"CHECK! {side} king is under attack."
    return False, "Ongoing"


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  <move>        Play a move in UCI notation (e.g., 'a2a3', 'b1c3', 'a4a5q')\n"
        "  moves         List all legal moves in the current position\n"
        "  undo          Undo the last move (unmake_move)\n"
        "  perft <d>     Run Perft (with root move divide) to depth <d> from current state\n"
        "  suite [d]     Run standard starting-position Perft verification up to depth [d] (default 4)\n"
        "  new / reset   Reset board to starting position\n"
        "  board / show  Print the current board\n"
        "  help          Show this help message\n"
        "  quit / exit   Exit the CLI\n"
    )


def run_cli() -> None:
    state = GameState(debug=True)
    print("=" * 60)
    print("  GARDNER'S 5x5 MINICHESS - INTERACTIVE CLI")
    print("=" * 60)
    print("Type a move (e.g. 'a2a3', 'b1c3', 'a4a5q') or 'help' for commands.\n")

    print(state.render_ascii())

    while True:
        is_over, status_msg = get_game_status(state)
        if status_msg != "Ongoing":
            print(f"\n>>> {status_msg}")

        legal_moves = state.generate_legal_moves()
        legal_map = {m.uci(): m for m in legal_moves}

        turn_label = "White" if state.turn == WHITE else "Black"
        try:
            raw = input(f"\n[{turn_label}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Mini Chess CLI.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif cmd in ("help", "h", "?"):
            print_help()
        elif cmd in ("board", "show", "d"):
            print()
            print(state.render_ascii())
        elif cmd in ("new", "reset"):
            state = GameState(debug=True)
            print("\nBoard reset to initial position.")
            print(state.render_ascii())
        elif cmd in ("moves", "legal", "m"):
            if not legal_moves:
                print("No legal moves available.")
            else:
                formatted = ", ".join(m.uci() for m in legal_moves)
                print(f"Legal moves ({len(legal_moves)}): {formatted}")
        elif cmd in ("undo", "u", "back"):
            try:
                undone = state.unmake_move()
                print(f"Undid move: {undone.uci()}")
                print(state.render_ascii())
            except RuntimeError as e:
                print(f"Cannot undo: {e}")
        elif cmd == "perft":
            depth = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
            t0 = time.perf_counter()
            total, breakdown = perft_divide(state, depth)
            elapsed = time.perf_counter() - t0
            print(f"\nPerft Divide (depth={depth}):")
            for m_str, cnt in sorted(breakdown.items()):
                print(f"  {m_str}: {cnt}")
            nps = int(total / elapsed) if elapsed > 0 else 0
            print(f"Total leaf nodes: {total}  (Time: {elapsed:.4f}s, {nps} NPS)")
        elif cmd == "suite":
            depth = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 4
            run_perft_suite(max_depth=depth, debug_hash=True)
        else:
            move_str = cmd
            if move_str in legal_map:
                chosen = legal_map[move_str]
                state.make_move(chosen)
                print()
                print(state.render_ascii())
            else:
                matching_promos = [u for u in legal_map if u.startswith(move_str) and len(u) == 5]
                if matching_promos:
                    print(
                        f"Move {move_str!r} is a pawn promotion! Specify promotion piece "
                        f"({', '.join(matching_promos)})."
                    )
                else:
                    print(
                        f"Illegal or unrecognized move {raw!r}. "
                        f"Type 'moves' to see {len(legal_moves)} legal moves, or 'help'."
                    )


if __name__ == "__main__":
    run_cli()

