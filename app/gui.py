"""
Gardner's 5x5 Minichess - Interactive Desktop GUI Application (app/gui.py)
Built with Python's standard `tkinter` library (zero external dependencies).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base_agent import Agent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.search_agent import SearchAgent
from agents.tuned_agent import TunedSearchAgent
from benchmarks.arena import adjudicate_state
from engine.eval import MATE_SCORE
from engine.eval_tuned import evaluate_tuned_white
from engine.game_state import (
    BISHOP,
    BLACK,
    EMPTY,
    FILES,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    RANKS,
    ROOK,
    WHITE,
    GameState,
    Move,
    coords_to_sq,
    sq_to_coords,
    sq_to_str,
)
from engine.search import SearchStats

BG_DARK = "#181926"
BG_PANEL = "#24273a"
BG_CARD = "#1e2030"
FG_TEXT = "#cad3f5"
FG_SUBTEXT = "#a5adcb"
ACCENT_BLUE = "#8aadf4"
ACCENT_GREEN = "#a6da95"
ACCENT_AMBER = "#eed49f"
ACCENT_RED = "#ed8796"

LIGHT_SQ = "#e8d8c8"
DARK_SQ = "#966f53"
LAST_MOVE_LIGHT = "#f2e38c"
LAST_MOVE_DARK = "#c9b04a"
SELECTED_SQ = "#7dc4e4"
CHECK_SQ = "#e65c65"
LEGAL_DOT_COLOR = "#2e6f40"
LEGAL_RING_COLOR = "#c93b3b"
ARROW_COLOR = "#e07a14"

PIECE_GLYPHS: Dict[int, str] = {
    KING: "♚",
    QUEEN: "♛",
    ROOK: "♜",
    BISHOP: "♝",
    KNIGHT: "♞",
    PAWN: "♟",
}

PIECE_NAMES: Dict[int, str] = {
    KING: "King",
    QUEEN: "Queen",
    ROOK: "Rook",
    BISHOP: "Bishop",
    KNIGHT: "Knight",
    PAWN: "Pawn",
}

PIECE_VALUES: Dict[int, int] = {
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 300,
    ROOK: 500,
    QUEEN: 900,
    KING: 0,
}

PLAYER_OPTIONS = (
    "Human",
    "TunedSearchAgent (d=4 + Quiescence)",
    "TunedSearchAgent (d=5 + Quiescence)",
    "Baseline SearchAgent (d=4)",
    "Baseline SearchAgent (d=3)",
    "GreedyBot (1-Ply)",
    "RandomBot",
)


@dataclass
class StepSnapshot:
    ply: int
    move: Move
    uci: str
    step_text: str
    board: List[int]
    turn: int
    in_check: bool
    stats: Optional[SearchStats] = None


def create_agent_from_label(label: str) -> Optional[Agent]:
    if label == "Human":
        return None
    if label == "TunedSearchAgent (d=4 + Quiescence)":
        return TunedSearchAgent(max_depth=4, use_quiescence=True, name="Tuned(d=4+Q)")
    if label == "TunedSearchAgent (d=5 + Quiescence)":
        return TunedSearchAgent(max_depth=5, use_quiescence=True, name="Tuned(d=5+Q)")
    if label == "Baseline SearchAgent (d=4)":
        return SearchAgent(max_depth=4, use_quiescence=False, name="Baseline(d=4)")
    if label == "Baseline SearchAgent (d=3)":
        return SearchAgent(max_depth=3, use_quiescence=False, name="Baseline(d=3)")
    if label == "GreedyBot (1-Ply)":
        return GreedyAgent(seed=42, name="GreedyBot")
    if label == "RandomBot":
        return RandomAgent(seed=42, name="RandomBot")
    return None


def short_label(label: str) -> str:
    if label == "Human":
        return "Human"
    return label.split(" (")[0]


class MiniChessApp:
    BOARD_PX = 500
    MARGIN_PX = 34
    SQ_PX = (BOARD_PX - 2 * MARGIN_PX) // 5

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gardner's 5×5 Minichess — AI Arena & Engine Studio")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self.state = GameState()
        self.initial_board = list(self.state.board)
        self.move_uci_history: List[str] = []
        self.snapshots: List[StepSnapshot] = []
        self.view_ply: Optional[int] = None  # None = live board; int = reviewing step

        self.last_move: Optional[Move] = None
        self.selected_sq: Optional[int] = None
        self.legal_moves_for_selected: List[Move] = []
        self.flipped: bool = False
        self.ai_paused: bool = False
        self.step_once: bool = False
        self.ai_thinking: bool = False
        self.game_id: int = 0
        self.ai_delay_var = tk.DoubleVar(value=0.8)

        self.white_choice = tk.StringVar(value=PLAYER_OPTIONS[0])
        self.black_choice = tk.StringVar(value=PLAYER_OPTIONS[1])
        self.white_agent: Optional[Agent] = create_agent_from_label(self.white_choice.get())
        self.black_agent: Optional[Agent] = create_agent_from_label(self.black_choice.get())

        self.hint_agent = TunedSearchAgent(max_depth=4, use_quiescence=True, name="Analyzer(d=4+Q)")

        self._build_ui()
        self._refresh_all()
        self._schedule_ai_if_needed()

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=14)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------- LEFT COLUMN ----------------
        left_col = tk.Frame(main_frame, bg=BG_DARK)
        left_col.pack(side=tk.LEFT, padx=(0, 16))

        # Primary Status Banner (Shows Turn or Final Winner/Loser)
        self.status_label = tk.Label(
            left_col,
            text="White (+1) to move",
            font=("Segoe UI", 13, "bold"),
            bg=BG_PANEL,
            fg=FG_TEXT,
            pady=7,
            padx=12,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(0, 4))

        # Step-by-Step Last Move Narration Banner
        self.step_label = tk.Label(
            left_col,
            text="Step 0: Starting Position — Ready to play",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=ACCENT_AMBER,
            pady=5,
            padx=12,
            anchor="w",
        )
        self.step_label.pack(fill=tk.X, pady=(0, 8))

        board_row = tk.Frame(left_col, bg=BG_DARK)
        board_row.pack()

        self.eval_canvas = tk.Canvas(
            board_row,
            width=32,
            height=self.BOARD_PX,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground="#363a4f",
        )
        self.eval_canvas.pack(side=tk.LEFT, padx=(0, 8))

        self.board_canvas = tk.Canvas(
            board_row,
            width=self.BOARD_PX,
            height=self.BOARD_PX,
            bg=BG_PANEL,
            highlightthickness=1,
            highlightbackground="#363a4f",
        )
        self.board_canvas.pack(side=tk.LEFT)
        self.board_canvas.bind("<Button-1>", self._on_board_click)

        # Captured Pieces & Material Balance Bar
        self.material_label = tk.Label(
            left_col,
            text="Captured — White: None   |   Black: None",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=FG_SUBTEXT,
            pady=4,
            padx=10,
            anchor="w",
        )
        self.material_label.pack(fill=tk.X, pady=(6, 0))

        # Row 1 Buttons: Game Controls
        btn_row = tk.Frame(left_col, bg=BG_DARK, pady=8)
        btn_row.pack(fill=tk.X)

        self._make_button(btn_row, "New Game", self._on_new_game, bg="#8aadf4", fg="#181926").pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self._make_button(btn_row, "Undo", self._on_undo, bg="#363a4f", fg=FG_TEXT).pack(
            side=tk.LEFT, padx=5
        )
        self._make_button(btn_row, "Flip Board", self._on_flip, bg="#363a4f", fg=FG_TEXT).pack(
            side=tk.LEFT, padx=5
        )
        self._make_button(btn_row, "Hint (d=4+Q)", self._on_hint, bg="#a6da95", fg="#181926").pack(
            side=tk.LEFT, padx=5
        )

        # Row 2 Buttons: Step-by-Step AI & Replay Controls
        step_row = tk.Frame(left_col, bg=BG_DARK)
        step_row.pack(fill=tk.X)

        self._make_button(step_row, "◀ Prev Step", self._on_prev_step, bg=BG_PANEL, fg=FG_TEXT).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._make_button(step_row, "Next Step ▶", self._on_next_step, bg=BG_PANEL, fg=FG_TEXT).pack(
            side=tk.LEFT, padx=4
        )
        self._make_button(step_row, "⏭ Step AI Move", self._on_step_ai, bg=ACCENT_GREEN, fg="#181926").pack(
            side=tk.LEFT, padx=6
        )
        self.pause_btn = self._make_button(
            step_row, "Pause AI", self._on_toggle_pause, bg="#eed49f", fg="#181926"
        )
        self.pause_btn.pack(side=tk.RIGHT)

        # ---------------- RIGHT COLUMN ----------------
        right_col = tk.Frame(main_frame, bg=BG_DARK, width=360)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        player_card = tk.LabelFrame(
            right_col,
            text=" Match Setup & Step Pacing ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
            padx=12,
            pady=8,
        )
        player_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(player_card, text="White (+1):", font=("Segoe UI", 10), bg=BG_PANEL, fg=FG_TEXT).grid(
            row=0, column=0, sticky="w", pady=3
        )
        w_menu = ttk.Combobox(
            player_card,
            textvariable=self.white_choice,
            values=PLAYER_OPTIONS,
            state="readonly",
            width=30,
        )
        w_menu.grid(row=0, column=1, padx=(8, 0), pady=3)
        w_menu.bind("<<ComboboxSelected>>", self._on_player_changed)

        tk.Label(player_card, text="Black (-1):", font=("Segoe UI", 10), bg=BG_PANEL, fg=FG_TEXT).grid(
            row=1, column=0, sticky="w", pady=3
        )
        b_menu = ttk.Combobox(
            player_card,
            textvariable=self.black_choice,
            values=PLAYER_OPTIONS,
            state="readonly",
            width=30,
        )
        b_menu.grid(row=1, column=1, padx=(8, 0), pady=3)
        b_menu.bind("<<ComboboxSelected>>", self._on_player_changed)

        # AI vs AI Step Delay Slider
        tk.Label(player_card, text="AI Step Delay:", font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_SUBTEXT).grid(
            row=2, column=0, sticky="w", pady=(6, 2)
        )
        delay_frame = tk.Frame(player_card, bg=BG_PANEL)
        delay_frame.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 2))

        self.delay_val_label = tk.Label(
            delay_frame, text="0.8s", font=("Consolas", 9, "bold"), bg=BG_PANEL, fg=ACCENT_AMBER, width=5
        )
        self.delay_val_label.pack(side=tk.RIGHT)

        delay_scale = tk.Scale(
            delay_frame,
            variable=self.ai_delay_var,
            from_=0.2,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=BG_PANEL,
            fg=FG_TEXT,
            troughcolor=BG_CARD,
            highlightthickness=0,
            command=lambda v: self.delay_val_label.configure(text=f"{float(v):.1f}s"),
        )
        delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        telemetry_card = tk.LabelFrame(
            right_col,
            text=" Engine Search Telemetry ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_GREEN,
            padx=12,
            pady=8,
        )
        telemetry_card.pack(fill=tk.X, pady=(0, 10))

        self.telemetry_vars: Dict[str, tk.StringVar] = {}
        fields = [
            ("Static Eval (Tuned)", "eval_str"),
            ("Last Search Score", "score_str"),
            ("Depth / Mode", "depth_str"),
            ("Total Nodes / Q-Nodes", "nodes_str"),
            ("Branching Factor (b*)", "bstar_str"),
            ("Search Time / Speed", "time_str"),
            ("TT Hit Rate", "tt_str"),
            ("Principal Variation", "pv_str"),
            ("Halfmove / Plies", "clock_str"),
        ]
        for row_idx, (label_text, key) in enumerate(fields):
            tk.Label(
                telemetry_card,
                text=f"{label_text}:",
                font=("Segoe UI", 9),
                bg=BG_PANEL,
                fg=FG_SUBTEXT,
            ).grid(row=row_idx, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="—")
            self.telemetry_vars[key] = var
            tk.Label(
                telemetry_card,
                textvariable=var,
                font=("Consolas", 9, "bold"),
                bg=BG_PANEL,
                fg=FG_TEXT,
                anchor="w",
            ).grid(row=row_idx, column=1, sticky="w", padx=(10, 0), pady=2)

        history_card = tk.LabelFrame(
            right_col,
            text=" Step-by-Step Move History ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_AMBER,
            padx=10,
            pady=8,
        )
        history_card.pack(fill=tk.BOTH, expand=True)

        self.history_text = tk.Text(
            history_card,
            width=36,
            height=9,
            font=("Consolas", 10),
            bg=BG_CARD,
            fg=FG_TEXT,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(history_card, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _make_button(
        self, parent: tk.Widget, text: str, command, bg: str, fg: str
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief=tk.FLAT,
            padx=9,
            pady=5,
            cursor="hand2",
        )

    def _sq_to_canvas_xy(self, sq: int) -> Tuple[int, int, int, int]:
        r, f = sq_to_coords(sq)
        if not self.flipped:
            vis_row = 4 - r
            vis_col = f
        else:
            vis_row = r
            vis_col = 4 - f
        x0 = self.MARGIN_PX + vis_col * self.SQ_PX
        y0 = self.MARGIN_PX + vis_row * self.SQ_PX
        return x0, y0, x0 + self.SQ_PX, y0 + self.SQ_PX

    def _sq_center(self, sq: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def _canvas_xy_to_sq(self, x: int, y: int) -> Optional[int]:
        vis_col = (x - self.MARGIN_PX) // self.SQ_PX
        vis_row = (y - self.MARGIN_PX) // self.SQ_PX
        if not (0 <= vis_col < 5 and 0 <= vis_row < 5):
            return None
        if not self.flipped:
            r = 4 - vis_row
            f = vis_col
        else:
            r = vis_row
            f = 4 - vis_col
        return coords_to_sq(r, f)

    def _get_active_board_and_move(self) -> Tuple[List[int], Optional[Move], int, bool]:
        if self.view_ply is None or not self.snapshots:
            return (
                self.state.board,
                self.last_move,
                self.state.turn,
                self.state.in_check(self.state.turn),
            )
        if self.view_ply == 0:
            return self.initial_board, None, WHITE, False
        snap = self.snapshots[min(self.view_ply - 1, len(self.snapshots) - 1)]
        return snap.board, snap.move, snap.turn, snap.in_check

    def _draw_board(self) -> None:
        c = self.board_canvas
        c.delete("all")

        board_to_draw, active_move, active_turn, is_in_check = self._get_active_board_and_move()

        # Subtle inner board border
        m = self.MARGIN_PX
        b_end = self.BOARD_PX - m
        c.create_rectangle(m - 2, m - 2, b_end + 2, b_end + 2, outline="#c9b04a", width=2)

        checked_king_sq = -1
        if is_in_check:
            target_k = KING * active_turn
            for s_idx, p_val in enumerate(board_to_draw):
                if p_val == target_k:
                    checked_king_sq = s_idx
                    break

        legal_targets = {m.to_sq: m for m in self.legal_moves_for_selected} if self.view_ply is None else {}

        for sq in range(25):
            r, f = sq_to_coords(sq)
            x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
            is_light = (r + f) % 2 == 1

            if self.view_ply is None and sq == self.selected_sq:
                fill_color = SELECTED_SQ
            elif sq == checked_king_sq:
                fill_color = CHECK_SQ
            elif active_move and sq in (active_move.from_sq, active_move.to_sq):
                fill_color = LAST_MOVE_LIGHT if is_light else LAST_MOVE_DARK
            else:
                fill_color = LIGHT_SQ if is_light else DARK_SQ

            c.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="#24273a", width=1)

            piece = board_to_draw[sq]
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            if piece != EMPTY:
                glyph = PIECE_GLYPHS[abs(piece)]
                if piece > 0:
                    c.create_text(cx + 1, cy + 2, text=glyph, font=("Segoe UI Symbol", 38), fill="#181926")
                    c.create_text(cx, cy, text=glyph, font=("Segoe UI Symbol", 38), fill="#ffffff")
                else:
                    c.create_text(cx + 1, cy + 1, text=glyph, font=("Segoe UI Symbol", 38), fill="#d0d0d0")
                    c.create_text(cx, cy, text=glyph, font=("Segoe UI Symbol", 38), fill="#191924")

            if sq in legal_targets:
                if board_to_draw[sq] == EMPTY:
                    rad = 11
                    c.create_oval(
                        cx - rad, cy - rad, cx + rad, cy + rad,
                        fill=LEGAL_DOT_COLOR, outline="",
                    )
                else:
                    pad = 5
                    c.create_oval(
                        x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                        outline=LEGAL_RING_COLOR, width=4,
                    )

        # Draw directional move arrow for the active move so user sees step-by-step movement
        if active_move is not None:
            self._draw_move_arrow(active_move.from_sq, active_move.to_sq)

        for i in range(5):
            file_idx = i if not self.flipped else (4 - i)
            rank_idx = (4 - i) if not self.flipped else i

            fx = self.MARGIN_PX + i * self.SQ_PX + self.SQ_PX // 2
            ry = self.MARGIN_PX + i * self.SQ_PX + self.SQ_PX // 2

            c.create_text(
                fx, self.BOARD_PX - self.MARGIN_PX // 2,
                text=FILES[file_idx], font=("Segoe UI", 10, "bold"), fill=FG_SUBTEXT,
            )
            c.create_text(
                self.MARGIN_PX // 2, ry,
                text=RANKS[rank_idx], font=("Segoe UI", 10, "bold"), fill=FG_SUBTEXT,
            )

        # If game is over and viewing live board, draw a clean bottom outcome banner on the board
        verdict = adjudicate_state(self.state)
        if verdict is not None and self.view_ply is None:
            self._draw_gameover_banner(verdict)

    def _draw_move_arrow(self, from_sq: int, to_sq: int) -> None:
        c = self.board_canvas
        x0, y0 = self._sq_center(from_sq)
        x1, y1 = self._sq_center(to_sq)
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return
        ux, uy = dx / dist, dy / dist
        sx, sy = x0 + ux * 14, y0 + uy * 14
        ex, ey = x1 - ux * 18, y1 - uy * 18

        c.create_line(
            sx, sy, ex, ey,
            fill="#181926", width=7, capstyle=tk.ROUND, arrow=tk.LAST, arrowshape=(13, 16, 6),
        )
        c.create_line(
            sx, sy, ex, ey,
            fill=ARROW_COLOR, width=4, capstyle=tk.ROUND, arrow=tk.LAST, arrowshape=(12, 15, 5),
        )

    def _draw_gameover_banner(self, verdict: Tuple[int, str]) -> None:
        winner, reason = verdict
        c = self.board_canvas
        w_label = short_label(self.white_choice.get())
        b_label = short_label(self.black_choice.get())
        plies = len(self.move_uci_history)

        if winner == WHITE:
            title = f"🏆 WINNER: WHITE (+1) — {w_label}"
            sub = f"❌ DEFEATED: BLACK (-1) — {b_label}   |   Checkmate in {plies} plies"
            border_col = ACCENT_GREEN
        elif winner == BLACK:
            title = f"🏆 WINNER: BLACK (-1) — {b_label}"
            sub = f"❌ DEFEATED: WHITE (+1) — {w_label}   |   Checkmate in {plies} plies"
            border_col = ACCENT_GREEN
        else:
            r_clean = reason.replace("_", " ").title()
            title = f"½–½ DRAW — {r_clean}"
            sub = f"White ({w_label}) vs. Black ({b_label})   |   {plies} plies"
            border_col = ACCENT_AMBER

        cx = self.BOARD_PX // 2
        cy = self.BOARD_PX // 2
        bw, bh = 420, 82
        x0, y0, x1, y1 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2

        c.create_rectangle(x0 + 3, y0 + 3, x1 + 3, y1 + 3, fill="#0f1018", outline="")
        c.create_rectangle(x0, y0, x1, y1, fill=BG_CARD, outline=border_col, width=2)
        c.create_text(cx, y0 + 26, text=title, font=("Segoe UI", 11, "bold"), fill=border_col)
        c.create_text(cx, y0 + 54, text=sub, font=("Segoe UI", 9, "bold"), fill=FG_TEXT)

    def _draw_eval_bar(self, white_cp: int) -> None:
        ec = self.eval_canvas
        ec.delete("all")
        w = 32
        h = self.BOARD_PX

        clamped = max(-1500, min(1500, white_cp))
        white_frac = 1.0 / (1.0 + math.exp(-clamped / 350.0))
        white_h = int(round(h * white_frac))

        if not self.flipped:
            black_h = h - white_h
            ec.create_rectangle(0, 0, w, black_h, fill="#24273a", outline="")
            ec.create_rectangle(0, black_h, w, h, fill="#e8d8c8", outline="")
        else:
            ec.create_rectangle(0, 0, w, white_h, fill="#e8d8c8", outline="")
            ec.create_rectangle(0, white_h, w, h, fill="#24273a", outline="")

        ec.create_line(0, h // 2, w, h // 2, fill="#ed8796", width=1)

        pawns = white_cp / 100.0
        txt = f"{pawns:+.1f}"
        if white_cp >= 0:
            ec.create_text(w // 2, h - 14, text=txt, font=("Consolas", 8, "bold"), fill="#181926")
        else:
            ec.create_text(w // 2, 14, text=txt, font=("Consolas", 8, "bold"), fill=FG_TEXT)

    def _update_material_bar(self, board: List[int]) -> None:
        start_counts = {PAWN: 5, KNIGHT: 1, BISHOP: 1, ROOK: 1, QUEEN: 1, KING: 1}
        w_cnt = {k: 0 for k in start_counts}
        b_cnt = {k: 0 for k in start_counts}
        w_mat, b_mat = 0, 0
        for p in board:
            if p > 0:
                w_cnt[p] = w_cnt.get(p, 0) + 1
                w_mat += PIECE_VALUES.get(p, 0)
            elif p < 0:
                b_cnt[-p] = b_cnt.get(-p, 0) + 1
                b_mat += PIECE_VALUES.get(-p, 0)

        cap_w, cap_b = [], []
        for pt in (QUEEN, ROOK, BISHOP, KNIGHT, PAWN):
            cap_w.extend([PIECE_GLYPHS[pt]] * max(0, start_counts[pt] - b_cnt[pt]))
            cap_b.extend([PIECE_GLYPHS[pt]] * max(0, start_counts[pt] - w_cnt[pt]))

        diff = (w_mat - b_mat) // 100
        w_adv = f" (+{diff})" if diff > 0 else ""
        b_adv = f" (+{-diff})" if diff < 0 else ""
        w_str = "".join(cap_w) if cap_w else "None"
        b_str = "".join(cap_b) if cap_b else "None"
        self.material_label.configure(
            text=f"Captured by White{w_adv}: {w_str}     |     Captured by Black{b_adv}: {b_str}"
        )

    def _is_human_turn(self) -> bool:
        if adjudicate_state(self.state) is not None:
            return False
        active_agent = self.white_agent if self.state.turn == WHITE else self.black_agent
        return active_agent is None

    def _on_board_click(self, event: tk.Event) -> None:
        if self.view_ply is not None:
            self.view_ply = None
            self._refresh_all()
            return
        if self.ai_thinking or not self._is_human_turn():
            return

        sq = self._canvas_xy_to_sq(event.x, event.y)
        if sq is None:
            return

        matching_moves = [m for m in self.legal_moves_for_selected if m.to_sq == sq]
        if matching_moves:
            if len(matching_moves) == 1:
                self._execute_move(matching_moves[0])
            else:
                promo_piece = self._prompt_promotion_choice()
                if promo_piece is not None:
                    for m in matching_moves:
                        if m.promotion == promo_piece:
                            self._execute_move(m)
                            break
            return

        piece = self.state.board[sq]
        if piece != EMPTY and (piece > 0) == (self.state.turn > 0):
            self.selected_sq = sq
            all_legal = self.state.generate_legal_moves()
            self.legal_moves_for_selected = [m for m in all_legal if m.from_sq == sq]
        else:
            self.selected_sq = None
            self.legal_moves_for_selected = []

        self._draw_board()

    def _prompt_promotion_choice(self) -> Optional[int]:
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.configure(bg=BG_PANEL, padx=16, pady=12)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        choice_var = tk.IntVar(value=QUEEN)

        tk.Label(
            dialog,
            text="Choose Promotion Piece:",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PANEL,
            fg=FG_TEXT,
        ).pack(pady=(0, 10))

        btn_frame = tk.Frame(dialog, bg=BG_PANEL)
        btn_frame.pack()

        options = [
            (QUEEN, "♛ Queen"),
            (ROOK, "♜ Rook"),
            (BISHOP, "♝ Bishop"),
            (KNIGHT, "♞ Knight"),
        ]
        for p_code, label in options:
            def _pick(val=p_code):
                choice_var.set(val)
                dialog.destroy()

            self._make_button(btn_frame, label, _pick, bg=ACCENT_BLUE, fg="#181926").pack(
                side=tk.LEFT, padx=4
            )

        self.root.wait_window(dialog)
        return choice_var.get()

    def _execute_move(self, move: Move, stats: Optional[SearchStats] = None) -> None:
        mover = self.state.turn
        mover_label = short_label(self.white_choice.get() if mover == WHITE else self.black_choice.get())
        moved_piece = abs(self.state.board[move.from_sq])
        captured_piece = abs(self.state.board[move.to_sq])

        self.state.make_move(move)
        self.view_ply = None
        ply = len(self.move_uci_history) + 1
        self.move_uci_history.append(move.uci())
        self.last_move = move
        self.selected_sq = None
        self.legal_moves_for_selected = []

        side_str = "White" if mover == WHITE else "Black"
        p_glyph = PIECE_GLYPHS.get(moved_piece, "")
        p_name = PIECE_NAMES.get(moved_piece, "Piece")
        src, dst = sq_to_str(move.from_sq), sq_to_str(move.to_sq)

        if captured_piece != EMPTY:
            cap_name = PIECE_NAMES.get(captured_piece, "piece")
            desc = f"{src} × {dst} (captured {cap_name})"
        else:
            desc = f"{src} → {dst}"
        if move.promotion is not None:
            desc += f" [={PIECE_NAMES.get(move.promotion, 'Queen')}]"
        if self.state.in_check(self.state.turn):
            desc += " [CHECK!]"

        step_text = f"Step {ply}: {side_str} ({mover_label}) played {p_glyph} {p_name} {desc}"
        self.snapshots.append(
            StepSnapshot(
                ply=ply,
                move=move,
                uci=move.uci(),
                step_text=step_text,
                board=list(self.state.board),
                turn=self.state.turn,
                in_check=self.state.in_check(self.state.turn),
                stats=stats,
            )
        )

        if stats is not None:
            self._update_telemetry_from_stats(stats)

        self._refresh_all()
        self._schedule_ai_if_needed()

    def _schedule_ai_if_needed(self) -> None:
        if self.ai_thinking:
            return
        if adjudicate_state(self.state) is not None:
            return
        if self.ai_paused and not self.step_once:
            return

        active_agent = self.white_agent if self.state.turn == WHITE else self.black_agent
        if active_agent is None:
            return

        self.step_once = False
        self.ai_thinking = True
        side_str = "White" if self.state.turn == WHITE else "Black"
        self.status_label.configure(
            text=f"{side_str} ({active_agent.name}) is thinking...",
            fg=ACCENT_BLUE,
        )

        current_game_id = self.game_id
        state_copy = self.state.copy()
        both_ai = (self.white_agent is not None) and (self.black_agent is not None)
        min_delay = float(self.ai_delay_var.get()) if both_ai else 0.15

        def _worker():
            t0 = time.perf_counter()
            chosen_move = active_agent.select_move(state_copy)
            elapsed = time.perf_counter() - t0
            stats = getattr(active_agent, "last_stats", None)
            if stats is None and chosen_move is not None:
                stats = SearchStats(
                    depth=1,
                    nodes=len(state_copy.generate_legal_moves()),
                    elapsed_sec=elapsed,
                    best_move=chosen_move,
                )
            rem = min_delay - elapsed
            if rem > 0:
                time.sleep(rem)
            self.root.after(0, lambda: self._on_ai_done(current_game_id, chosen_move, stats))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ai_done(
        self,
        worker_game_id: int,
        chosen_move: Optional[Move],
        stats: Optional[SearchStats],
    ) -> None:
        if worker_game_id != self.game_id:
            return
        self.ai_thinking = False
        if chosen_move is not None and adjudicate_state(self.state) is None:
            self._execute_move(chosen_move, stats=stats)

    def _update_telemetry_from_stats(self, st: SearchStats) -> None:
        if abs(st.score) > MATE_SCORE - 1000:
            plies_to_mate = MATE_SCORE - abs(st.score)
            sign = "+" if st.score > 0 else "-"
            self.telemetry_vars["score_str"].set(f"{sign}Mate in {plies_to_mate} plies")
        else:
            self.telemetry_vars["score_str"].set(f"{st.score:+d} cp")

        self.telemetry_vars["depth_str"].set(f"d = {st.depth}")
        self.telemetry_vars["nodes_str"].set(f"{st.nodes:,}  (Q: {st.q_nodes:,})")
        self.telemetry_vars["bstar_str"].set(f"{st.effective_branching_factor:.2f}")
        self.telemetry_vars["time_str"].set(f"{st.elapsed_sec * 1000:.1f} ms  ({st.nps:,} NPS)")
        tt_pct = f"{st.tt_hit_rate * 100:.1f}% ({st.tt_hits}/{st.tt_probes})" if st.tt_probes > 0 else "N/A"
        self.telemetry_vars["tt_str"].set(tt_pct)
        self.telemetry_vars["pv_str"].set(
            " ".join(st.pv[:5]) if st.pv else (st.best_move.uci() if st.best_move else "—")
        )

    def _refresh_all(self) -> None:
        w_eval = evaluate_tuned_white(self.state)
        self._draw_eval_bar(w_eval)
        self.telemetry_vars["eval_str"].set(f"{w_eval:+d} cp ({w_eval / 100.0:+.2f})")
        self.telemetry_vars["clock_str"].set(
            f"{self.state.halfmove_clock} / 100  (Ply {len(self.state.history)})"
        )

        board_to_draw, _, _, _ = self._get_active_board_and_move()
        self._update_material_bar(board_to_draw)
        self._draw_board()

        # Update Step Narration Banner
        if self.view_ply is not None:
            if self.view_ply == 0:
                self.step_label.configure(
                    text=f"[REPLAY 0/{len(self.snapshots)}] Initial Starting Position", fg=ACCENT_BLUE
                )
            else:
                snap = self.snapshots[self.view_ply - 1]
                self.step_label.configure(
                    text=f"[REPLAY {self.view_ply}/{len(self.snapshots)}] {snap.step_text}", fg=ACCENT_BLUE
                )
        elif self.snapshots:
            self.step_label.configure(text=self.snapshots[-1].step_text, fg=ACCENT_AMBER)
        else:
            self.step_label.configure(text="Step 0: Starting Position — Ready to play", fg=ACCENT_AMBER)

        # Update Top Status Label (with Explicit Winner & Loser on Game End)
        verdict = adjudicate_state(self.state)
        if verdict is not None:
            winner, reason = verdict
            w_name = short_label(self.white_choice.get())
            b_name = short_label(self.black_choice.get())
            if winner == WHITE:
                msg = f"🏆 WHITE ({w_name}) WINS!  |  ❌ BLACK ({b_name}) LOSES  [{reason.title()}]"
                color = ACCENT_GREEN
            elif winner == BLACK:
                msg = f"🏆 BLACK ({b_name}) WINS!  |  ❌ WHITE ({w_name}) LOSES  [{reason.title()}]"
                color = ACCENT_GREEN
            else:
                msg = f"½–½ DRAW ({w_name} vs {b_name}) — {reason.replace('_', ' ').title()}"
                color = ACCENT_AMBER
            self.status_label.configure(text=msg, fg=color)
        else:
            side = "White (+1)" if self.state.turn == WHITE else "Black (-1)"
            if self.ai_paused and not self._is_human_turn():
                self.status_label.configure(
                    text=f"⏸ AI Paused — {side} to move (Click 'Step AI Move' or 'Resume AI')",
                    fg=ACCENT_AMBER,
                )
            elif self.state.in_check():
                self.status_label.configure(text=f"CHECK! {side} to move", fg=ACCENT_RED)
            else:
                self.status_label.configure(text=f"{side} to move", fg=FG_TEXT)

        self.history_text.configure(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        lines = []
        for idx in range(0, len(self.move_uci_history), 2):
            move_num = (idx // 2) + 1
            w_m = self.move_uci_history[idx]
            b_m = self.move_uci_history[idx + 1] if idx + 1 < len(self.move_uci_history) else ""
            lines.append(f"{move_num:>2}. {w_m:<8} {b_m:<8}")
        self.history_text.insert(tk.END, "\n".join(lines))
        self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)

    def _on_new_game(self) -> None:
        self.game_id += 1
        self.ai_thinking = False
        self.step_once = False
        self.state = GameState()
        self.initial_board = list(self.state.board)
        self.move_uci_history.clear()
        self.snapshots.clear()
        self.view_ply = None
        self.last_move = None
        self.selected_sq = None
        self.legal_moves_for_selected.clear()

        if self.white_agent is not None:
            self.white_agent.reset()
        if self.black_agent is not None:
            self.black_agent.reset()

        for key in ("score_str", "depth_str", "nodes_str", "bstar_str", "time_str", "tt_str", "pv_str"):
            self.telemetry_vars[key].set("—")

        self._refresh_all()
        self._schedule_ai_if_needed()

    def _on_undo(self) -> None:
        if not self.move_uci_history:
            return
        self.game_id += 1
        self.ai_thinking = False
        self.view_ply = None

        if self.white_agent is not None and self.black_agent is not None:
            self.ai_paused = True
            self.pause_btn.configure(text="Resume AI", bg=ACCENT_GREEN)

        human_vs_ai = (self.white_agent is None) != (self.black_agent is None)
        plies_to_undo = 2 if (human_vs_ai and len(self.move_uci_history) >= 2) else 1

        for _ in range(plies_to_undo):
            if self.move_uci_history:
                self.state.unmake_move()
                self.move_uci_history.pop()
                if self.snapshots:
                    self.snapshots.pop()

        self.last_move = self.state._undo_stack[-1][0] if self.state._undo_stack else None
        self.selected_sq = None
        self.legal_moves_for_selected.clear()
        self._refresh_all()

    def _on_flip(self) -> None:
        self.flipped = not self.flipped
        self._refresh_all()

    def _on_prev_step(self) -> None:
        if not self.snapshots:
            return
        if self.view_ply is None:
            self.view_ply = max(0, len(self.snapshots) - 1)
        else:
            self.view_ply = max(0, self.view_ply - 1)
        if self.view_ply > 0 and self.snapshots[self.view_ply - 1].stats is not None:
            self._update_telemetry_from_stats(self.snapshots[self.view_ply - 1].stats)
        self._refresh_all()

    def _on_next_step(self) -> None:
        if self.view_ply is None:
            return
        if self.view_ply < len(self.snapshots) - 1:
            self.view_ply += 1
            if self.snapshots[self.view_ply - 1].stats is not None:
                self._update_telemetry_from_stats(self.snapshots[self.view_ply - 1].stats)
        else:
            self.view_ply = None
        self._refresh_all()

    def _on_step_ai(self) -> None:
        """Advance AI vs AI by one single move and remain paused for step-by-step inspection."""
        if self.view_ply is not None:
            self._on_next_step()
            return
        self.ai_paused = True
        self.pause_btn.configure(text="Resume AI", bg=ACCENT_GREEN)
        self.step_once = True
        self._schedule_ai_if_needed()

    def _on_hint(self) -> None:
        if self.ai_thinking or adjudicate_state(self.state) is not None:
            return
        self.view_ply = None
        state_copy = self.state.copy()
        best_move = self.hint_agent.select_move(state_copy)
        st = self.hint_agent.last_stats
        if st is not None:
            self._update_telemetry_from_stats(st)
        if best_move is not None:
            self.selected_sq = best_move.from_sq
            self.legal_moves_for_selected = [best_move]
            self.status_label.configure(
                text=f"Hint: Best move is {best_move.uci()} (Score: {st.score:+d} cp)",
                fg=ACCENT_GREEN,
            )
            self._draw_board()

    def _on_toggle_pause(self) -> None:
        self.ai_paused = not self.ai_paused
        if self.ai_paused:
            self.pause_btn.configure(text="Resume AI", bg=ACCENT_GREEN)
            self._refresh_all()
        else:
            self.view_ply = None
            self.pause_btn.configure(text="Pause AI", bg=ACCENT_AMBER)
            self._refresh_all()
            self._schedule_ai_if_needed()

    def _on_player_changed(self, _event: Optional[tk.Event] = None) -> None:
        self.game_id += 1
        self.ai_thinking = False
        self.white_agent = create_agent_from_label(self.white_choice.get())
        self.black_agent = create_agent_from_label(self.black_choice.get())
        self._refresh_all()
        self._schedule_ai_if_needed()


def main() -> None:
    root = tk.Tk()
    MiniChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
