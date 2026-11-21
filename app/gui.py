"""
Gardner's 5x5 Minichess - Smooth High-DPI Desktop GUI Studio (app/gui.py)

Performance & Visual Architecture:
  - Pre-baked 1-item static board background + pre-warmed 3x Staunton sprites (0 ms frame cache misses)
  - Zero-redraw 60 FPS piece-sliding animations & Drag-and-Drop via Canvas item `coords()`
  - Zero GIL contention: AI search starts strictly after the 120 ms slide animation completes
  - Cached terminal adjudication (no move generation during redraws)
  - Chess.com-style 3-tier Move Quality Grading: Great (!), Good (✓), Mistake (?)
    displayed as circular badges on the destination square, Step Banner, Counter, and Move History
  - Async non-blocking Hint (d=4+Q) with emerald Hint Arrow & toggleable dashed PV Reply Arrow
  - Animated Evaluation Gauge + clickable Advantage Sparkline + clickable Move History + Shortcuts
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base_agent import Agent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.search_agent import SearchAgent
from agents.tuned_agent import TunedSearchAgent
from app.move_grader import MoveGrade, grade_move
from app.ui_assets import (
    BOARD_THEMES,
    GRADE_COLORS,
    BoardTheme,
    PieceSpriteCache,
    SoundEffects,
)
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

# Catppuccin Macchiato-inspired dark palette
BG_DARK = "#151722"
BG_PANEL = "#202334"
BG_CARD = "#1a1c2b"
BG_INPUT = "#262a3e"
BORDER_SUBTLE = "#333852"
FG_TEXT = "#d8e0fa"
FG_SUBTEXT = "#98a2c4"
ACCENT_BLUE = "#8aadf4"
ACCENT_GREEN = "#a6da95"
ACCENT_AMBER = "#eed49f"
ACCENT_RED = "#ed8796"

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
    "Plain Minimax (Negamax d=3)",
    "GreedyBot (1-Ply)",
    "RandomBot",
)


def _shift_hex(hex_color: str, delta: int) -> str:
    h = hex_color.lstrip("#")
    r = max(0, min(255, int(h[0:2], 16) + delta))
    g = max(0, min(255, int(h[2:4], 16) + delta))
    b = max(0, min(255, int(h[4:6], 16) + delta))
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class StepSnapshot:
    ply: int
    move: Move
    uci: str
    step_text: str
    board: List[int]
    turn: int
    in_check: bool
    eval_cp: int
    stats: Optional[SearchStats] = None
    grade: Optional[MoveGrade] = None


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
    if label == "Plain Minimax (Negamax d=3)":
        return SearchAgent(
            max_depth=3,
            mode="negamax",
            use_move_ordering=False,
            use_tt=False,
            use_iterative_deepening=False,
            name="Minimax(d=3)",
        )
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
    FRAME_PAD = 14
    DEFAULT_SQ_PX = 96

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gardner's 5×5 Minichess — AI Arena & Engine Studio")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.root.minsize(930, 700)

        self.sq_px: int = self.DEFAULT_SQ_PX
        self.board_px: int = self.sq_px * 5 + 2 * self.FRAME_PAD
        self._resize_after_id: Optional[str] = None

        self.theme_name = tk.StringVar(value="Classic Wood")
        self.theme: BoardTheme = BOARD_THEMES["Classic Wood"]
        self.sprites = PieceSpriteCache()
        self.sounds = SoundEffects(enabled=True)
        self.sound_enabled = True

        self.state = GameState()
        self.initial_board = list(self.state.board)
        self.current_verdict: Optional[Tuple[int, str]] = None
        self.move_uci_history: List[str] = []
        self.snapshots: List[StepSnapshot] = []
        self.eval_history: List[int] = [0]
        self.view_ply: Optional[int] = None  # None = live board; int = reviewing step

        self.last_move: Optional[Move] = None
        self.selected_sq: Optional[int] = None
        self.legal_moves_for_selected: List[Move] = []
        self.hint_move: Optional[Move] = None
        self.pv_reply_move: Optional[Move] = None
        self.show_pv_arrow = tk.BooleanVar(value=True)

        self.flipped: bool = False
        self.ai_paused: bool = False
        self.step_once: bool = False
        self.ai_thinking: bool = False
        self.hint_thinking: bool = False
        self.game_id: int = 0
        self.gameover_overlay_hidden: bool = False
        self._gameover_dismiss_rect: Optional[Tuple[int, int, int, int]] = None
        self.ai_delay_var = tk.DoubleVar(value=0.8)

        # Zero-redraw Drag-and-Drop state
        self._drag_from_sq: Optional[int] = None
        self._drag_piece: int = EMPTY
        self._drag_start_xy: Tuple[int, int] = (0, 0)
        self._drag_active: bool = False
        self._drag_hover_sq: Optional[int] = None
        self._drag_item_id: Optional[int] = None
        self._drag_hover_rect_id: Optional[int] = None

        # Zero-redraw 60 FPS piece slide animation state
        self._anim_after_id: Optional[str] = None
        self._anim_item_id: Optional[int] = None
        self._anim_piece: int = EMPTY
        self._anim_from_sq: int = -1
        self._anim_to_sq: int = -1
        self._anim_start_xy: Tuple[int, int] = (0, 0)
        self._anim_end_xy: Tuple[int, int] = (0, 0)
        self._anim_start_time: float = 0.0
        self._anim_duration: float = 0.12
        self._anim_active: bool = False
        self._pending_after_anim: Optional[Callable[[], None]] = None

        # Smooth animated evaluation bar state
        self._eval_display_cp: float = 0.0
        self._eval_target_cp: float = 0.0
        self._eval_anim_id: Optional[str] = None

        self.white_choice = tk.StringVar(value=PLAYER_OPTIONS[0])
        self.black_choice = tk.StringVar(value=PLAYER_OPTIONS[1])
        self.white_agent: Optional[Agent] = create_agent_from_label(self.white_choice.get())
        self.black_agent: Optional[Agent] = create_agent_from_label(self.black_choice.get())
        self.hint_agent = TunedSearchAgent(max_depth=4, use_quiescence=True, name="Analyzer(d=4+Q)")

        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()

        self._configure_styles()
        self._build_ui()
        # Pre-warm all sprites, baked wood board, glow, and grade badges once for 0-stutter frames
        self.sprites.prewarm(self.theme, self.sq_px, self.FRAME_PAD)
        self._bind_shortcuts()
        self._refresh_all()
        self._poll_ui_queue()
        self._schedule_ai_if_needed()

    def _poll_ui_queue(self) -> None:
        """Safely drain callbacks queued by background AI / Hint / Grader threads on the UI thread."""
        while True:
            try:
                cb = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            cb()
        if self.root.winfo_exists():
            self.root.after(20, self._poll_ui_queue)

    # ------------------------------------------------------------------
    # STYLING & UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Dark.TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_PANEL,
            foreground=FG_TEXT,
            arrowcolor=ACCENT_BLUE,
            bordercolor=BORDER_SUBTLE,
            lightcolor=BORDER_SUBTLE,
            darkcolor=BG_CARD,
            selectbackground=BG_INPUT,
            selectforeground=FG_TEXT,
            padding=4,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", BG_INPUT), ("disabled", BG_CARD)],
            foreground=[("readonly", FG_TEXT), ("disabled", FG_SUBTEXT)],
            background=[("active", "#31364f"), ("readonly", BG_PANEL)],
            selectbackground=[("readonly", BG_INPUT)],
            selectforeground=[("readonly", FG_TEXT)],
        )

        self.root.option_add("*TCombobox*Listbox.background", BG_CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG_TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#394266")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))

        style.configure(
            "Dark.Vertical.TScrollbar",
            background="#363c57",
            troughcolor=BG_CARD,
            bordercolor=BG_CARD,
            arrowcolor=FG_SUBTEXT,
            relief="flat",
        )
        style.map("Dark.Vertical.TScrollbar", background=[("active", "#4b5378")])

        style.configure(
            "Dark.Horizontal.TScale",
            background=ACCENT_AMBER,
            troughcolor=BG_CARD,
            bordercolor=BG_PANEL,
            lightcolor=ACCENT_AMBER,
            darkcolor=ACCENT_AMBER,
        )

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------- LEFT COLUMN (Board + Controls) ----------------
        self.left_col = tk.Frame(main_frame, bg=BG_DARK)
        self.left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        # Primary Status Banner (Turn / Hint / Winner & Loser)
        self.status_label = tk.Label(
            self.left_col,
            text="White (+1) to move",
            font=("Segoe UI", 12, "bold"),
            bg=BG_PANEL,
            fg=FG_TEXT,
            pady=7,
            padx=12,
            anchor="w",
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
        )
        self.status_label.pack(fill=tk.X, pady=(0, 4))

        # Step-by-Step Narration Banner + Move Quality Pill
        step_bar = tk.Frame(self.left_col, bg=BG_CARD)
        step_bar.pack(fill=tk.X, pady=(0, 8))

        self.step_label = tk.Label(
            step_bar,
            text="Step 0: Starting Position — Click or drag any piece to play",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=ACCENT_AMBER,
            pady=5,
            padx=12,
            anchor="w",
        )
        self.step_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.grade_pill_label = tk.Label(
            step_bar,
            text="",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD,
            fg=FG_TEXT,
            padx=10,
            pady=3,
        )
        self.grade_pill_label.pack(side=tk.RIGHT, padx=(4, 8), pady=3)

        # Resizable Board + Eval Bar Host
        self.board_host = tk.Frame(self.left_col, bg=BG_DARK)
        self.board_host.pack(fill=tk.BOTH, expand=True)
        self.board_host.bind("<Configure>", self._on_board_host_resize)

        self.board_row = tk.Frame(self.board_host, bg=BG_DARK)
        self.board_row.place(relx=0.5, rely=0.5, anchor="center")

        self.eval_canvas = tk.Canvas(
            self.board_row,
            width=34,
            height=self.board_px,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
        )
        self.eval_canvas.pack(side=tk.LEFT, padx=(0, 10))

        self.board_canvas = tk.Canvas(
            self.board_row,
            width=self.board_px,
            height=self.board_px,
            bg=BG_PANEL,
            highlightthickness=0,
        )
        self.board_canvas.pack(side=tk.LEFT)
        self.board_canvas.bind("<ButtonPress-1>", self._on_board_press)
        self.board_canvas.bind("<B1-Motion>", self._on_board_drag)
        self.board_canvas.bind("<ButtonRelease-1>", self._on_board_release)

        # Captured Pieces & Material Balance Bar
        self.material_label = tk.Label(
            self.left_col,
            text="Captured by White: None     |     Captured by Black: None",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=FG_SUBTEXT,
            pady=5,
            padx=10,
            anchor="w",
        )
        self.material_label.pack(fill=tk.X, pady=(8, 0))

        # Row 1 Buttons: Game Controls
        btn_row = tk.Frame(self.left_col, bg=BG_DARK, pady=7)
        btn_row.pack(fill=tk.X)

        self._make_button(btn_row, "New Game", self._on_new_game, bg=ACCENT_BLUE, fg="#151722").pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self._make_button(btn_row, "Undo", self._on_undo, bg="#2e334a", fg=FG_TEXT).pack(
            side=tk.LEFT, padx=4
        )
        self._make_button(btn_row, "Flip Board", self._on_flip, bg="#2e334a", fg=FG_TEXT).pack(
            side=tk.LEFT, padx=4
        )
        self.hint_btn = self._make_button(
            btn_row, "Hint (d=4+Q)", self._on_hint, bg=ACCENT_GREEN, fg="#151722"
        )
        self.hint_btn.pack(side=tk.LEFT, padx=4)

        self.sound_btn = self._make_button(
            btn_row, "🔊 Sound", self._on_toggle_sound, bg="#2e334a", fg=FG_TEXT
        )
        self.sound_btn.pack(side=tk.RIGHT)

        # Row 2 Buttons: Step-by-Step AI & Replay Controls
        step_row = tk.Frame(self.left_col, bg=BG_DARK)
        step_row.pack(fill=tk.X)

        self._make_button(step_row, "◀ Prev", self._on_prev_step, bg=BG_PANEL, fg=FG_TEXT).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._make_button(step_row, "Next ▶", self._on_next_step, bg=BG_PANEL, fg=FG_TEXT).pack(
            side=tk.LEFT, padx=4
        )
        self._make_button(step_row, "⏭ Step AI Move", self._on_step_ai, bg=ACCENT_GREEN, fg="#151722").pack(
            side=tk.LEFT, padx=6
        )
        self.pause_btn = self._make_button(
            step_row, "Pause AI", self._on_toggle_pause, bg=ACCENT_AMBER, fg="#151722"
        )
        self.pause_btn.pack(side=tk.RIGHT)

        # Shortcut legend footer
        tk.Label(
            self.left_col,
            text="Shortcuts: ←/→ Replay  •  Space Pause AI  •  S Step AI  •  H Hint  •  F Flip  •  Ctrl+Z Undo  •  M Sound",
            font=("Segoe UI", 8),
            bg=BG_DARK,
            fg="#6e7799",
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 0))

        # ---------------- RIGHT COLUMN ----------------
        right_col = tk.Frame(main_frame, bg=BG_DARK, width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)
        right_col.pack_propagate(False)

        # Card 1: Match Setup, Theme & Pacing
        player_card = tk.LabelFrame(
            right_col,
            text=" Match Setup & Studio Options ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
            padx=11,
            pady=7,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
            bd=0,
        )
        player_card.pack(fill=tk.X, pady=(0, 8))
        player_card.columnconfigure(1, weight=1)

        tk.Label(player_card, text="White (+1):", font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_TEXT).grid(
            row=0, column=0, sticky="w", pady=2
        )
        w_menu = ttk.Combobox(
            player_card,
            textvariable=self.white_choice,
            values=PLAYER_OPTIONS,
            state="readonly",
            style="Dark.TCombobox",
            width=29,
        )
        w_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        w_menu.bind("<<ComboboxSelected>>", self._on_player_changed)

        tk.Label(player_card, text="Black (-1):", font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_TEXT).grid(
            row=1, column=0, sticky="w", pady=2
        )
        b_menu = ttk.Combobox(
            player_card,
            textvariable=self.black_choice,
            values=PLAYER_OPTIONS,
            state="readonly",
            style="Dark.TCombobox",
            width=29,
        )
        b_menu.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)
        b_menu.bind("<<ComboboxSelected>>", self._on_player_changed)

        tk.Label(player_card, text="Board Theme:", font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_SUBTEXT).grid(
            row=2, column=0, sticky="w", pady=2
        )
        theme_row = tk.Frame(player_card, bg=BG_PANEL)
        theme_row.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=2)

        theme_menu = ttk.Combobox(
            theme_row,
            textvariable=self.theme_name,
            values=tuple(BOARD_THEMES.keys()),
            state="readonly",
            style="Dark.TCombobox",
            width=16,
        )
        theme_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        theme_menu.bind("<<ComboboxSelected>>", self._on_theme_changed)

        pv_chk = tk.Checkbutton(
            theme_row,
            text="PV Arrow",
            variable=self.show_pv_arrow,
            command=self._draw_board,
            font=("Segoe UI", 8, "bold"),
            bg=BG_PANEL,
            fg=FG_SUBTEXT,
            activebackground=BG_PANEL,
            activeforeground=FG_TEXT,
            selectcolor=BG_CARD,
            cursor="hand2",
        )
        pv_chk.pack(side=tk.RIGHT, padx=(6, 0))

        tk.Label(player_card, text="AI Step Delay:", font=("Segoe UI", 9), bg=BG_PANEL, fg=FG_SUBTEXT).grid(
            row=3, column=0, sticky="w", pady=(4, 1)
        )
        delay_frame = tk.Frame(player_card, bg=BG_PANEL)
        delay_frame.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(4, 1))

        self.delay_val_label = tk.Label(
            delay_frame, text="0.8s", font=("Consolas", 9, "bold"), bg=BG_PANEL, fg=ACCENT_AMBER, width=5
        )
        self.delay_val_label.pack(side=tk.RIGHT)

        delay_scale = ttk.Scale(
            delay_frame,
            variable=self.ai_delay_var,
            from_=0.2,
            to=2.0,
            orient=tk.HORIZONTAL,
            style="Dark.Horizontal.TScale",
            command=self._on_delay_changed,
        )
        delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Card 2: Live Engine Search Telemetry
        telemetry_card = tk.LabelFrame(
            right_col,
            text=" Engine Search Telemetry ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_GREEN,
            padx=11,
            pady=6,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
            bd=0,
        )
        telemetry_card.pack(fill=tk.X, pady=(0, 8))

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
                font=("Segoe UI", 8),
                bg=BG_PANEL,
                fg=FG_SUBTEXT,
            ).grid(row=row_idx, column=0, sticky="w", pady=1)
            var = tk.StringVar(value="—")
            self.telemetry_vars[key] = var
            tk.Label(
                telemetry_card,
                textvariable=var,
                font=("Consolas", 9, "bold"),
                bg=BG_PANEL,
                fg=FG_TEXT,
                anchor="w",
            ).grid(row=row_idx, column=1, sticky="w", padx=(10, 0), pady=1)

        # Card 3: Advantage Sparkline (Clickable Eval History)
        spark_card = tk.LabelFrame(
            right_col,
            text=" Advantage Graph (Click to Jump) ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
            bd=0,
        )
        spark_card.pack(fill=tk.X, pady=(0, 8))

        self.spark_canvas = tk.Canvas(
            spark_card,
            height=58,
            bg=BG_CARD,
            highlightthickness=0,
            cursor="hand2",
        )
        self.spark_canvas.pack(fill=tk.X)
        self.spark_canvas.bind("<Button-1>", self._on_sparkline_click)
        self.spark_canvas.bind("<Configure>", lambda _e: self._draw_sparkline())

        # Card 4: Move Quality Summary & Clickable Move History
        history_card = tk.LabelFrame(
            right_col,
            text=" Move Review & History (Click Move to Inspect) ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_AMBER,
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
            bd=0,
        )
        history_card.pack(fill=tk.BOTH, expand=True)

        # Compact Chess.com-style Grade Summary Strip (Great / Good / Mistake ONLY)
        grade_strip = tk.Frame(history_card, bg=BG_CARD, pady=4, padx=8)
        grade_strip.pack(fill=tk.X, pady=(0, 5))

        self.great_cnt_lbl = tk.Label(
            grade_strip, text="! Great: 0", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=GRADE_COLORS["Great"][0]
        )
        self.great_cnt_lbl.pack(side=tk.LEFT, padx=(0, 14))

        self.good_cnt_lbl = tk.Label(
            grade_strip, text="✓ Good: 0", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=GRADE_COLORS["Good"][0]
        )
        self.good_cnt_lbl.pack(side=tk.LEFT, padx=(0, 14))

        self.mistake_cnt_lbl = tk.Label(
            grade_strip, text="? Mistake: 0", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=GRADE_COLORS["Mistake"][0]
        )
        self.mistake_cnt_lbl.pack(side=tk.LEFT)

        hist_inner = tk.Frame(history_card, bg=BG_CARD)
        hist_inner.pack(fill=tk.BOTH, expand=True)

        self.history_text = tk.Text(
            hist_inner,
            width=36,
            height=6,
            font=("Consolas", 9),
            bg=BG_CARD,
            fg=FG_TEXT,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="arrow",
        )
        self.history_text.tag_configure("move_num", foreground=FG_SUBTEXT)
        self.history_text.tag_configure(
            "active_move", background="#384066", foreground=ACCENT_AMBER, font=("Consolas", 9, "bold")
        )
        self.history_text.tag_configure("grade_Great", foreground=GRADE_COLORS["Great"][0], font=("Consolas", 9, "bold"))
        self.history_text.tag_configure("grade_Good", foreground=GRADE_COLORS["Good"][0], font=("Consolas", 9, "bold"))
        self.history_text.tag_configure("grade_Mistake", foreground=GRADE_COLORS["Mistake"][0], font=("Consolas", 9, "bold"))

        scrollbar = ttk.Scrollbar(
            hist_inner,
            orient=tk.VERTICAL,
            style="Dark.Vertical.TScrollbar",
            command=self.history_text.yview,
        )
        self.history_text.configure(yscrollcommand=scrollbar.set)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _make_button(
        self, parent: tk.Widget, text: str, command, bg: str, fg: str
    ) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg=fg,
            activebackground=_shift_hex(bg, -12),
            activeforeground=fg,
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2",
            bd=0,
        )
        btn._base_bg = bg  # type: ignore[attr-defined]

        def _on_enter(_e):
            base = getattr(btn, "_base_bg", bg)
            btn.configure(bg=_shift_hex(base, 18))

        def _on_leave(_e):
            base = getattr(btn, "_base_bg", bg)
            btn.configure(bg=base)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)
        return btn

    def _set_button_style(self, btn: tk.Button, text: str, bg: str, fg: Optional[str] = None) -> None:
        btn._base_bg = bg  # type: ignore[attr-defined]
        cfg = {"text": text, "bg": bg, "activebackground": _shift_hex(bg, -12)}
        if fg is not None:
            cfg["fg"] = fg
            cfg["activeforeground"] = fg
        btn.configure(**cfg)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _e: self._on_prev_step())
        self.root.bind("<Right>", lambda _e: self._on_next_step())
        self.root.bind("<space>", lambda _e: self._on_toggle_pause())
        self.root.bind("<Key-h>", lambda _e: self._on_hint())
        self.root.bind("<Key-H>", lambda _e: self._on_hint())
        self.root.bind("<Key-f>", lambda _e: self._on_flip())
        self.root.bind("<Key-F>", lambda _e: self._on_flip())
        self.root.bind("<Key-s>", lambda _e: self._on_step_ai())
        self.root.bind("<Key-S>", lambda _e: self._on_step_ai())
        self.root.bind("<Key-m>", lambda _e: self._on_toggle_sound())
        self.root.bind("<Key-M>", lambda _e: self._on_toggle_sound())
        self.root.bind("<Control-z>", lambda _e: self._on_undo())
        self.root.bind("<Control-Z>", lambda _e: self._on_undo())
        self.root.bind("<Control-n>", lambda _e: self._on_new_game())
        self.root.bind("<Control-N>", lambda _e: self._on_new_game())

    # ------------------------------------------------------------------
    # RESPONSIVE BOARD GEOMETRY
    # ------------------------------------------------------------------

    def _on_board_host_resize(self, event: tk.Event) -> None:
        avail_w = max(200, event.width - 48)
        avail_h = max(200, event.height - 4)
        target_board = min(avail_w, avail_h)
        new_sq = max(68, min(136, (target_board - 2 * self.FRAME_PAD) // 5))
        if new_sq == self.sq_px:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(90, lambda s=new_sq: self._apply_board_size(s))

    def _apply_board_size(self, new_sq: int) -> None:
        self._resize_after_id = None
        if new_sq == self.sq_px:
            return
        self._cancel_animation()
        self.sq_px = new_sq
        self.board_px = self.sq_px * 5 + 2 * self.FRAME_PAD
        self.sprites.prewarm(self.theme, self.sq_px, self.FRAME_PAD)
        self.board_canvas.configure(width=self.board_px, height=self.board_px)
        self.eval_canvas.configure(height=self.board_px)
        self._draw_board()
        self._draw_eval_bar(self._eval_display_cp)

    def _sq_to_canvas_xy(self, sq: int) -> Tuple[int, int, int, int]:
        r, f = sq_to_coords(sq)
        if not self.flipped:
            vis_row = 4 - r
            vis_col = f
        else:
            vis_row = r
            vis_col = 4 - f
        x0 = self.FRAME_PAD + vis_col * self.sq_px
        y0 = self.FRAME_PAD + vis_row * self.sq_px
        return x0, y0, x0 + self.sq_px, y0 + self.sq_px

    def _sq_center(self, sq: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def _canvas_xy_to_sq(self, x: int, y: int) -> Optional[int]:
        vis_col = (x - self.FRAME_PAD) // self.sq_px
        vis_row = (y - self.FRAME_PAD) // self.sq_px
        if not (0 <= vis_col < 5 and 0 <= vis_row < 5):
            return None
        if not self.flipped:
            r = 4 - vis_row
            f = vis_col
        else:
            r = vis_row
            f = 4 - vis_col
        return coords_to_sq(r, f)

    def _get_active_snapshot_data(
        self,
    ) -> Tuple[List[int], Optional[Move], int, bool, Optional[MoveGrade]]:
        if self.view_ply is None or not self.snapshots:
            last_grade = self.snapshots[-1].grade if self.snapshots else None
            return (
                self.state.board,
                self.last_move,
                self.state.turn,
                self.state.in_check(self.state.turn),
                last_grade,
            )
        if self.view_ply == 0:
            return self.initial_board, None, WHITE, False, None
        snap = self.snapshots[min(self.view_ply - 1, len(self.snapshots) - 1)]
        return snap.board, snap.move, snap.turn, snap.in_check, snap.grade

    # ------------------------------------------------------------------
    # FAST LAYERED BOARD, PIECES, BADGES & OVERLAY RENDERING
    # ------------------------------------------------------------------

    def _create_piece_item(self, cx: int, cy: int, piece: int, tags: Tuple[str, ...]) -> int:
        """Create a single cached piece sprite on the canvas (always uses prewarmed size)."""
        c = self.board_canvas
        px = max(24, int(round(self.sq_px * 0.90)))
        sprite = self.sprites.get_piece(piece, px)
        if sprite is not None:
            return c.create_image(cx, cy, image=sprite, tags=tags)

        # Pure-Tkinter Unicode fallback (single item so coords() animates/drags cleanly)
        glyph = PIECE_GLYPHS[abs(piece)]
        font_sz = max(18, int(round(self.sq_px * 0.44)))
        color = "#ffffff" if piece > 0 else "#191924"
        return c.create_text(cx, cy, text=glyph, font=("Segoe UI Symbol", font_sz), fill=color, tags=tags)

    def _draw_grade_badge_on_sq(self, sq: int, grade: MoveGrade) -> None:
        """Draw the Chess.com-style circular badge (! / ✓ / ?) at the top-right corner of `sq`."""
        c = self.board_canvas
        x0, y0, x1, _y1 = self._sq_to_canvas_xy(sq)
        diam = max(18, int(round(self.sq_px * 0.30)))
        bx = x1 - diam // 2 - 2
        by = y0 + diam // 2 + 2

        badge_img = self.sprites.get_grade_badge(grade.label, diam)
        if badge_img is not None:
            c.create_image(bx, by, image=badge_img, tags=("badge",))
            return

        color_hex, sym, _ = GRADE_COLORS.get(grade.label, GRADE_COLORS["Good"])
        r = diam // 2
        c.create_oval(bx - r, by - r, bx + r, by + r, fill=color_hex, outline="#ffffff", width=2, tags=("badge",))
        c.create_text(bx, by, text=sym, font=("Segoe UI", max(9, r - 2), "bold"), fill="#ffffff", tags=("badge",))

    def _draw_board(self, hide_sq: Optional[int] = None) -> None:
        c = self.board_canvas
        c.delete("all")
        self._gameover_dismiss_rect = None
        self._drag_item_id = None
        self._drag_hover_rect_id = None

        t = self.theme
        board_to_draw, active_move, active_turn, is_in_check, active_grade = self._get_active_snapshot_data()

        # 1. Single-blit baked static board (frame + all 25 squares with wood grain)
        bp = self.board_px
        fp = self.FRAME_PAD
        bg_img = self.sprites.get_board_bg(t, self.sq_px, fp)
        if bg_img is not None:
            c.create_image(0, 0, anchor="nw", image=bg_img, tags=("bg",))
        else:
            c.create_rectangle(0, 0, bp, bp, fill=t.frame_outer, outline="", tags=("bg",))
            c.create_rectangle(2, 2, bp - 2, bp - 2, outline=t.frame_accent, width=1, tags=("bg",))
            for sq in range(25):
                r, f = sq_to_coords(sq)
                x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
                fill_c = t.light_sq if ((r + f) % 2 == 1) else t.dark_sq
                c.create_rectangle(x0, y0, x1, y1, fill=fill_c, outline="", tags=("bg",))

        checked_king_sq = -1
        if is_in_check:
            target_k = KING * active_turn
            for s_idx, p_val in enumerate(board_to_draw):
                if p_val == target_k:
                    checked_king_sq = s_idx
                    break

        legal_targets = {m.to_sq: m for m in self.legal_moves_for_selected} if self.view_ply is None else {}

        # 2. Dynamic square highlights (only on squares that actually have a highlight!)
        for sq in range(25):
            r, f = sq_to_coords(sq)
            x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
            is_light = (r + f) % 2 == 1

            if self.view_ply is None and sq == self.selected_sq:
                c.create_rectangle(x0, y0, x1, y1, fill=t.selected_sq, outline="", tags=("highlights",))
            elif active_move is not None and sq in (active_move.from_sq, active_move.to_sq):
                fill_col = t.last_move_light if is_light else t.last_move_dark
                c.create_rectangle(x0, y0, x1, y1, fill=fill_col, outline="", tags=("highlights",))

            if sq == checked_king_sq:
                glow = self.sprites.get_check_glow(self.sq_px)
                if glow is not None:
                    c.create_image(x0, y0, anchor="nw", image=glow, tags=("highlights",))
                else:
                    c.create_rectangle(x0, y0, x1, y1, fill=t.check_sq, outline="", tags=("highlights",))

            # Crisp ClearType in-square edge coordinates
            vis_col = f if not self.flipped else (4 - f)
            vis_row = (4 - r) if not self.flipped else r
            coord_col = t.coord_on_light if is_light else t.coord_on_dark
            coord_font = ("Segoe UI", max(8, self.sq_px // 10), "bold")
            if vis_col == 0:
                c.create_text(x0 + 5, y0 + 4, anchor="nw", text=RANKS[r], font=coord_font, fill=coord_col, tags=("coords",))
            if vis_row == 4:
                c.create_text(x1 - 5, y1 - 3, anchor="se", text=FILES[f], font=coord_font, fill=coord_col, tags=("coords",))

        # 3. Pieces (each tagged individually so Drag-and-Drop can hide/show without redraw!)
        for sq in range(25):
            piece = board_to_draw[sq]
            if piece != EMPTY and sq != hide_sq:
                cx, cy = self._sq_center(sq)
                self._create_piece_item(cx, cy, piece, tags=("pieces", f"piece_sq_{sq}"))

        # 4. Legal move dots / capture rings
        for sq in legal_targets:
            x0, y0, x1, y1 = self._sq_to_canvas_xy(sq)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if board_to_draw[sq] == EMPTY:
                rad = max(8, self.sq_px // 8)
                c.create_oval(
                    cx - rad, cy - rad, cx + rad, cy + rad,
                    fill=t.legal_dot, outline="#ffffff", width=1, tags=("markers",),
                )
            else:
                pad = max(4, self.sq_px // 16)
                c.create_oval(
                    x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                    outline=t.legal_ring, width=max(3, self.sq_px // 20), tags=("markers",),
                )

        # 5. Arrows (Last Move, Hint, PV Reply)
        if active_move is not None:
            self._draw_arrow(active_move.from_sq, active_move.to_sq, color=t.move_arrow, dashed=False, width=4)

        if self.view_ply is None and self.hint_move is not None:
            self._draw_arrow(self.hint_move.from_sq, self.hint_move.to_sq, color=t.hint_arrow, dashed=False, width=5)
            if self.show_pv_arrow.get() and self.pv_reply_move is not None:
                self._draw_arrow(
                    self.pv_reply_move.from_sq,
                    self.pv_reply_move.to_sq,
                    color=t.pv_arrow,
                    dashed=True,
                    width=3,
                )

        # 6. Chess.com-style Move Quality Badge (! / ✓ / ?) on destination square
        if active_move is not None and active_grade is not None and hide_sq is None:
            self._draw_grade_badge_on_sq(active_move.to_sq, active_grade)

        # 7. Non-blocking dismissible Game-Over overlay (uses cached self.current_verdict!)
        if (
            self.current_verdict is not None
            and self.view_ply is None
            and not self.gameover_overlay_hidden
            and not self._anim_active
        ):
            self._draw_gameover_banner(self.current_verdict)

    def _draw_arrow(
        self,
        from_sq: int,
        to_sq: int,
        color: str,
        dashed: bool = False,
        width: int = 4,
    ) -> None:
        c = self.board_canvas
        x0, y0 = self._sq_center(from_sq)
        x1, y1 = self._sq_center(to_sq)
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return
        ux, uy = dx / dist, dy / dist
        start_off = self.sq_px * 0.16
        end_off = self.sq_px * 0.22
        sx, sy = x0 + ux * start_off, y0 + uy * start_off
        ex, ey = x1 - ux * end_off, y1 - uy * end_off

        arrowshape = (12, 16, 6) if width >= 4 else (10, 13, 5)
        dash_cfg = {"dash": (8, 5)} if dashed else {}

        c.create_line(
            sx, sy, ex, ey,
            fill="#151722",
            width=width + 3,
            capstyle=tk.ROUND,
            arrow=tk.LAST,
            arrowshape=arrowshape,
            tags=("arrows",),
            **dash_cfg,
        )
        c.create_line(
            sx, sy, ex, ey,
            fill=color,
            width=width,
            capstyle=tk.ROUND,
            arrow=tk.LAST,
            arrowshape=arrowshape,
            tags=("arrows",),
            **dash_cfg,
        )

    def _draw_gameover_banner(self, verdict: Tuple[int, str]) -> None:
        winner, reason = verdict
        c = self.board_canvas
        w_label = short_label(self.white_choice.get())
        b_label = short_label(self.black_choice.get())
        plies = len(self.move_uci_history)

        if winner == WHITE:
            title = f"🏆 WINNER: WHITE (+1) — {w_label}"
            sub = f"Defeated: Black ({b_label})  •  Checkmate in {plies} plies"
            border_col = ACCENT_GREEN
        elif winner == BLACK:
            title = f"🏆 WINNER: BLACK (-1) — {b_label}"
            sub = f"Defeated: White ({w_label})  •  Checkmate in {plies} plies"
            border_col = ACCENT_GREEN
        else:
            r_clean = reason.replace("_", " ").title()
            title = f"½–½ DRAW — {r_clean}"
            sub = f"White ({w_label}) vs. Black ({b_label})  •  {plies} plies"
            border_col = ACCENT_AMBER

        cx = self.board_px // 2
        bw = min(self.board_px - 36, 430)
        bh = 82
        x0 = cx - bw // 2
        x1 = cx + bw // 2
        y1 = self.board_px - self.FRAME_PAD - 14
        y0 = y1 - bh

        c.create_rectangle(x0 + 4, y0 + 4, x1 + 4, y1 + 4, fill="#0c0d14", outline="", tags=("overlay",))
        c.create_rectangle(x0, y0, x1, y1, fill=BG_CARD, outline=border_col, width=2, tags=("overlay",))
        c.create_text(cx - 18, y0 + 27, text=title, font=("Segoe UI", 10, "bold"), fill=border_col, tags=("overlay",))
        c.create_text(cx, y0 + 55, text=sub, font=("Segoe UI", 9), fill=FG_TEXT, tags=("overlay",))

        dx1, dy0 = x1 - 8, y0 + 8
        dx0, dy1 = dx1 - 54, dy0 + 22
        self._gameover_dismiss_rect = (dx0, dy0, dx1, dy1)
        c.create_rectangle(dx0, dy0, dx1, dy1, fill="#2e334a", outline=BORDER_SUBTLE, width=1, tags=("overlay",))
        c.create_text(
            (dx0 + dx1) // 2, (dy0 + dy1) // 2, text="✕ Hide", font=("Segoe UI", 8, "bold"), fill=FG_SUBTEXT, tags=("overlay",)
        )

    # ------------------------------------------------------------------
    # ZERO-REDRAW 60 FPS PIECE SLIDING ANIMATION
    # ------------------------------------------------------------------

    def _cancel_animation(self) -> None:
        if self._anim_after_id is not None:
            self.root.after_cancel(self._anim_after_id)
            self._anim_after_id = None
        if self._anim_item_id is not None:
            self.board_canvas.delete("anim_piece")
            self._anim_item_id = None
        self._anim_active = False
        self._anim_piece = EMPTY
        self._pending_after_anim = None

    def _start_move_animation(
        self,
        piece: int,
        from_sq: int,
        to_sq: int,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        self._cancel_animation()
        self._anim_active = True
        self._anim_piece = piece
        self._anim_from_sq = from_sq
        self._anim_to_sq = to_sq
        self._anim_start_xy = self._sq_center(from_sq)
        self._anim_end_xy = self._sq_center(to_sq)
        self._anim_start_time = time.perf_counter()
        self._pending_after_anim = on_done

        # Render board ONCE hiding the destination square piece, then move ONLY 1 canvas item!
        self._draw_board(hide_sq=to_sq)
        sx, sy = self._anim_start_xy
        self._anim_item_id = self._create_piece_item(sx, sy, piece, tags=("anim_piece",))
        self.board_canvas.tag_raise("anim_piece")
        self._anim_after_id = self.root.after(16, self._step_move_animation)

    def _step_move_animation(self) -> None:
        if not self._anim_active or self._anim_item_id is None:
            return
        elapsed = time.perf_counter() - self._anim_start_time
        t = min(1.0, elapsed / self._anim_duration)
        eased = 1.0 - (1.0 - t) ** 3

        sx, sy = self._anim_start_xy
        ex, ey = self._anim_end_xy
        ax = sx + (ex - sx) * eased
        ay = sy + (ey - sy) * eased

        # Ultra-fast <0.05 ms coordinate update (ZERO delete("all")!)
        self.board_canvas.coords(self._anim_item_id, ax, ay)

        if t < 1.0:
            self._anim_after_id = self.root.after(16, self._step_move_animation)
        else:
            callback = self._pending_after_anim
            self._anim_after_id = None
            self._anim_active = False
            self._pending_after_anim = None
            self.board_canvas.delete("anim_piece")
            self._anim_item_id = None
            self._anim_piece = EMPTY
            self._draw_board()
            if callback is not None:
                callback()

    # ------------------------------------------------------------------
    # ANIMATED EVAL BAR & CLICKABLE ADVANTAGE SPARKLINE
    # ------------------------------------------------------------------

    def _animate_eval_bar_to(self, target_cp: int) -> None:
        self._eval_target_cp = float(target_cp)
        if self._eval_anim_id is not None:
            self.root.after_cancel(self._eval_anim_id)
            self._eval_anim_id = None
        self._step_eval_bar_anim()

    def _step_eval_bar_anim(self) -> None:
        diff = self._eval_target_cp - self._eval_display_cp
        if abs(diff) <= 5.0:
            self._eval_display_cp = self._eval_target_cp
            self._eval_anim_id = None
            self._draw_eval_bar(self._eval_display_cp)
            return
        self._eval_display_cp += diff * 0.30
        self._draw_eval_bar(self._eval_display_cp)
        self._eval_anim_id = self.root.after(16, self._step_eval_bar_anim)

    def _draw_eval_bar(self, white_cp: float) -> None:
        ec = self.eval_canvas
        ec.delete("all")
        w = 34
        h = self.board_px

        clamped = max(-1500.0, min(1500.0, white_cp))
        white_frac = 1.0 / (1.0 + math.exp(-clamped / 350.0))
        white_h = int(round(h * white_frac))

        if not self.flipped:
            black_h = h - white_h
            ec.create_rectangle(0, 0, w, black_h, fill="#24273a", outline="")
            ec.create_rectangle(0, black_h, w, h, fill="#efe6d8", outline="")
        else:
            ec.create_rectangle(0, 0, w, white_h, fill="#efe6d8", outline="")
            ec.create_rectangle(0, white_h, w, h, fill="#24273a", outline="")

        ec.create_line(0, h // 2, w, h // 2, fill=ACCENT_RED, width=1)

        pawns = white_cp / 100.0
        txt = f"{pawns:+.1f}"
        if (white_cp >= 0 and not self.flipped) or (white_cp < 0 and self.flipped):
            ec.create_text(
                w // 2, h - 14, text=txt, font=("Consolas", 8, "bold"),
                fill="#181926" if not self.flipped else FG_TEXT,
            )
        else:
            ec.create_text(
                w // 2, 14, text=txt, font=("Consolas", 8, "bold"),
                fill=FG_TEXT if not self.flipped else "#181926",
            )

    def _draw_sparkline(self) -> None:
        sc = self.spark_canvas
        sc.delete("all")
        w = sc.winfo_width()
        h = sc.winfo_height()
        if w < 30 or h < 20:
            w, h = 348, 58

        pad_x = 10
        pad_y = 7
        mid_y = h / 2.0
        sc.create_line(pad_x, mid_y, w - pad_x, mid_y, fill="#383e5c", dash=(3, 3))

        vals = self.eval_history
        n = len(vals)
        if n <= 1:
            sc.create_text(w / 2, mid_y, text="Play moves to plot advantage curve", font=("Segoe UI", 8), fill="#656d8f")
            return

        max_cp = 1000.0
        span_x = max(1.0, (w - 2 * pad_x) / (n - 1))
        coords: List[Tuple[float, float]] = []
        for idx, cp in enumerate(vals):
            clamped = max(-max_cp, min(max_cp, float(cp)))
            x = pad_x + idx * span_x
            y = mid_y - (clamped / max_cp) * (mid_y - pad_y)
            coords.append((x, y))

        for idx in range(n - 1):
            x0, y0 = coords[idx]
            x1, y1 = coords[idx + 1]
            avg_y = (y0 + y1) / 2.0
            fill_c = "#2a4538" if avg_y <= mid_y else "#472c38"
            sc.create_polygon(x0, mid_y, x0, y0, x1, y1, x1, mid_y, fill=fill_c, outline="")

        flat_pts: List[float] = []
        for x, y in coords:
            flat_pts.extend([x, y])
        sc.create_line(*flat_pts, fill=ACCENT_BLUE, width=2, smooth=False)

        active_idx = len(vals) - 1 if self.view_ply is None else min(self.view_ply, len(vals) - 1)
        ax, ay = coords[active_idx]
        sc.create_oval(ax - 4, ay - 4, ax + 4, ay + 4, fill=ACCENT_AMBER, outline="#151722", width=1)

    def _on_sparkline_click(self, event: tk.Event) -> None:
        n = len(self.eval_history)
        if n <= 1:
            return
        w = max(40, self.spark_canvas.winfo_width())
        pad_x = 10
        frac = max(0.0, min(1.0, (event.x - pad_x) / max(1.0, w - 2 * pad_x)))
        ply_idx = int(round(frac * (n - 1)))
        self._jump_to_ply(ply_idx)

    # ------------------------------------------------------------------
    # MATERIAL BAR & TURN HELPERS
    # ------------------------------------------------------------------

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
        if self.current_verdict is not None:
            return False
        active_agent = self.white_agent if self.state.turn == WHITE else self.black_agent
        return active_agent is None

    # ------------------------------------------------------------------
    # ZERO-REDRAW MOUSE INTERACTION (CLICK-TO-MOVE + DRAG-AND-DROP)
    # ------------------------------------------------------------------

    def _complete_human_move(self, matching_moves: List[Move], animate: bool) -> None:
        if len(matching_moves) == 1:
            self._execute_move(matching_moves[0], animate=animate)
        else:
            promo_piece = self._prompt_promotion_choice()
            if promo_piece is not None:
                for m in matching_moves:
                    if m.promotion == promo_piece:
                        self._execute_move(m, animate=animate)
                        break
            else:
                self._draw_board()

    def _on_board_press(self, event: tk.Event) -> None:
        if self._gameover_dismiss_rect is not None:
            x0, y0, x1, y1 = self._gameover_dismiss_rect
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self.gameover_overlay_hidden = True
                self._draw_board()
                return

        if self.view_ply is not None:
            self.view_ply = None
            self._refresh_all()
            return
        if self.ai_thinking or self.hint_thinking or self._anim_active or not self._is_human_turn():
            return

        sq = self._canvas_xy_to_sq(event.x, event.y)
        if sq is None:
            return

        matching_moves = [m for m in self.legal_moves_for_selected if m.to_sq == sq]
        if matching_moves:
            self._complete_human_move(matching_moves, animate=True)
            return

        piece = self.state.board[sq]
        if piece != EMPTY and (piece > 0) == (self.state.turn > 0):
            self.selected_sq = sq
            self.hint_move = None
            self.pv_reply_move = None
            all_legal = self.state.generate_legal_moves()
            self.legal_moves_for_selected = [m for m in all_legal if m.from_sq == sq]

            self._drag_from_sq = sq
            self._drag_piece = piece
            self._drag_start_xy = (event.x, event.y)
            self._drag_active = False
            self._drag_hover_sq = sq
        else:
            self.selected_sq = None
            self.legal_moves_for_selected = []
            self._drag_from_sq = None

        self._draw_board()

    def _on_board_drag(self, event: tk.Event) -> None:
        if self._drag_from_sq is None:
            return
        c = self.board_canvas
        if not self._drag_active:
            sx, sy = self._drag_start_xy
            if math.hypot(event.x - sx, event.y - sy) >= 5:
                self._drag_active = True
                # Hide ONLY the origin piece item & spawn 1 lifted sprite (ZERO full redraw!)
                c.itemconfigure(f"piece_sq_{self._drag_from_sq}", state="hidden")
                self._drag_item_id = self._create_piece_item(
                    event.x, event.y, self._drag_piece, tags=("drag_piece",)
                )
                c.tag_raise("drag_piece")

        if self._drag_active and self._drag_item_id is not None:
            # Ultra-fast <0.05 ms coordinate update (silky smooth at any mouse polling rate!)
            c.coords(self._drag_item_id, event.x, event.y)

            hover_sq = self._canvas_xy_to_sq(event.x, event.y)
            if hover_sq != self._drag_hover_sq:
                self._drag_hover_sq = hover_sq
                if self._drag_hover_rect_id is not None:
                    c.delete(self._drag_hover_rect_id)
                    self._drag_hover_rect_id = None
                if hover_sq is not None and any(m.to_sq == hover_sq for m in self.legal_moves_for_selected):
                    x0, y0, x1, y1 = self._sq_to_canvas_xy(hover_sq)
                    self._drag_hover_rect_id = c.create_rectangle(
                        x0, y0, x1, y1, fill=self.theme.hover_sq, outline="", tags=("drag_hover",)
                    )
                    # Keep hover highlight neatly under pieces & markers
                    c.tag_lower(self._drag_hover_rect_id, "pieces")

    def _on_board_release(self, event: tk.Event) -> None:
        if self._drag_from_sq is None:
            return
        c = self.board_canvas
        was_dragging = self._drag_active
        from_sq = self._drag_from_sq
        self._drag_from_sq = None
        self._drag_active = False
        self._drag_hover_sq = None
        self._drag_piece = EMPTY

        if self._drag_item_id is not None:
            c.delete("drag_piece")
            self._drag_item_id = None
        if self._drag_hover_rect_id is not None:
            c.delete(self._drag_hover_rect_id)
            self._drag_hover_rect_id = None

        if was_dragging:
            drop_sq = self._canvas_xy_to_sq(event.x, event.y)
            if drop_sq is not None and drop_sq != from_sq:
                matching = [m for m in self.legal_moves_for_selected if m.to_sq == drop_sq]
                if matching:
                    self._complete_human_move(matching, animate=False)
                    return
            # Instant 0-redraw snap-back
            c.itemconfigure(f"piece_sq_{from_sq}", state="normal")

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

            self._make_button(btn_frame, label, _pick, bg=ACCENT_BLUE, fg="#151722").pack(
                side=tk.LEFT, padx=4
            )

        self.root.wait_window(dialog)
        return choice_var.get()

    # ------------------------------------------------------------------
    # MOVE EXECUTION, ASYNC MOVE GRADER & ZERO-CONTENTION AI WORKER
    # ------------------------------------------------------------------

    def _execute_move(self, move: Move, stats: Optional[SearchStats] = None, animate: bool = True) -> None:
        self._cancel_animation()
        state_before = self.state.copy()
        mover = self.state.turn
        mover_label = short_label(self.white_choice.get() if mover == WHITE else self.black_choice.get())
        moved_piece_signed = self.state.board[move.from_sq]
        moved_piece = abs(moved_piece_signed)
        captured_piece = abs(self.state.board[move.to_sq])

        self.state.make_move(move)
        self.current_verdict = adjudicate_state(self.state)
        self.view_ply = None
        self.gameover_overlay_hidden = False
        self.hint_move = None
        self.pv_reply_move = None

        ply = len(self.move_uci_history) + 1
        self.move_uci_history.append(move.uci())
        self.last_move = move
        self.selected_sq = None
        self.legal_moves_for_selected = []

        w_eval = evaluate_tuned_white(self.state)
        self.eval_history.append(w_eval)

        side_str = "White" if mover == WHITE else "Black"
        p_glyph = PIECE_GLYPHS.get(moved_piece, "")
        p_name = PIECE_NAMES.get(moved_piece, "Piece")
        src, dst = sq_to_str(move.from_sq), sq_to_str(move.to_sq)

        if captured_piece != EMPTY:
            cap_name = PIECE_NAMES.get(captured_piece, "piece")
            desc = f"{src} × {dst} (captured {cap_name})"
        else:
            desc = f"{src} → {dst}"
        if move.promotion is not None and move.promotion != 0:
            desc += f" [={PIECE_NAMES.get(move.promotion, 'Queen')}]"
        in_chk = self.state.in_check(self.state.turn)
        if in_chk:
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
                in_check=in_chk,
                eval_cp=w_eval,
                stats=stats,
                grade=None,
            )
        )

        # Play crisp procedural audio cue
        if self.current_verdict is not None:
            self.sounds.play("gameover")
        elif in_chk:
            self.sounds.play("check")
        elif move.promotion is not None and move.promotion != 0:
            self.sounds.play("promote")
        elif captured_piece != EMPTY:
            self.sounds.play("capture")
        else:
            self.sounds.play("move")

        if stats is not None:
            self._update_telemetry_from_stats(stats)

        current_game_id = self.game_id

        def _after_move_visuals() -> None:
            # Grade move first (~45ms off-thread) so badge pops in cleanly before next AI search uses CPU
            self._launch_move_grader(
                current_game_id,
                ply,
                state_before,
                move,
                on_done=self._schedule_ai_if_needed,
            )

        self._refresh_all(redraw_board=not animate)
        if animate:
            self._start_move_animation(moved_piece_signed, move.from_sq, move.to_sq, on_done=_after_move_visuals)
        else:
            _after_move_visuals()

    def _launch_move_grader(
        self,
        worker_game_id: int,
        ply: int,
        state_before: GameState,
        played_move: Move,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        def _worker():
            grade_res: Optional[MoveGrade] = None
            try:
                grade_res = grade_move(state_before, played_move, search_depth=3)
            except Exception:
                grade_res = None
            finally:
                self._ui_queue.put(
                    lambda: self._on_grade_done(worker_game_id, ply, grade_res, on_done)
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_grade_done(
        self,
        worker_game_id: int,
        ply: int,
        grade: Optional[MoveGrade],
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        if worker_game_id != self.game_id:
            return
        idx = ply - 1
        if grade is not None and 0 <= idx < len(self.snapshots):
            self.snapshots[idx].grade = grade
            self._update_grade_counters()
            self._render_clickable_history()

            active_ply = len(self.snapshots) if self.view_ply is None else self.view_ply
            if active_ply == ply:
                self._update_grade_pill(grade)
                if not self._anim_active and not self._drag_active:
                    self.board_canvas.delete("badge")
                    self._draw_grade_badge_on_sq(self.snapshots[idx].move.to_sq, grade)

        if on_done is not None:
            on_done()

    def _schedule_ai_if_needed(self) -> None:
        if self.ai_thinking or self.hint_thinking or self._anim_active:
            return
        if self.current_verdict is not None:
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
            text=f"⏳ {side_str} ({active_agent.name}) is thinking...",
            fg=ACCENT_BLUE,
        )

        current_game_id = self.game_id
        state_copy = self.state.copy()
        both_ai = (self.white_agent is not None) and (self.black_agent is not None)
        min_delay = float(self.ai_delay_var.get()) if both_ai else 0.16

        def _worker():
            chosen_move: Optional[Move] = None
            stats: Optional[SearchStats] = None
            try:
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
            finally:
                self._ui_queue.put(lambda: self._on_ai_done(current_game_id, chosen_move, stats))

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
        if chosen_move is not None and self.current_verdict is None:
            self._execute_move(chosen_move, stats=stats, animate=True)

    def _on_hint(self) -> None:
        if self.ai_thinking or self.hint_thinking or self._anim_active or self.current_verdict is not None:
            return
        self.view_ply = None
        self.hint_thinking = True
        self.status_label.configure(text="⏳ Analyzing best move (d=4+Q)...", fg=ACCENT_BLUE)
        self.hint_btn.configure(state=tk.DISABLED)

        current_game_id = self.game_id
        expected_ply = len(self.move_uci_history)
        state_copy = self.state.copy()

        def _worker():
            best_move: Optional[Move] = None
            st: Optional[SearchStats] = None
            try:
                best_move = self.hint_agent.select_move(state_copy)
                st = self.hint_agent.last_stats
            finally:
                self._ui_queue.put(
                    lambda: self._on_hint_done(current_game_id, expected_ply, best_move, st)
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_hint_done(
        self,
        worker_game_id: int,
        expected_ply: int,
        best_move: Optional[Move],
        st: Optional[SearchStats],
    ) -> None:
        self.hint_thinking = False
        self.hint_btn.configure(state=tk.NORMAL)
        if worker_game_id != self.game_id or len(self.move_uci_history) != expected_ply:
            return
        if st is not None:
            self._update_telemetry_from_stats(st)
        if best_move is not None:
            self.hint_move = best_move
            self.pv_reply_move = None
            if st is not None and len(st.pv) >= 2:
                try:
                    self.pv_reply_move = Move.from_uci(st.pv[1])
                except Exception:
                    self.pv_reply_move = None

            self.selected_sq = best_move.from_sq
            self.legal_moves_for_selected = [best_move]
            score_txt = f"{st.score:+d} cp" if st is not None else "—"
            if st is not None and abs(st.score) > MATE_SCORE - 1000:
                plies_m = MATE_SCORE - abs(st.score)
                score_txt = f"{'+' if st.score > 0 else '-'}Mate in {plies_m}"
            self.status_label.configure(
                text=f"💡 Hint: Play {best_move.uci()}  (Eval: {score_txt})",
                fg=ACCENT_GREEN,
            )
            self._draw_board()

    # ------------------------------------------------------------------
    # TELEMETRY, MOVE GRADE COUNTERS, HISTORY & REFRESH
    # ------------------------------------------------------------------

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

    def _update_grade_counters(self) -> None:
        great_n = sum(1 for s in self.snapshots if s.grade and s.grade.label == "Great")
        good_n = sum(1 for s in self.snapshots if s.grade and s.grade.label == "Good")
        mistake_n = sum(1 for s in self.snapshots if s.grade and s.grade.label == "Mistake")
        self.great_cnt_lbl.configure(text=f"! Great: {great_n}")
        self.good_cnt_lbl.configure(text=f"✓ Good: {good_n}")
        self.mistake_cnt_lbl.configure(text=f"? Mistake: {mistake_n}")

    def _update_grade_pill(self, grade: Optional[MoveGrade]) -> None:
        if grade is None:
            self.grade_pill_label.configure(text="", bg=BG_CARD, fg=FG_TEXT)
            return
        color_hex, _sym, tag = GRADE_COLORS.get(grade.label, GRADE_COLORS["Good"])
        if grade.label == "Mistake" and grade.best_move_uci:
            txt = f" {tag}  (Best: {grade.best_move_uci}) "
        else:
            txt = f" {tag} "
        self.grade_pill_label.configure(text=txt, bg=color_hex, fg="#ffffff")

    def _refresh_all(self, redraw_board: bool = True) -> None:
        active_idx = len(self.eval_history) - 1 if self.view_ply is None else min(self.view_ply, len(self.eval_history) - 1)
        display_eval = self.eval_history[active_idx]
        self._animate_eval_bar_to(display_eval)
        self.telemetry_vars["eval_str"].set(f"{display_eval:+d} cp ({display_eval / 100.0:+.2f})")
        self.telemetry_vars["clock_str"].set(
            f"{self.state.halfmove_clock} / 100  (Ply {len(self.state.history)})"
        )

        board_to_draw, _, _, _, active_grade = self._get_active_snapshot_data()
        self._update_material_bar(board_to_draw)
        self._update_grade_counters()
        self._update_grade_pill(active_grade)
        if redraw_board:
            self._draw_board()
        self._draw_sparkline()

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
            self.step_label.configure(
                text="Step 0: Starting Position — Click or drag any piece to play", fg=ACCENT_AMBER
            )

        # Update Top Status Banner
        verdict = self.current_verdict
        if verdict is not None:
            winner, reason = verdict
            w_name = short_label(self.white_choice.get())
            b_name = short_label(self.black_choice.get())
            if winner == WHITE:
                msg = f"🏆 WHITE ({w_name}) WINS!  |  BLACK ({b_name}) LOSES  [{reason.title()}]"
                color = ACCENT_GREEN
            elif winner == BLACK:
                msg = f"🏆 BLACK ({b_name}) WINS!  |  WHITE ({w_name}) LOSES  [{reason.title()}]"
                color = ACCENT_GREEN
            else:
                msg = f"½–½ DRAW ({w_name} vs {b_name}) — {reason.replace('_', ' ').title()}"
                color = ACCENT_AMBER
            self.status_label.configure(text=msg, fg=color)
        else:
            side = "White (+1)" if self.state.turn == WHITE else "Black (-1)"
            if self.ai_paused and not self._is_human_turn():
                self.status_label.configure(
                    text=f"⏸ AI Paused — {side} to move (Step AI or Press Space)",
                    fg=ACCENT_AMBER,
                )
            elif self.state.in_check():
                self.status_label.configure(text=f"⚠ CHECK! {side} to move", fg=ACCENT_RED)
            else:
                self.status_label.configure(text=f"{side} to move", fg=FG_TEXT)

        self._render_clickable_history()

    def _render_clickable_history(self) -> None:
        ht = self.history_text
        ht.configure(state=tk.NORMAL)
        ht.delete("1.0", tk.END)

        active_ply = len(self.snapshots) if self.view_ply is None else self.view_ply
        total = len(self.move_uci_history)

        for idx in range(0, total, 2):
            move_num = (idx // 2) + 1
            ht.insert(tk.END, f"{move_num:>2}. ", ("move_num",))

            for offset in (0, 1):
                ply_idx = idx + offset + 1
                if ply_idx <= total:
                    uci_str = self.move_uci_history[ply_idx - 1]
                    snap = self.snapshots[ply_idx - 1]
                    tag_name = f"ply_{ply_idx}"
                    tags = [tag_name]
                    if ply_idx == active_ply:
                        tags.append("active_move")
                    ht.insert(tk.END, f"{uci_str:<6}", tuple(tags))
                    if snap.grade is not None:
                        _, sym, _ = GRADE_COLORS.get(snap.grade.label, GRADE_COLORS["Good"])
                        ht.insert(tk.END, f"{sym} ", (tag_name, f"grade_{snap.grade.label}"))
                    else:
                        ht.insert(tk.END, "  ", (tag_name,))

                    ht.tag_bind(tag_name, "<Button-1>", lambda _e, p=ply_idx: self._jump_to_ply(p))
                    ht.tag_bind(tag_name, "<Enter>", lambda _e: ht.configure(cursor="hand2"))
                    ht.tag_bind(tag_name, "<Leave>", lambda _e: ht.configure(cursor="arrow"))
                    if offset == 0:
                        ht.insert(tk.END, "  ")
            ht.insert(tk.END, "\n")

        ht.see(tk.END)
        ht.configure(state=tk.DISABLED)

    def _jump_to_ply(self, ply_idx: int) -> None:
        if not self.snapshots:
            return
        if ply_idx >= len(self.snapshots):
            self.view_ply = None
            last_st = self.snapshots[-1].stats
            if last_st is not None:
                self._update_telemetry_from_stats(last_st)
        else:
            self.view_ply = max(0, ply_idx)
            if self.view_ply > 0 and self.snapshots[self.view_ply - 1].stats is not None:
                self._update_telemetry_from_stats(self.snapshots[self.view_ply - 1].stats)
        self._cancel_animation()
        self._refresh_all()

    # ------------------------------------------------------------------
    # CONTROLS & CALLBACKS
    # ------------------------------------------------------------------

    def _on_new_game(self) -> None:
        self._cancel_animation()
        self.game_id += 1
        self.ai_thinking = False
        self.hint_thinking = False
        self.hint_btn.configure(state=tk.NORMAL)
        self.step_once = False
        self.gameover_overlay_hidden = False
        self.state = GameState()
        self.current_verdict = None
        self.initial_board = list(self.state.board)
        self.move_uci_history.clear()
        self.snapshots.clear()
        self.eval_history = [0]
        self.view_ply = None
        self.last_move = None
        self.selected_sq = None
        self.legal_moves_for_selected.clear()
        self.hint_move = None
        self.pv_reply_move = None

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
        self._cancel_animation()
        self.game_id += 1
        self.ai_thinking = False
        self.hint_thinking = False
        self.hint_btn.configure(state=tk.NORMAL)
        self.view_ply = None
        self.gameover_overlay_hidden = False
        self.hint_move = None
        self.pv_reply_move = None

        if self.white_agent is not None and self.black_agent is not None:
            self.ai_paused = True
            self._set_button_style(self.pause_btn, "Resume AI", ACCENT_GREEN, "#151722")

        human_vs_ai = (self.white_agent is None) != (self.black_agent is None)
        plies_to_undo = 2 if (human_vs_ai and len(self.move_uci_history) >= 2) else 1

        for _ in range(plies_to_undo):
            if self.move_uci_history:
                self.state.unmake_move()
                self.move_uci_history.pop()
                if self.snapshots:
                    self.snapshots.pop()
                if len(self.eval_history) > 1:
                    self.eval_history.pop()

        self.current_verdict = adjudicate_state(self.state)
        self.last_move = self.state._undo_stack[-1][0] if self.state._undo_stack else None
        self.selected_sq = None
        self.legal_moves_for_selected.clear()
        self._refresh_all()

    def _on_flip(self) -> None:
        self._cancel_animation()
        self.flipped = not self.flipped
        self._refresh_all()

    def _on_theme_changed(self, _event: Optional[tk.Event] = None) -> None:
        chosen = self.theme_name.get()
        if chosen in BOARD_THEMES:
            self.theme = BOARD_THEMES[chosen]
            self.sprites.get_board_bg(self.theme, self.sq_px, self.FRAME_PAD)
            self._draw_board()

    def _on_delay_changed(self, val: str) -> None:
        v = round(float(val), 1)
        self.ai_delay_var.set(v)
        self.delay_val_label.configure(text=f"{v:.1f}s")

    def _on_toggle_sound(self) -> None:
        self.sound_enabled = not self.sound_enabled
        self.sounds.enabled = self.sound_enabled
        if self.sound_enabled:
            self._set_button_style(self.sound_btn, "🔊 Sound", "#2e334a", FG_TEXT)
            self.sounds.play("move")
        else:
            self._set_button_style(self.sound_btn, "🔇 Muted", BG_CARD, FG_SUBTEXT)

    def _on_prev_step(self) -> None:
        if not self.snapshots:
            return
        target = max(0, len(self.snapshots) - 1) if self.view_ply is None else max(0, self.view_ply - 1)
        self._jump_to_ply(target)

    def _on_next_step(self) -> None:
        if self.view_ply is None:
            return
        self._jump_to_ply(self.view_ply + 1)

    def _on_step_ai(self) -> None:
        """Advance AI vs AI by one single move and remain paused for step-by-step inspection."""
        if self.view_ply is not None:
            self._on_next_step()
            return
        self.ai_paused = True
        self._set_button_style(self.pause_btn, "Resume AI", ACCENT_GREEN, "#151722")
        self.step_once = True
        self._schedule_ai_if_needed()

    def _on_toggle_pause(self) -> None:
        self.ai_paused = not self.ai_paused
        if self.ai_paused:
            self._set_button_style(self.pause_btn, "Resume AI", ACCENT_GREEN, "#151722")
            self._refresh_all()
        else:
            self.view_ply = None
            self._set_button_style(self.pause_btn, "Pause AI", ACCENT_AMBER, "#151722")
            self._refresh_all()
            self._schedule_ai_if_needed()

    def _on_player_changed(self, _event: Optional[tk.Event] = None) -> None:
        self._cancel_animation()
        self.game_id += 1
        self.ai_thinking = False
        self.hint_thinking = False
        self.hint_btn.configure(state=tk.NORMAL)
        self.white_agent = create_agent_from_label(self.white_choice.get())
        self.black_agent = create_agent_from_label(self.black_choice.get())
        self._refresh_all()
        self._schedule_ai_if_needed()


def main() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    MiniChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
