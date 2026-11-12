"""
Phase 1: Board Representation, Move Generation, and Make/Unmake Move
Variant: Gardner's Minichess (5x5)
Single source of truth: rules.txt
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. CONSTANTS & PIECE ENCODING (rules.txt Section 1)
# ---------------------------------------------------------------------------

BOARD_SIZE = 25
FILES = "abcde"
RANKS = "12345"

WHITE = 1
BLACK = -1

EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)

PIECE_TO_CHAR = {
    EMPTY: ".",
    PAWN: "P",
    KNIGHT: "N",
    BISHOP: "B",
    ROOK: "R",
    QUEEN: "Q",
    KING: "K",
    -PAWN: "p",
    -KNIGHT: "n",
    -BISHOP: "b",
    -ROOK: "r",
    -QUEEN: "q",
    -KING: "k",
}

CHAR_TO_PROMO = {
    "q": QUEEN,
    "r": ROOK,
    "b": BISHOP,
    "n": KNIGHT,
}

PROMO_TO_CHAR = {
    QUEEN: "q",
    ROOK: "r",
    BISHOP: "b",
    KNIGHT: "n",
}

INITIAL_BOARD: List[int] = [
    ROOK, KNIGHT, BISHOP, QUEEN, KING,          # Rank 1 (a1..e1)
    PAWN, PAWN, PAWN, PAWN, PAWN,               # Rank 2 (a2..e2)
    EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,          # Rank 3 (a3..e3)
    -PAWN, -PAWN, -PAWN, -PAWN, -PAWN,          # Rank 4 (a4..e4)
    -ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING,     # Rank 5 (a5..e5)
]


# ---------------------------------------------------------------------------
# 2. COORDINATE HELPERS & PRECOMPUTED ATTACK / RAY TABLES
# ---------------------------------------------------------------------------

def sq_to_coords(sq: int) -> Tuple[int, int]:
    """Return (rank_idx 0..4, file_idx 0..4) for square 0..24."""
    return divmod(sq, 5)


def coords_to_sq(rank_idx: int, file_idx: int) -> int:
    """Return square index 0..24 from rank_idx (0..4) and file_idx (0..4)."""
    return rank_idx * 5 + file_idx


def sq_to_str(sq: int) -> str:
    """Convert square index (0..24) to algebraic string ('a1'..'e5')."""
    r, f = divmod(sq, 5)
    return f"{FILES[f]}{RANKS[r]}"


def str_to_sq(s: str) -> int:
    """Convert algebraic string ('a1'..'e5') to square index (0..24)."""
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or s[1] not in RANKS:
        raise ValueError(f"Invalid square notation: {s!r}")
    return (int(s[1]) - 1) * 5 + FILES.index(s[0])


KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_DELTAS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALL_DIRS = ORTHO_DIRS + DIAG_DIRS

KNIGHT_MOVES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(
        coords_to_sq(r + dr, f + df)
        for dr, df in KNIGHT_DELTAS
        if 0 <= r + dr < 5 and 0 <= f + df < 5
    )
    for r in range(5)
    for f in range(5)
)

KING_MOVES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(
        coords_to_sq(r + dr, f + df)
        for dr, df in KING_DELTAS
        if 0 <= r + dr < 5 and 0 <= f + df < 5
    )
    for r in range(5)
    for f in range(5)
)


def _build_rays(directions: Tuple[Tuple[int, int], ...]) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    rays_per_sq = []
    for sq in range(BOARD_SIZE):
        r, f = sq_to_coords(sq)
        sq_rays = []
        for dr, df in directions:
            ray = []
            nr, nf = r + dr, f + df
            while 0 <= nr < 5 and 0 <= nf < 5:
                ray.append(coords_to_sq(nr, nf))
                nr += dr
                nf += df
            if ray:
                sq_rays.append(tuple(ray))
        rays_per_sq.append(tuple(sq_rays))
    return tuple(rays_per_sq)


ORTHO_RAYS = _build_rays(ORTHO_DIRS)
DIAG_RAYS = _build_rays(DIAG_DIRS)
QUEEN_RAYS = _build_rays(ALL_DIRS)

WHITE_PAWN_CAPTURES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(
        coords_to_sq(r + 1, f + df)
        for df in (-1, 1)
        if 0 <= r + 1 < 5 and 0 <= f + df < 5
    )
    for r in range(5)
    for f in range(5)
)

BLACK_PAWN_CAPTURES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(
        coords_to_sq(r - 1, f + df)
        for df in (-1, 1)
        if 0 <= r - 1 < 5 and 0 <= f + df < 5
    )
    for r in range(5)
    for f in range(5)
)


# ---------------------------------------------------------------------------
# 3. ZOBRIST HASHING (rules.txt Section 6.2)
# ---------------------------------------------------------------------------
_rng = random.Random(0x5F3759DF)

ZOBRIST_PIECE_SQ: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(_rng.getrandbits(64) if p != 0 else 0 for _ in range(BOARD_SIZE))
    for p in range(-6, 7)
)
ZOBRIST_BLACK_TO_MOVE: int = _rng.getrandbits(64)


# ---------------------------------------------------------------------------
# 4. MOVE REPRESENTATION
# ---------------------------------------------------------------------------

class Move:
    """
    Represents a move on the 5x5 board.
    - from_sq: int (0..24)
    - to_sq: int (0..24)
    - promotion: int (0 if none, else unsigned piece type 2..5: KNIGHT, BISHOP, ROOK, QUEEN)
    """
    __slots__ = ("from_sq", "to_sq", "promotion")

    def __init__(self, from_sq: int, to_sq: int, promotion: int = 0) -> None:
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.promotion = abs(promotion)

    def uci(self) -> str:
        base = f"{sq_to_str(self.from_sq)}{sq_to_str(self.to_sq)}"
        if self.promotion:
            return base + PROMO_TO_CHAR[self.promotion]
        return base

    @classmethod
    def from_uci(cls, uci_str: str) -> "Move":
        s = uci_str.strip().lower()
        if len(s) not in (4, 5):
            raise ValueError(f"Invalid UCI move format: {uci_str!r} (expected e.g. 'a2a3' or 'a4a5q')")
        from_sq = str_to_sq(s[0:2])
        to_sq = str_to_sq(s[2:4])
        promo = 0
        if len(s) == 5:
            if s[4] not in CHAR_TO_PROMO:
                raise ValueError(f"Invalid promotion piece {s[4]!r} in {uci_str!r} (choose q, r, b, n)")
            promo = CHAR_TO_PROMO[s[4]]
        return cls(from_sq, to_sq, promo)

    def __iter__(self):
        yield self.from_sq
        yield self.to_sq
        yield self.promotion

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Move):
            return (
                self.from_sq == other.from_sq
                and self.to_sq == other.to_sq
                and self.promotion == other.promotion
            )
        return False

    def __hash__(self) -> int:
        return (self.from_sq << 8) | (self.to_sq << 3) | self.promotion

    def __repr__(self) -> str:
        return f"Move({self.uci()!r})"

    def __str__(self) -> str:
        return self.uci()


# ---------------------------------------------------------------------------
# 5. GAME STATE CLASS (rules.txt Section 1 & Section 6)
# ---------------------------------------------------------------------------

class GameState:
    """
    Maintains the complete state of a 5x5 Gardner Minichess game:
    - board: list of length 25 (signed ints -6..+6)
    - turn: +1 (White) or -1 (Black)
    - halfmove_clock: int (plies since last pawn move or capture)
    - history: list of past Zobrist hashes (for 3-fold repetition detection)
    - zobrist_hash: current 64-bit Zobrist hash of (board, turn)
    """

    def __init__(
        self,
        board: Optional[List[int]] = None,
        turn: int = WHITE,
        halfmove_clock: int = 0,
        history: Optional[List[int]] = None,
        debug: bool = False,
    ) -> None:
        self.board: List[int] = list(board) if board is not None else list(INITIAL_BOARD)
        if len(self.board) != BOARD_SIZE:
            raise ValueError(f"Board must have length {BOARD_SIZE}, got {len(self.board)}")
        self.turn: int = turn
        self.halfmove_clock: int = halfmove_clock
        self.debug: bool = debug

        self.zobrist_hash: int = self.compute_zobrist_hash()
        self.history: List[int] = list(history) if history is not None else []
        self._undo_stack: List[Tuple[Move, int, int, int, int]] = []

    def copy(self) -> "GameState":
        clone = GameState(
            board=self.board,
            turn=self.turn,
            halfmove_clock=self.halfmove_clock,
            history=self.history,
            debug=self.debug,
        )
        clone.zobrist_hash = self.zobrist_hash
        clone._undo_stack = list(self._undo_stack)
        return clone

    def compute_zobrist_hash(self) -> int:
        """Full recomputation of the 64-bit Zobrist hash (rules.txt Section 6.2)."""
        h = 0
        for sq, piece in enumerate(self.board):
            if piece != EMPTY:
                h ^= ZOBRIST_PIECE_SQ[piece + 6][sq]
        if self.turn == BLACK:
            h ^= ZOBRIST_BLACK_TO_MOVE
        return h

    def find_king(self, color: int) -> int:
        """Return the square index (0..24) of `color`'s king, or -1 if not found."""
        target = KING * color
        board = self.board
        for sq in range(BOARD_SIZE):
            if board[sq] == target:
                return sq
        return -1

    def is_square_attacked(self, sq: int, by_color: int) -> bool:
        """Return True if square `sq` is attacked by any piece of `by_color`."""
        board = self.board

        enemy_pawn = PAWN * by_color
        pawn_lookup = BLACK_PAWN_CAPTURES[sq] if by_color == WHITE else WHITE_PAWN_CAPTURES[sq]
        for att_sq in pawn_lookup:
            if board[att_sq] == enemy_pawn:
                return True

        enemy_knight = KNIGHT * by_color
        for att_sq in KNIGHT_MOVES[sq]:
            if board[att_sq] == enemy_knight:
                return True

        enemy_king = KING * by_color
        for att_sq in KING_MOVES[sq]:
            if board[att_sq] == enemy_king:
                return True

        enemy_rook = ROOK * by_color
        enemy_queen = QUEEN * by_color
        for ray in ORTHO_RAYS[sq]:
            for att_sq in ray:
                p = board[att_sq]
                if p != EMPTY:
                    if p == enemy_rook or p == enemy_queen:
                        return True
                    break

        enemy_bishop = BISHOP * by_color
        for ray in DIAG_RAYS[sq]:
            for att_sq in ray:
                p = board[att_sq]
                if p != EMPTY:
                    if p == enemy_bishop or p == enemy_queen:
                        return True
                    break

        return False

    def in_check(self, color: Optional[int] = None) -> bool:
        """Return True if `color`'s king is currently under attack (rules.txt Section 3.2)."""
        if color is None:
            color = self.turn
        king_sq = self.find_king(color)
        if king_sq == -1:
            return False
        return self.is_square_attacked(king_sq, -color)

    def make_move(self, move: Move) -> None:
        """Update board state in-place for `move`."""
        board = self.board
        from_sq = move.from_sq
        to_sq = move.to_sq
        promo = move.promotion

        moving_piece = board[from_sq]
        captured_piece = board[to_sq]
        prev_clock = self.halfmove_clock
        prev_hash = self.zobrist_hash

        self.history.append(prev_hash)
        self._undo_stack.append((move, moving_piece, captured_piece, prev_clock, prev_hash))

        h = prev_hash ^ ZOBRIST_PIECE_SQ[moving_piece + 6][from_sq]
        if captured_piece != EMPTY:
            h ^= ZOBRIST_PIECE_SQ[captured_piece + 6][to_sq]

        placed_piece = (promo * self.turn) if promo else moving_piece
        h ^= ZOBRIST_PIECE_SQ[placed_piece + 6][to_sq]
        h ^= ZOBRIST_BLACK_TO_MOVE

        board[from_sq] = EMPTY
        board[to_sq] = placed_piece

        if abs(moving_piece) == PAWN or captured_piece != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock = prev_clock + 1

        self.turn = -self.turn
        self.zobrist_hash = h

        if self.debug:
            assert self.zobrist_hash == self.compute_zobrist_hash(), (
                "Zobrist hash mismatch after make_move!"
            )

    def unmake_move(self, move: Optional[Move] = None) -> Move:
        """Rollback the last move in-place and restore exact previous state."""
        if not self._undo_stack:
            raise RuntimeError("Cannot unmake_move: undo stack is empty.")

        last_move, moving_piece, captured_piece, prev_clock, prev_hash = self._undo_stack.pop()
        self.history.pop()

        if self.debug and move is not None and move != last_move:
            raise AssertionError(f"unmake_move called with {move}, expected {last_move}")

        board = self.board
        board[last_move.from_sq] = moving_piece
        board[last_move.to_sq] = captured_piece

        self.turn = -self.turn
        self.halfmove_clock = prev_clock
        self.zobrist_hash = prev_hash

        if self.debug:
            assert self.zobrist_hash == self.compute_zobrist_hash(), (
                "Zobrist hash mismatch after unmake_move!"
            )

        return last_move

    def generate_pseudo_legal_moves(self) -> List[Move]:
        """Generate all pseudo-legal moves for `self.turn`."""
        moves: List[Move] = []
        board = self.board
        turn = self.turn

        for sq in range(BOARD_SIZE):
            piece = board[sq]
            if piece == EMPTY or (piece > 0) != (turn > 0):
                continue

            piece_type = abs(piece)

            if piece_type == PAWN:
                if turn == WHITE:
                    to_sq = sq + 5
                    if to_sq < BOARD_SIZE and board[to_sq] == EMPTY:
                        if to_sq >= 20:
                            for promo in PROMOTION_PIECES:
                                moves.append(Move(sq, to_sq, promo))
                        else:
                            moves.append(Move(sq, to_sq))

                    for cap_sq in WHITE_PAWN_CAPTURES[sq]:
                        target = board[cap_sq]
                        if target < 0 and target != -KING:
                            if cap_sq >= 20:
                                for promo in PROMOTION_PIECES:
                                    moves.append(Move(sq, cap_sq, promo))
                            else:
                                moves.append(Move(sq, cap_sq))
                else:
                    to_sq = sq - 5
                    if to_sq >= 0 and board[to_sq] == EMPTY:
                        if to_sq < 5:
                            for promo in PROMOTION_PIECES:
                                moves.append(Move(sq, to_sq, promo))
                        else:
                            moves.append(Move(sq, to_sq))

                    for cap_sq in BLACK_PAWN_CAPTURES[sq]:
                        target = board[cap_sq]
                        if target > 0 and target != KING:
                            if cap_sq < 5:
                                for promo in PROMOTION_PIECES:
                                    moves.append(Move(sq, cap_sq, promo))
                            else:
                                moves.append(Move(sq, cap_sq))

            elif piece_type == KNIGHT:
                for to_sq in KNIGHT_MOVES[sq]:
                    target = board[to_sq]
                    if target == EMPTY or ((target > 0) != (turn > 0) and abs(target) != KING):
                        moves.append(Move(sq, to_sq))

            elif piece_type == KING:
                for to_sq in KING_MOVES[sq]:
                    target = board[to_sq]
                    if target == EMPTY or ((target > 0) != (turn > 0) and abs(target) != KING):
                        moves.append(Move(sq, to_sq))

            elif piece_type in (BISHOP, ROOK, QUEEN):
                rays = (
                    DIAG_RAYS[sq]
                    if piece_type == BISHOP
                    else ORTHO_RAYS[sq]
                    if piece_type == ROOK
                    else QUEEN_RAYS[sq]
                )
                for ray in rays:
                    for to_sq in ray:
                        target = board[to_sq]
                        if target == EMPTY:
                            moves.append(Move(sq, to_sq))
                        else:
                            if (target > 0) != (turn > 0) and abs(target) != KING:
                                moves.append(Move(sq, to_sq))
                            break

        return moves

    def generate_legal_moves(self) -> List[Move]:
        """Generate all strictly legal moves for `self.turn`."""
        legal_moves: List[Move] = []
        mover = self.turn
        for move in self.generate_pseudo_legal_moves():
            self.make_move(move)
            if not self.in_check(mover):
                legal_moves.append(move)
            self.unmake_move(move)
        return legal_moves

    def is_terminal(self) -> bool:
        """Return True if the current side to move has no legal moves (checkmate or stalemate)."""
        return len(self.generate_legal_moves()) == 0

    def is_threefold_repetition(self) -> bool:
        current = self.zobrist_hash
        count = 1 + sum(1 for h in self.history if h == current)
        return count >= 3

    def is_fifty_move_draw(self) -> bool:
        return self.halfmove_clock >= 100

    def is_dead_position(self) -> bool:
        non_kings = [abs(p) for p in self.board if p != EMPTY and abs(p) != KING]
        if not non_kings:
            return True
        if len(non_kings) == 1 and non_kings[0] in (KNIGHT, BISHOP):
            return True
        return False

    def is_hard_cap_draw(self) -> bool:
        return len(self.history) >= 300

    def render_ascii(self) -> str:
        lines = []
        for r in range(4, -1, -1):
            row_chars = [PIECE_TO_CHAR[self.board[r * 5 + f]] for f in range(5)]
            lines.append(f"  {r + 1} | {'  '.join(row_chars)}")
        lines.append("      a  b  c  d  e")
        turn_str = "White (+1)" if self.turn == WHITE else "Black (-1)"
        lines.append(
            f"Turn: {turn_str} | Halfmove clock: {self.halfmove_clock} | "
            f"Ply: {len(self.history)} | Hash: 0x{self.zobrist_hash:016x}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render_ascii()

