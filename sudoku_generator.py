import random
from typing import List, Optional, Tuple

import numpy as np


Grid = np.ndarray


def _is_valid(board: Grid, row: int, col: int, value: int) -> bool:
    """Return True if placing value at (row, col) keeps Sudoku rules valid."""
    if value in board[row, :]:
        return False
    if value in board[:, col]:
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if value in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False

    return True


def _find_empty_with_fewest_candidates(board: Grid) -> Optional[Tuple[int, int, List[int]]]:
    """
    Choose the next empty cell by minimum remaining values (MRV).
    This speeds up both solving and uniqueness checks significantly.
    """
    best: Optional[Tuple[int, int, List[int]]] = None
    best_count = 10

    for row in range(9):
        for col in range(9):
            if board[row, col] != 0:
                continue

            candidates: List[int] = []
            for value in range(1, 10):
                if _is_valid(board, row, col, value):
                    candidates.append(value)

            count = len(candidates)
            if count == 0:
                return (row, col, [])

            if count < best_count:
                best_count = count
                best = (row, col, candidates)
                if best_count == 1:
                    return best

    return best


def _solve_random(board: Grid) -> bool:
    """Fill board in place using randomized backtracking."""
    next_cell = _find_empty_with_fewest_candidates(board)
    if next_cell is None:
        return True

    row, col, candidates = next_cell
    if not candidates:
        return False

    random.shuffle(candidates)
    for value in candidates:
        board[row, col] = value
        if _solve_random(board):
            return True
        board[row, col] = 0

    return False


def _count_solutions(board: Grid, limit: int = 2) -> int:
    """Count Sudoku solutions up to a limit to test uniqueness quickly."""
    count = 0

    def backtrack() -> None:
        nonlocal count
        if count >= limit:
            return

        next_cell = _find_empty_with_fewest_candidates(board)
        if next_cell is None:
            count += 1
            return

        row, col, candidates = next_cell
        if not candidates:
            return

        for value in candidates:
            board[row, col] = value
            backtrack()
            board[row, col] = 0
            if count >= limit:
                return

    backtrack()
    return count


def _generate_complete_grid() -> Grid:
    """Generate a complete valid 9x9 Sudoku solution grid."""
    board = np.zeros((9, 9), dtype=int)
    _solve_random(board)
    return board


def _remove_cells_with_uniqueness(solution: Grid, target_remove: int) -> Grid:
    """
    Remove numbers while preserving a unique solution.
    If the exact target is not reachable, returns the closest valid puzzle.
    """
    puzzle = solution.copy()
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)

    removed = 0
    stalled_rounds = 0

    while removed < target_remove and stalled_rounds < 4:
        progress_in_round = False

        for row, col in positions:
            if removed >= target_remove:
                break
            if puzzle[row, col] == 0:
                continue

            old_value = puzzle[row, col]
            puzzle[row, col] = 0

            # Keep the removal only if puzzle still has exactly one solution.
            if _count_solutions(puzzle.copy(), limit=2) == 1:
                removed += 1
                progress_in_round = True
            else:
                puzzle[row, col] = old_value

        if progress_in_round:
            stalled_rounds = 0
            random.shuffle(positions)
        else:
            stalled_rounds += 1

    return puzzle


def generate_sudoku(difficulty: str = "medium") -> Grid:
    """
    Generate a Sudoku puzzle as a 9x9 NumPy array where 0 means empty.

    Difficulty removal targets:
    - easy:   about 35 cells removed
    - medium: about 45 cells removed
    - hard:   about 55 cells removed
    """
    difficulty_map = {
        "easy": 35,
        "medium": 45,
        "hard": 55,
    }

    key = difficulty.lower().strip()
    if key not in difficulty_map:
        raise ValueError("difficulty must be one of: easy, medium, hard")

    full_grid = _generate_complete_grid()
    puzzle = _remove_cells_with_uniqueness(full_grid, difficulty_map[key])
    return puzzle.astype(int)


def print_sudoku(grid: Grid) -> None:
    """Print Sudoku grid with 3x3 separators for readability."""
    for row in range(9):
        if row > 0 and row % 3 == 0:
            print("------+-------+------")

        row_values = []
        for col in range(9):
            if col > 0 and col % 3 == 0:
                row_values.append("|")
            row_values.append(str(int(grid[row, col])))

        print(" ".join(row_values))
