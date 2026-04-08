"""Part 2 solver engine for the Sudoku linear-algebra project."""

from __future__ import annotations

from typing import Any

import numpy as np

from matrix_representation import sudoku_to_matrix
from sudoku_generator import generate_sudoku


TOLERANCE = 1e-10


def _validate_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize the linear system inputs.

    Args:
        A: Coefficient matrix.
        b: Right-hand-side vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Float copies of ``A`` and ``b``.

    Raises:
        ValueError: If the dimensions are incompatible.
    """
    A_array = np.array(A, dtype=float, copy=True)
    b_array = np.array(b, dtype=float, copy=True).reshape(-1)

    if A_array.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b_array.ndim != 1:
        raise ValueError("b must be a 1D vector.")
    if A_array.shape[0] != b_array.shape[0]:
        raise ValueError("A and b must have the same number of rows.")

    return A_array, b_array


def _rref(matrix: np.ndarray, tolerance: float = TOLERANCE) -> tuple[np.ndarray, list[int]]:
    """Compute the reduced row-echelon form of a matrix.

    Args:
        matrix: Matrix to reduce.
        tolerance: Absolute tolerance used to treat values as zero.

    Returns:
        tuple[np.ndarray, list[int]]: The RREF matrix and pivot column indices.
    """
    rref = np.array(matrix, dtype=float, copy=True)
    n_rows, n_cols = rref.shape
    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break

        candidate_offset = int(np.argmax(np.abs(rref[pivot_row:, col])))
        candidate_row = pivot_row + candidate_offset
        pivot_value = rref[candidate_row, col]

        if abs(pivot_value) <= tolerance:
            continue

        if candidate_row != pivot_row:
            rref[[pivot_row, candidate_row]] = rref[[candidate_row, pivot_row]]

        rref[pivot_row] = rref[pivot_row] / rref[pivot_row, col]

        for row in range(n_rows):
            if row == pivot_row:
                continue

            factor = rref[row, col]
            if abs(factor) > tolerance:
                rref[row] = rref[row] - factor * rref[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1

    rref[np.abs(rref) <= tolerance] = 0.0
    return rref, pivot_cols


def gaussian_elimination(
    A: np.ndarray,
    b: np.ndarray,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Solve ``Ax = b`` using Gaussian elimination with partial pivoting.

    Args:
        A: Coefficient matrix.
        b: Right-hand-side vector.
        tolerance: Absolute tolerance used to treat values as zero.

    Returns:
        tuple[np.ndarray, np.ndarray | None, bool]: The RREF of ``[A|b]``,
        a basic solution vector when the system is consistent, and a boolean
        indicating whether the system is consistent.

    Raises:
        ValueError: If ``A`` and ``b`` do not define a valid linear system.
    """
    A_array, b_array = _validate_linear_system(A, b)
    augmented = np.column_stack((A_array, b_array))
    rref_matrix, pivot_cols = _rref(augmented, tolerance=tolerance)

    n_cols = A_array.shape[1]
    is_consistent = True
    for row in rref_matrix:
        if np.all(np.abs(row[:n_cols]) <= tolerance) and abs(row[-1]) > tolerance:
            is_consistent = False
            break

    if not is_consistent:
        return rref_matrix, None, False

    solution = np.zeros(n_cols, dtype=float)
    for pivot_row, pivot_col in enumerate(pivot_cols):
        if pivot_col >= n_cols:
            continue
        solution[pivot_col] = rref_matrix[pivot_row, -1]

    return rref_matrix, solution, True


def matrix_rank(A: np.ndarray, tolerance: float = TOLERANCE) -> int:
    """Compute matrix rank using pivot count from the matrix RREF.

    Args:
        A: Matrix whose rank should be computed.
        tolerance: Absolute tolerance used to treat values as zero.

    Returns:
        int: Rank of the matrix.

    Raises:
        ValueError: If ``A`` is not two-dimensional.
    """
    A_array = np.array(A, dtype=float, copy=True)
    if A_array.ndim != 2:
        raise ValueError("A must be a 2D array.")

    _, pivot_cols = _rref(A_array, tolerance=tolerance)
    return len(pivot_cols)


def compute_nullity(A: np.ndarray, tolerance: float = TOLERANCE) -> int:
    """Compute matrix nullity via the rank-nullity theorem.

    Args:
        A: Matrix whose nullity should be computed.
        tolerance: Absolute tolerance used to treat values as zero.

    Returns:
        int: Nullity of the matrix.
    """
    A_array = np.array(A, dtype=float, copy=False)
    return int(A_array.shape[1] - matrix_rank(A_array, tolerance=tolerance))


def rank_nullity_report(A: np.ndarray, tolerance: float = TOLERANCE) -> None:
    """Print a formatted rank-nullity summary for a matrix.

    Args:
        A: Matrix to summarize.
        tolerance: Absolute tolerance used to treat values as zero.

    Returns:
        None
    """
    A_array = np.array(A, dtype=float, copy=False)
    if A_array.ndim != 2:
        raise ValueError("A must be a 2D array.")

    n_rows, n_cols = A_array.shape
    rank = matrix_rank(A_array, tolerance=tolerance)
    nullity = compute_nullity(A_array, tolerance=tolerance)

    print(
        f"Rows: {n_rows}  |  Cols: {n_cols}  |  "
        f"Rank: {rank}  |  Nullity: {nullity}"
    )
    print(f"Rank-Nullity check: {rank} + {nullity} = {n_cols} [OK]")


def classify_difficulty(grid: np.ndarray) -> dict[str, int | str]:
    """Classify a Sudoku puzzle from the rank/nullity of its matrix encoding.

    Args:
        grid: Sudoku grid as a ``9 x 9`` NumPy array, where ``0`` marks empty
            cells.

    Returns:
        dict[str, int | str]: Rank, nullity, difficulty label, and empty-cell
        count for the grid.
    """
    A, _ = sudoku_to_matrix(grid)
    rank = matrix_rank(A)
    nullity = compute_nullity(A)
    n_empty_cells = int(np.count_nonzero(grid == 0))

    if nullity == 0:
        difficulty = "Easy"
    elif 1 <= nullity <= 3:
        difficulty = "Medium"
    else:
        difficulty = "Hard"

    return {
        "rank": rank,
        "nullity": nullity,
        "difficulty": difficulty,
        "n_empty_cells": n_empty_cells,
    }


def _classify_difficulty_by_rank_nullity(rank: int, nullity: int) -> str:
    """Classify Sudoku difficulty based on rank and nullity values.

    Args:
        rank: Matrix rank.
        nullity: Matrix nullity.

    Returns:
        str: Difficulty label ("Easy", "Medium", or "Hard").
    """
    if rank >= 310 and nullity <= 14:
        return "Easy"
    elif 290 <= rank <= 309 and 15 <= nullity <= 30:
        return "Medium"
    elif rank < 290 and nullity > 30:
        return "Hard"
    else:
        # Fallback: interpolate based on nullity
        if nullity <= 14:
            return "Easy"
        elif nullity <= 30:
            return "Medium"
        else:
            return "Hard"


def _is_valid_sudoku_cell(grid: np.ndarray, row: int, col: int, value: int) -> bool:
    """Check if placing a value at (row, col) violates Sudoku rules.

    Args:
        grid: Current state of the 9x9 Sudoku grid.
        row: Row index (0-8).
        col: Column index (0-8).
        value: Value to test (1-9).

    Returns:
        bool: True if placing the value is valid.
    """
    # Check row constraint
    if value in grid[row, :]:
        return False

    # Check column constraint
    if value in grid[:, col]:
        return False

    # Check 3x3 box constraint
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    if value in grid[box_row:box_row+3, box_col:box_col+3]:
        return False

    return True


def _solve_sudoku_backtrack(grid: np.ndarray) -> np.ndarray | None:
    """Solve a Sudoku puzzle using backtracking with constraint satisfaction.

    This is a fallback method used when the linear algebra approach fails.

    Args:
        grid: 9x9 Sudoku grid with 0s for empty cells.

    Returns:
        np.ndarray | None: Solved grid if successful, None if unsolvable.
    """
    grid = grid.copy()

    # Find next empty cell using minimum remaining values (MRV) heuristic
    best_cell = None
    best_candidates = None
    best_count = 10

    for row in range(9):
        for col in range(9):
            if grid[row, col] != 0:
                continue

            candidates = [v for v in range(1, 10) if _is_valid_sudoku_cell(grid, row, col, v)]
            count = len(candidates)

            if count == 0:
                return None  # No valid value for this cell
            if count < best_count:
                best_count = count
                best_cell = (row, col)
                best_candidates = candidates
                if best_count == 1:
                    break
        if best_count == 1:
            break

    # If no empty cell found, puzzle is solved
    if best_cell is None:
        return grid

    row, col = best_cell

    # Try each candidate value
    for value in best_candidates:
        grid[row, col] = value
        result = _solve_sudoku_backtrack(grid)
        if result is not None:
            return result
        grid[row, col] = 0

    return None


def _solve_sudoku_backtrack_with_stats(grid: np.ndarray) -> tuple[np.ndarray | None, dict[str, int]]:
    """Solve Sudoku with backtracking and collect search-effort statistics.

    Args:
        grid: 9x9 Sudoku grid with 0 for empty cells.

    Returns:
        tuple[np.ndarray | None, dict[str, int]]: Solved grid (or None) and
        stats including visited nodes and maximum recursion depth.
    """
    working = grid.copy()
    stats = {"nodes": 0, "max_depth": 0}

    def backtrack(depth: int = 0) -> bool:
        stats["nodes"] += 1
        if depth > stats["max_depth"]:
            stats["max_depth"] = depth

        # Find next empty cell using minimum remaining values (MRV) heuristic
        best_cell = None
        best_candidates = None
        best_count = 10

        for row in range(9):
            for col in range(9):
                if working[row, col] != 0:
                    continue

                candidates = [v for v in range(1, 10) if _is_valid_sudoku_cell(working, row, col, v)]
                count = len(candidates)

                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_cell = (row, col)
                    best_candidates = candidates
                    if best_count == 1:
                        break
            if best_count == 1:
                break

        if best_cell is None:
            return True

        row, col = best_cell
        for value in best_candidates:
            working[row, col] = value
            if backtrack(depth + 1):
                return True
            working[row, col] = 0

        return False

    solved = backtrack()
    return (working if solved else None), stats


def _is_valid_sudoku(grid: np.ndarray, tolerance: float = TOLERANCE) -> tuple[bool, str]:
    """Validate a 9x9 Sudoku grid against Sudoku rules.

    Args:
        grid: 9x9 NumPy array to validate.
        tolerance: Absolute tolerance for floating-point comparisons.

    Returns:
        tuple[bool, str]: (is_valid, message) where message explains any violations.
    """
    if grid.shape != (9, 9):
        return False, f"Grid shape is {grid.shape}, expected (9, 9)"

    # Check that all values are integers in the range 1-9
    rounded = np.round(grid).astype(int)
    if not np.all((rounded >= 1) & (rounded <= 9)):
        return False, "Grid contains values outside the range 1-9"

    # Validate rows: each row must contain digits 1-9
    for row in range(9):
        row_vals = rounded[row, :]
        if not np.array_equal(np.sort(row_vals), np.arange(1, 10)):
            return False, f"Row {row} does not contain exactly one of each digit 1-9"

    # Validate columns: each column must contain digits 1-9
    for col in range(9):
        col_vals = rounded[:, col]
        if not np.array_equal(np.sort(col_vals), np.arange(1, 10)):
            return False, f"Column {col} does not contain exactly one of each digit 1-9"

    # Validate 3x3 boxes: each box must contain digits 1-9
    for box_row in range(3):
        for box_col in range(3):
            box_vals = rounded[
                box_row*3:(box_row+1)*3,
                box_col*3:(box_col+1)*3
            ].flatten()
            if not np.array_equal(np.sort(box_vals), np.arange(1, 10)):
                return False, f"Box ({box_row}, {box_col}) does not contain exactly one of each digit 1-9"

    return True, "Valid"


def solve_via_rank_nullity(
    A: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray | None, str]:
    """Solve a Sudoku puzzle using rank-nullity analysis and Gaussian elimination.

    This function implements a four-step solver:
    1. Pre-solve Analysis: Compute rank and nullity, check system consistency
    2. Gaussian Elimination: Apply RREF to solve Ax = b
    3. Grid Reconstruction: Convert solution vector to 9x9 grid and validate
    4. Difficulty Classification: Use rank/nullity to classify puzzle difficulty

    Args:
        A: Coefficient matrix (324 x 81).
        b: Right-hand-side vector (324,).
        grid: Original Sudoku grid for reference.
        tolerance: Absolute tolerance for floating-point comparisons.

    Returns:
        tuple[np.ndarray | None, str]: (solved_grid, difficulty) where solved_grid
        is the completed 9x9 Sudoku grid (or None if unsolvable), and difficulty
        is the classified difficulty level.
    """
    print("=" * 50)
    print("=== Rank-Nullity Analysis ===")
    print("=" * 50)

    # STEP 1: Pre-solve Analysis
    # Compute rank and nullity to understand constraint structure
    A_array, b_array = _validate_linear_system(A, b)
    
    # For solving, we need to remove the cell constraints for empty cells,
    # keeping only non-empty cells and the row/column/box constraints.
    # This is necessary because the original matrix sets empty cells to 0,
    # which would force them to remain empty.
    A_modified = A_array.copy()
    b_modified = b_array.copy()
    
    # Remove cell constraint rows for empty cells (rows 243-323 correspond to cell constraints)
    # Cell constraints are in the last 81 rows
    empty_cell_rows = []
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0:  # Empty cell
                cell_index = row * 9 + col
                constraint_row = 243 + cell_index  # Cell constraints start at row 243
                empty_cell_rows.append(constraint_row)
    
    # Keep only rows that are NOT empty cell constraints
    rows_to_keep = [i for i in range(A_modified.shape[0]) if i not in empty_cell_rows]
    A_modified = A_modified[rows_to_keep, :]
    b_modified = b_modified[rows_to_keep]
    
    rank = matrix_rank(A_array, tolerance=tolerance)
    nullity = compute_nullity(A_array, tolerance=tolerance)

    print(f"Rank(A)    : {rank}")
    print(f"Nullity(A) : {nullity}")
    print(f"Variables  : {A_array.shape[1]}")
    print(f"Constraints (modified): {A_modified.shape[0]} rows (removed {len(empty_cell_rows)} empty cell rows)")
    print()

    # Decision point: Check if the system is uniquely determined
    if nullity == 0:
        print("[OK] Nullity = 0 => Unique solution exists. Proceeding to solve...")
        print()
    else:
        print(f"[!] Nullity = {nullity} > 0 => Under-constrained system.")
        print("    Attempting partial solve with available constraints...")
        print()

    print("=" * 50)
    print("=== Solving via Gaussian Elimination ===")
    print("=" * 50)

    # STEP 2: Gaussian Elimination
    # Apply RREF to the augmented matrix [A | b] to solve Ax = b
    rref_matrix, solution, is_consistent = gaussian_elimination(
        A_modified, b_modified, tolerance=tolerance
    )

    # Check system consistency
    if not is_consistent:
        print("[FAIL] System is INCONSISTENT => No solution exists for this Sudoku.")
        print()
        return None, "Inconsistent"

    if solution is None:
        print("[FAIL] Failed to extract solution vector.")
        print()
        return None, "Failed"

    print("[OK] RREF applied successfully.")
    print(f"  RREF shape: {rref_matrix.shape}")
    print(f"  Solution vector extracted (length: {len(solution)})")
    print()

    print("=" * 50)
    print("=== Grid Reconstruction & Validation ===")
    print("=" * 50)

    # STEP 3: Reconstruct and Validate the Sudoku Grid
    # The linear system approach provides a basic solution, but it may not satisfy
    # the discrete constraint that each cell contains exactly one digit 1-9.
    # If linear solution is invalid, we fall back to backtracking solver.
    solved_grid = None

    try:
        # Round to nearest integer to handle floating-point errors
        solution_rounded = np.round(solution).astype(int)
        solved_grid = solution_rounded.reshape((9, 9))

        # Validate the reconstructed grid
        is_valid, validation_msg = _is_valid_sudoku(solved_grid, tolerance=tolerance)

        if is_valid:
            print("[OK] Grid reconstructed successfully from linear solution.")
            print(f"  {validation_msg}")
            print()
        else:
            print(f"[!] Linear solution produced invalid grid: {validation_msg}")
            print("  Falling back to constraint-satisfaction backtracking solver...")
            print()

            # Fallback: Use backtracking solver
            solved_grid = _solve_sudoku_backtrack(grid)
            if solved_grid is not None:
                print("[OK] Backtracking solver found a valid solution.")
                print()
            else:
                print("[FAIL] Backtracking solver could not find a solution.")
                print()
                return None, "Unsolvable"

    except Exception as e:
        print(f"[!] Error during linear reconstruction: {e}")
        print("  Falling back to constraint-satisfaction backtracking solver...")
        print()

        solved_grid = _solve_sudoku_backtrack(grid)
        if solved_grid is not None:
            print("[OK] Backtracking solver found a valid solution.")
            print()
        else:
            print("[FAIL] Backtracking solver could not find a solution.")
            print()
            return None, "Unsolvable"

    print("=" * 50)
    print("=== Difficulty Classification ===")
    print("=" * 50)

    # STEP 4: Classify Difficulty Based on Rank/Nullity
    # Use the rank and nullity values to determine puzzle difficulty
    difficulty = _classify_difficulty_by_rank_nullity(rank, nullity)

    print(f"Rank   : {rank}")
    print(f"Nullity: {nullity}")
    print(f"Difficulty: {difficulty}")
    print()

    return solved_grid, difficulty


def solve_binary_system(
    A: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    tolerance: float = TOLERANCE,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, dict[str, int]]:
    """
    Solve Sudoku using binary variable model (729 variables) with RREF and backtracking.
    
    This function:
    1. Uses RREF to extract forced assignments from the binary constraint matrix
    2. Applies lightweight backtracking only for remaining free cells
    3. Tracks and reports how many cells were resolved by RREF vs backtracking
    4. Validates the final solution
    
    Args:
        A: Binary constraint matrix (typically shape (324+filled_cells, 729))
        b: Right-hand side binary vector (shape: (324+filled_cells,))
        grid: Original 9×9 Sudoku grid with filled cells (0 = empty)
        tolerance: Floating-point tolerance for RREF
    
    Returns:
        np.ndarray | tuple[np.ndarray | None, dict[str, int]]:
        Solved grid, and optionally solver stats when return_stats=True.
    """
    print("=" * 50)
    print("=== Binary System RREF Solving ===")
    print("=" * 50)
    
    A_array, b_array = _validate_linear_system(A, b)
    
    # Apply RREF to constraint matrix
    augmented = np.column_stack((A_array, b_array))
    rref_matrix, pivot_cols = _rref(augmented, tolerance=tolerance)
    
    print(f"RREF computed. Pivot columns: {len(pivot_cols)}")
    
    # Initialize working grid
    working_grid = grid.copy()
    rref_resolved_count = int(np.count_nonzero(working_grid))  # Count already filled cells
    
    # Extract forced assignments from RREF
    n_vars = A_array.shape[1]  # Should be 729
    binary_solution = np.zeros(n_vars, dtype=int)
    
    # Set filled cells directly
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                digit = int(grid[r, c]) - 1
                var_idx = r * 81 + c * 9 + digit
                binary_solution[var_idx] = 1
    
    # Extract deterministic assignments from RREF
    for pivot_row, pivot_col in enumerate(pivot_cols):
        if pivot_col < n_vars:  # Valid variable column
            rhs_value = rref_matrix[pivot_row, -1]
            # Check if this is a binary variable that's uniquely determined to 1
            if abs(rhs_value - 1.0) < tolerance and pivot_col < n_vars:
                row_values = rref_matrix[pivot_row, :n_vars]
                # Check if this row represents a single variable (= 1)
                nonzero_count = np.count_nonzero(np.abs(row_values) > tolerance)
                if nonzero_count == 1 and abs(rref_matrix[pivot_row, pivot_col] - 1.0) < tolerance:
                    binary_solution[pivot_col] = 1
    
    # Reconstruct grid from binary solution
    temp_grid = np.zeros((9, 9), dtype=int)
    for r in range(9):
        for c in range(9):
            for d in range(9):
                var_idx = r * 81 + c * 9 + d
                if binary_solution[var_idx] == 1:
                    temp_grid[r, c] = d + 1
                    break
    
    # Count cells resolved by RREF
    rref_resolved_count = int(np.count_nonzero(temp_grid))
    
    # Use filled cells from original grid where RREF didn't assign
    for r in range(9):
        for c in range(9):
            if temp_grid[r, c] == 0 and grid[r, c] != 0:
                temp_grid[r, c] = grid[r, c]
    
    print(f"RREF resolved {rref_resolved_count} cells directly.")
    
    # Lightweight backtracking for remaining empty cells
    unsolved_count = 81 - int(np.count_nonzero(temp_grid))
    search_nodes = 0
    search_depth = 0
    if unsolved_count > 0:
        print(f"Backtracking needed for remaining {unsolved_count} cells.")
        solved_grid, bt_stats = _solve_sudoku_backtrack_with_stats(temp_grid)
        search_nodes = int(bt_stats["nodes"])
        search_depth = int(bt_stats["max_depth"])
        print(f"Backtracking search nodes: {search_nodes}")
        if solved_grid is None:
            print("Backtracking failed to find solution.")
            if return_stats:
                return None, {
                    "rref_resolved_cells": rref_resolved_count,
                    "backtracking_cells": unsolved_count,
                    "search_nodes": search_nodes,
                    "search_max_depth": search_depth,
                }
            return None
        backtrack_count = unsolved_count
    else:
        solved_grid = temp_grid
        backtrack_count = 0
    
    # Validate solution
    is_valid, validation_msg = _is_valid_sudoku(solved_grid, tolerance=tolerance)
    if is_valid:
        print(f"Validation: PASSED - {validation_msg}")
    else:
        print(f"Validation: FAILED - {validation_msg}")
        if return_stats:
            return None, {
                "rref_resolved_cells": rref_resolved_count,
                "backtracking_cells": backtrack_count,
                "search_nodes": search_nodes,
                "search_max_depth": search_depth,
            }
        return None
    
    print()
    stats = {
        "rref_resolved_cells": rref_resolved_count,
        "backtracking_cells": backtrack_count,
        "search_nodes": search_nodes,
        "search_max_depth": search_depth,
    }
    if return_stats:
        return solved_grid, stats
    return solved_grid


def _format_grid(grid: np.ndarray) -> str:
    """Return a readable string representation of a Sudoku grid.

    Args:
        grid: Sudoku grid to format.

    Returns:
        str: Human-readable grid layout with box separators.
    """
    lines: list[str] = []
    for row in range(9):
        if row > 0 and row % 3 == 0:
            lines.append("------+-------+------")

        values: list[str] = []
        for col in range(9):
            if col > 0 and col % 3 == 0:
                values.append("|")
            values.append(str(int(grid[row, col])))
        lines.append(" ".join(values))
    return "\n".join(lines)


def sample_workflow() -> dict[str, Any]:
    """Run the full Part 2 Sudoku solver workflow as a standalone demo.

    Args:
        None

    Returns:
        dict[str, Any]: Collected outputs from the demo pipeline.
    """
    grid = generate_sudoku()
    print("=== Step 1: Generated Sudoku Grid ===")
    print(_format_grid(grid))
    print()

    A, b = sudoku_to_matrix(grid)
    print("=== Step 2: Matrix System ===")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    print()

    rref_matrix, solution_vector, is_consistent = gaussian_elimination(A, b)
    preview = np.column_stack((rref_matrix[:5, :10], rref_matrix[:5, -1]))
    print("=== Step 3: Gaussian Elimination ===")
    print(f"Consistent system: {is_consistent}")
    print(f"RREF shape: {rref_matrix.shape}")
    print("RREF preview (first 5 rows, first 10 cols, and RHS):")
    print(np.array2string(preview, precision=3, suppress_small=True))
    if solution_vector is None:
        print("Solution vector: None")
    else:
        print("Solution preview (first 10 entries):")
        print(np.array2string(solution_vector[:10], precision=3, suppress_small=True))
    print()

    print("=== Step 4: Rank / Nullity Report ===")
    rank_nullity_report(A)
    print()

    classification = classify_difficulty(grid)
    print("=== Step 5: Difficulty Classification ===")
    print(classification)

    return {
        "grid": grid,
        "A": A,
        "b": b,
        "rref_matrix": rref_matrix,
        "solution_vector": solution_vector,
        "is_consistent": is_consistent,
        "classification": classification,
    }


if __name__ == "__main__":
    sample_workflow()
