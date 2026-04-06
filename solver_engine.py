"""Part 2 solver engine for the Sudoku linear-algebra project."""

from __future__ import annotations

from typing import Any

import numpy as np

from part1 import generate_sudoku, sudoku_to_matrix


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
    print(f"Rank-Nullity check: {rank} + {nullity} = {n_cols} \u2713")


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
