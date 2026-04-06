"""Pytest suite for the Part 2 Sudoku solver engine."""

from __future__ import annotations

import random

import numpy as np

from part1 import generate_sudoku, sudoku_to_matrix
from solver_engine import (
    classify_difficulty,
    compute_nullity,
    gaussian_elimination,
    matrix_rank,
    rank_nullity_report,
    sample_workflow,
)


def test_gaussian_elimination_returns_rref_and_solution() -> None:
    """Gaussian elimination should produce a consistent RREF and solution."""
    A = np.array([[2.0, 1.0], [1.0, -1.0]])
    b = np.array([5.0, 1.0])

    rref_matrix, solution, is_consistent = gaussian_elimination(A, b)

    expected_rref = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    assert is_consistent is True
    assert solution is not None
    assert np.allclose(rref_matrix, expected_rref)
    assert np.allclose(solution, np.array([2.0, 1.0]))


def test_gaussian_elimination_detects_inconsistent_system() -> None:
    """Gaussian elimination should flag inconsistent systems cleanly."""
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([2.0, 3.0])

    rref_matrix, solution, is_consistent = gaussian_elimination(A, b)

    assert is_consistent is False
    assert solution is None
    assert np.allclose(rref_matrix[-1], np.array([0.0, 0.0, 1.0]))


def test_matrix_rank_and_nullity_use_rref_pivots(capsys) -> None:
    """Rank/nullity helpers should agree with a simple known matrix."""
    A = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
        ]
    )

    rank = matrix_rank(A)
    nullity = compute_nullity(A)
    rank_nullity_report(A)
    captured = capsys.readouterr()

    assert rank == 2
    assert nullity == 1
    assert "Rows: 3  |  Cols: 3  |  Rank: 2  |  Nullity: 1" in captured.out
    assert "Rank-Nullity check: 2 + 1 = 3" in captured.out


def test_classify_difficulty_reports_part1_matrix_properties() -> None:
    """Difficulty classification should return the expected report fields."""
    np.random.seed(42)
    random.seed(42)
    grid = generate_sudoku()

    report = classify_difficulty(grid)
    A, _ = sudoku_to_matrix(grid)

    assert report["rank"] == matrix_rank(A)
    assert report["nullity"] == compute_nullity(A)
    assert report["difficulty"] == "Easy"
    assert report["n_empty_cells"] == int(np.count_nonzero(grid == 0))


def test_sample_workflow_runs_end_to_end(capsys) -> None:
    """The sample workflow should execute and print each pipeline section."""
    np.random.seed(42)
    random.seed(42)

    result = sample_workflow()
    captured = capsys.readouterr()

    assert "=== Step 1: Generated Sudoku Grid ===" in captured.out
    assert "=== Step 2: Matrix System ===" in captured.out
    assert "=== Step 3: Gaussian Elimination ===" in captured.out
    assert "=== Step 4: Rank / Nullity Report ===" in captured.out
    assert "=== Step 5: Difficulty Classification ===" in captured.out
    assert result["is_consistent"] is False
    assert result["classification"]["difficulty"] == "Easy"
