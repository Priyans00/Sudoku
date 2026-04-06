import numpy as np

from sudoku_generator import generate_sudoku, print_sudoku


def sudoku_to_matrix(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a linear system A x = b for a 9x9 Sudoku grid using 81 variables.

    Variable mapping:
    - x[k] corresponds to the scalar value in one Sudoku cell.
    - k = row * 9 + col

    Constraint groups (324 rows total):
    1) Row constraints: 81 rows (9 rows x 9 digit-indexed equations)
    2) Column constraints: 81 rows (9 cols x 9 digit-indexed equations)
    3) Box constraints: 81 rows (9 boxes x 9 digit-indexed equations)
    4) Cell constraints: 81 rows (one equation per cell)

    Note:
    With only 81 scalar variables, this is a compact matrix setup intended for
    structural analysis in later stages (rank/nullity). A strict exact-cover
    encoding would use 729 binary variables, which is intentionally deferred.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a NumPy array")
    if grid.shape != (9, 9):
        raise ValueError("grid must have shape (9, 9)")

    A = np.zeros((324, 81), dtype=float)
    b = np.zeros(324, dtype=float)

    eq = 0

    # 1) Row constraints: each row must total 45 (sum 1..9), repeated per digit index.
    for row in range(9):
        for _digit in range(1, 10):
            for col in range(9):
                var_index = row * 9 + col
                A[eq, var_index] = 1.0
            b[eq] = 45.0
            eq += 1

    # 2) Column constraints: each column must total 45, repeated per digit index.
    for col in range(9):
        for _digit in range(1, 10):
            for row in range(9):
                var_index = row * 9 + col
                A[eq, var_index] = 1.0
            b[eq] = 45.0
            eq += 1

    # 3) 3x3 box constraints: each box must total 45, repeated per digit index.
    for box_row in range(3):
        for box_col in range(3):
            for _digit in range(1, 10):
                for dr in range(3):
                    for dc in range(3):
                        row = box_row * 3 + dr
                        col = box_col * 3 + dc
                        var_index = row * 9 + col
                        A[eq, var_index] = 1.0
                b[eq] = 45.0
                eq += 1

    # 4) Cell constraints: one equation per cell (x_cell equals current grid value).
    # Empty cells remain 0 and are placeholders for later solving stages.
    for row in range(9):
        for col in range(9):
            var_index = row * 9 + col
            A[eq, var_index] = 1.0
            b[eq] = float(grid[row, col])
            eq += 1

    print(f"Matrix A shape: {A.shape}")
    print(f"Vector b shape: {b.shape}")

    if A.shape == (324, 81) and b.shape == (324,):
        print("System set up successfully.")
    else:
        print("System setup has unexpected dimensions.")

    return A, b


if __name__ == "__main__":
    puzzle = generate_sudoku("medium")

    print("Generated Sudoku Puzzle (0 = empty):")
    print_sudoku(puzzle)
    print()

    A, b = sudoku_to_matrix(puzzle)

    non_zeros = int(np.count_nonzero(A))
    density = non_zeros / A.size
    print(f"Matrix summary: non-zeros={non_zeros}, density={density:.4f}")
