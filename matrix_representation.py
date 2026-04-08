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


def sudoku_to_binary_matrix(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a binary variable Sudoku constraint matrix A and vector b.

    Uses 729 binary variables: x[r][c][d] = 1 if cell (r,c) contains digit d, else 0.
    
    Total variables: 9 rows × 9 cols × 9 digits = 729
    
    Constraint matrix A shape: (324, 729) with 4 constraint types (81 each):
    1) Cell constraints: each cell has exactly one digit (81 rows)
    2) Row constraints: each digit appears once per row (81 rows)
    3) Column constraints: each digit appears once per column (81 rows)
    4) Box constraints: each digit appears once per 3×3 box (81 rows)
    
    For filled cells, additional fixed-value constraints are added.
    
    Args:
        grid: 9×9 NumPy array (0 = empty, 1-9 = filled cells)
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (A, b) where A is the constraint matrix
        and b is the right-hand side vector.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a NumPy array")
    if grid.shape != (9, 9):
        raise ValueError("grid must have shape (9, 9)")
    
    # Count filled cells for later use
    filled_cells = int(np.count_nonzero(grid))
    
    # Initialize constraint matrix and vector (81 base constraints)
    num_constraints = 324
    constraints = []
    b_values = []
    
    # Variable mapping: x[r, c, d] -> index = r*81 + c*9 + d (0-indexed digit d:0-8)
    # where digit d represents value d+1
    
    # 1) CELL CONSTRAINTS: each cell has exactly one digit (81 constraints)
    for r in range(9):
        for c in range(9):
            row = np.zeros(729, dtype=float)
            for d in range(9):
                var_idx = r * 81 + c * 9 + d
                row[var_idx] = 1.0
            constraints.append(row)
            b_values.append(1.0)
    
    # 2) ROW CONSTRAINTS: each digit appears once per row (81 constraints)
    for r in range(9):
        for d in range(9):
            row = np.zeros(729, dtype=float)
            for c in range(9):
                var_idx = r * 81 + c * 9 + d
                row[var_idx] = 1.0
            constraints.append(row)
            b_values.append(1.0)
    
    # 3) COLUMN CONSTRAINTS: each digit appears once per column (81 constraints)
    for c in range(9):
        for d in range(9):
            row = np.zeros(729, dtype=float)
            for r in range(9):
                var_idx = r * 81 + c * 9 + d
                row[var_idx] = 1.0
            constraints.append(row)
            b_values.append(1.0)
    
    # 4) BOX CONSTRAINTS: each digit appears once per 3×3 box (81 constraints)
    for box_row in range(3):
        for box_col in range(3):
            for d in range(9):
                row = np.zeros(729, dtype=float)
                for dr in range(3):
                    for dc in range(3):
                        r = box_row * 3 + dr
                        c = box_col * 3 + dc
                        var_idx = r * 81 + c * 9 + d
                        row[var_idx] = 1.0
                constraints.append(row)
                b_values.append(1.0)
    
    # Add fixed-value constraints for filled cells
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                digit = int(grid[r, c]) - 1  # Convert to 0-indexed
                row = np.zeros(729, dtype=float)
                var_idx = r * 81 + c * 9 + digit
                row[var_idx] = 1.0
                constraints.append(row)
                b_values.append(1.0)
    
    A = np.array(constraints, dtype=float)
    b = np.array(b_values, dtype=float)
    
    print(f"Binary Variable Matrix A shape: {A.shape}")
    print(f"Binary Variable Vector b shape: {b.shape}")
    
    return A, b


def compute_rank_nullity(A: np.ndarray, grid: np.ndarray) -> tuple[int, int, str]:
    """
    Compute rank and nullity of the binary constraint matrix and classify difficulty.
    
    Args:
        A: Constraint matrix (typically 324×729 or larger with fixed-value constraints)
        grid: Original 9×9 Sudoku grid (for counting filled cells)
    
    Returns:
        tuple[int, int, str]: (rank, nullity, difficulty_label)
    """
    rank = int(np.linalg.matrix_rank(A))
    n_vars = A.shape[1]
    nullity = n_vars - rank
    filled_cells = int(np.count_nonzero(grid))
    
    print(f"Rank of A: {rank}")
    print(f"Nullity: {nullity}")
    print(f"Filled cells: {filled_cells}")
    
    # Baseline clue-density bands (retuned for this generator).
    if filled_cells >= 43:
        difficulty = "Easy"
    elif 32 <= filled_cells < 43:
        difficulty = "Medium"
    else:
        difficulty = "Hard"
    
    return rank, nullity, difficulty


def get_difficulty_breakdown(
    rank: int,
    nullity: int,
    filled_cells: int,
    rref_resolved_cells: int | None = None,
    backtracking_cells: int | None = None,
    search_nodes: int | None = None,
) -> dict[str, int | str]:
    """Compute detailed hybrid difficulty score components.

    Returns a dict with point contributions and final label so callers can
    print or log a concise score explanation.
    """
    # Baseline from clue count (retuned for this generator's clue ranges).
    if filled_cells >= 43:
        base = "Easy"
        base_points = 0
    elif 32 <= filled_cells < 43:
        base = "Medium"
        base_points = 2
    else:
        base = "Hard"
        base_points = 4

    backtrack_points = 0
    if backtracking_cells is not None:
        if backtracking_cells >= 55:
            backtrack_points = 2
        elif backtracking_cells >= 45:
            backtrack_points = 1

    search_points = 0
    if search_nodes is not None:
        if search_nodes >= 180:
            search_points = 2
        elif search_nodes >= 60:
            search_points = 1

    rref_points = 1 if (rref_resolved_cells is not None and rref_resolved_cells <= 30) else 0
    rank_points = 1 if rank <= 275 else 0
    nullity_points = 1 if nullity >= 454 else 0

    total_score = (
        base_points
        + backtrack_points
        + search_points
        + rref_points
        + rank_points
        + nullity_points
    )

    if total_score <= 1:
        difficulty = "Easy"
    elif total_score <= 4:
        difficulty = "Medium"
    else:
        difficulty = "Hard"

    return {
        "difficulty": difficulty,
        "base_label": base,
        "base_points": base_points,
        "backtrack_points": backtrack_points,
        "search_points": search_points,
        "rref_points": rref_points,
        "rank_points": rank_points,
        "nullity_points": nullity_points,
        "total_score": total_score,
    }


def classify_difficulty(
    rank: int,
    nullity: int,
    filled_cells: int,
    rref_resolved_cells: int | None = None,
    backtracking_cells: int | None = None,
    search_nodes: int | None = None,
) -> str:
    """
    Print a formatted rank-nullity analysis table.
    
    Args:
        rank: Matrix rank.
        nullity: Matrix nullity.
        filled_cells: Number of non-zero cells in the original grid.
        rref_resolved_cells: Cells directly determined before search.
        backtracking_cells: Cells left for search/backtracking.
        search_nodes: Backtracking search nodes visited.

    Returns:
        str: Final difficulty label.
    """
    breakdown = get_difficulty_breakdown(
        rank,
        nullity,
        filled_cells,
        rref_resolved_cells=rref_resolved_cells,
        backtracking_cells=backtracking_cells,
        search_nodes=search_nodes,
    )
    difficulty = str(breakdown["difficulty"])
    
    # Use ASCII-compatible characters for better cross-platform support
    print("+==============================+")
    print("|   Rank-Nullity Analysis      |")
    print("+==============================+")
    print(f"| Variables     : 729          |")
    print(f"| Rank(A)       : {rank:<22} |")
    print(f"| Nullity(A)    : {nullity:<22} |")
    print(f"| Filled cells  : {filled_cells:<22} |")
    print(f"| Difficulty    : {difficulty:<22} |")
    if rref_resolved_cells is not None:
        print(f"| RREF cells    : {rref_resolved_cells:<22} |")
    if backtracking_cells is not None:
        print(f"| Backtrack     : {backtracking_cells:<22} |")
    if search_nodes is not None:
        print(f"| Search nodes  : {search_nodes:<22} |")
    print("+==============================+")
    return difficulty


if __name__ == "__main__":
    puzzle = generate_sudoku("medium")

    print("Generated Sudoku Puzzle (0 = empty):")
    print_sudoku(puzzle)
    print()

    A, b = sudoku_to_matrix(puzzle)

    non_zeros = int(np.count_nonzero(A))
    density = non_zeros / A.size
    print(f"Matrix summary: non-zeros={non_zeros}, density={density:.4f}")
