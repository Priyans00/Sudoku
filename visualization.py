import matplotlib.pyplot as plt
import numpy as np

from sudoku_generator import generate_sudoku, print_sudoku
from matrix_representation import sudoku_to_matrix
from solver_engine import solve_via_rank_nullity, _format_grid


def display_grid(grid, difficulty="Unknown"):
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(10):
        linewidth = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, linewidth=linewidth, color='black')
        ax.axvline(i, linewidth=linewidth, color='black')

    for i in range(9):
        for j in range(9):
            val = grid[i][j]
            if val != 0:
                ax.text(j+0.5, 8.5-i, str(val),
                        ha='center', va='center',
                        fontsize=14, color='black')
            else:
                ax.text(j+0.5, 8.5-i, ".",
                        ha='center', va='center',
                        fontsize=10, color='gray')

    plt.title(f"Sudoku — {difficulty}")
    plt.gca().invert_yaxis()
    plt.show()


def run_demo():
    # Generate a Sudoku puzzle with specified difficulty
    difficulty_level = "medium"
    grid = generate_sudoku(difficulty_level)

    print(f"\n{'=' * 50}")
    print(f"Generated {difficulty_level.upper()} Sudoku Puzzle (0 = empty)")
    print(f"{'=' * 50}\n")
    print_sudoku(grid)

    # Convert Sudoku grid to matrix system Ax = b
    A, b = sudoku_to_matrix(grid)
    print()

    # Solve the puzzle using rank-nullity analysis
    solved_grid, difficulty = solve_via_rank_nullity(A, b, grid)

    if solved_grid is not None:
        print(f"\n{'=' * 50}")
        print(f"SOLVED PUZZLE")
        print(f"{'=' * 50}\n")
        print(_format_grid(solved_grid))
        print()

        # Display grids side-by-side
        display_grid(grid, f"Original — {difficulty_level.upper()}")
        display_grid(solved_grid, f"Solution — {difficulty.upper()}")
    else:
        print(f"\n{'=' * 50}")
        print(f"SOLVER FAILED")
        print(f"Difficulty classification: {difficulty}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    run_demo()