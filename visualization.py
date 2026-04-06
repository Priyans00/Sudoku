import matplotlib.pyplot as plt
import numpy as np

from sudoku_generator import generate_sudoku
from matrix_representation import sudoku_to_matrix
from solver_engine import gaussian_elimination, matrix_rank


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
    grid = generate_sudoku()

    A, b = sudoku_to_matrix(grid)

    gaussian_elimination(A, b)

    rank = matrix_rank(A)
    n = A.shape[1]
    nullity = n - rank

    if nullity == 0:
        difficulty = "Easy"
    elif nullity < 5:
        difficulty = "Medium"
    else:
        difficulty = "Hard"

    print(f"Rank: {rank}, Nullity: {nullity}")
    print(f"Difficulty: {difficulty}")

    display_grid(grid, difficulty)


if __name__ == "__main__":
    run_demo()