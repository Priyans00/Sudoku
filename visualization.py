import os

import matplotlib
import numpy as np
from matplotlib.widgets import Button

# Prefer an interactive backend unless the user explicitly forces MPLBACKEND.
if "MPLBACKEND" not in os.environ:
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        pass

import matplotlib.pyplot as plt

from sudoku_generator import generate_sudoku, print_sudoku
from matrix_representation import (
    classify_difficulty,
    compute_rank_nullity,
    get_difficulty_breakdown,
    sudoku_to_binary_matrix,
)
from solver_engine import _format_grid, solve_binary_system


def _ensure_interactive_backend() -> bool:
    """Try to switch to an interactive Matplotlib backend if currently headless."""
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        return True

    # Try common interactive backends in order.
    for candidate in ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]:
        try:
            plt.switch_backend(candidate)
            active = plt.get_backend().lower()
            if "agg" not in active:
                print(f"Switched Matplotlib backend to {plt.get_backend()} for interactive GUI.")
                return True
        except Exception:
            continue

    return False


def _run_tk_fallback_gui(initial_grid: np.ndarray, solved_grid: np.ndarray, difficulty: str) -> bool:
    """Run a single-window Tkinter Sudoku GUI with step-by-step solve animation.

    Returns True if the GUI was launched, False if Tkinter is unavailable.
    """
    try:
        import tkinter as tk
    except Exception:
        return False

    try:
        root = tk.Tk()
    except Exception:
        return False

    root.title("Sudoku Solver (Step-by-Step)")
    root.geometry("520x620")

    board_size = 450
    margin = 20
    cell = board_size // 9

    canvas = tk.Canvas(root, width=board_size + 2 * margin, height=board_size + 2 * margin, bg="white")
    canvas.pack(pady=10)

    current = initial_grid.copy()
    empty_cells = [(r, c) for r in range(9) for c in range(9) if initial_grid[r, c] == 0]
    step_index = {"value": 0}
    animating = {"value": False}

    def draw_board(highlight: tuple[int, int] | None = None) -> None:
        canvas.delete("all")
        x0 = margin
        y0 = margin

        # Grid lines
        for i in range(10):
            w = 3 if i % 3 == 0 else 1
            canvas.create_line(x0 + i * cell, y0, x0 + i * cell, y0 + 9 * cell, width=w, fill="black")
            canvas.create_line(x0, y0 + i * cell, x0 + 9 * cell, y0 + i * cell, width=w, fill="black")

        # Digits
        for r in range(9):
            for c in range(9):
                v = int(current[r, c])
                if v == 0:
                    text = "."
                    color = "gray"
                    size = 12
                else:
                    text = str(v)
                    color = "black" if initial_grid[r, c] != 0 else "royalblue"
                    size = 20
                if highlight == (r, c):
                    color = "crimson"

                canvas.create_text(
                    x0 + c * cell + cell / 2,
                    y0 + r * cell + cell / 2,
                    text=text,
                    fill=color,
                    font=("Segoe UI", size, "bold" if highlight == (r, c) else "normal"),
                )

    status_var = tk.StringVar(
        value=f"Difficulty: {difficulty} | Click Solve to start step-by-step solving"
    )
    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 11))
    status_label.pack(pady=5)

    def advance() -> None:
        if step_index["value"] >= len(empty_cells):
            animating["value"] = False
            solve_btn.config(text="Solved", state=tk.DISABLED)
            status_var.set(
                f"Difficulty: {difficulty} | Solved: {len(empty_cells)}/{len(empty_cells)} steps"
            )
            draw_board()
            return

        r, c = empty_cells[step_index["value"]]
        current[r, c] = solved_grid[r, c]
        step_index["value"] += 1
        status_var.set(f"Difficulty: {difficulty} | Step {step_index['value']}/{len(empty_cells)}")
        draw_board(highlight=(r, c))
        root.after(170, advance)

    def on_solve() -> None:
        if animating["value"]:
            return
        animating["value"] = True
        solve_btn.config(text="Solving...", state=tk.DISABLED)
        status_var.set(f"Difficulty: {difficulty} | Step 0/{len(empty_cells)}")
        root.after(170, advance)

    solve_btn = tk.Button(root, text="Solve", command=on_solve, font=("Segoe UI", 11, "bold"))
    solve_btn.pack(pady=8)

    draw_board()
    root.mainloop()
    return True


class SudokuInteractiveGUI:
    """Single-window Sudoku GUI that animates solving step by step."""

    def __init__(self, initial_grid: np.ndarray, solved_grid: np.ndarray, difficulty: str) -> None:
        self.initial_grid = initial_grid.copy()
        self.current_grid = initial_grid.copy()
        self.solved_grid = solved_grid.copy()
        self.difficulty = difficulty

        self.empty_cells = [
            (r, c)
            for r in range(9)
            for c in range(9)
            if self.initial_grid[r, c] == 0
        ]
        self.step_index = 0
        self.timer = None
        self.is_animating = False

        self.fig, self.ax = plt.subplots(figsize=(7, 8))
        self.fig.subplots_adjust(bottom=0.18)

        self._setup_board_axes()
        self._draw_grid_lines()
        self.cell_text = [[None for _ in range(9)] for _ in range(9)]
        self._render_board()

        self.status_text = self.fig.text(
            0.5,
            0.08,
            f"Difficulty: {self.difficulty} | Click Solve to start step-by-step solving",
            ha="center",
            va="center",
            fontsize=10,
        )

        button_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.05])
        self.solve_button = Button(button_ax, "Solve")
        self.solve_button.on_clicked(self._on_solve_clicked)

    def _setup_board_axes(self) -> None:
        self.ax.set_xticks(np.arange(10))
        self.ax.set_yticks(np.arange(10))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.invert_yaxis()
        self.ax.set_title("Sudoku Solver (Step-by-Step)")

    def _draw_grid_lines(self) -> None:
        for i in range(10):
            linewidth = 2 if i % 3 == 0 else 0.5
            self.ax.axhline(i, linewidth=linewidth, color="black")
            self.ax.axvline(i, linewidth=linewidth, color="black")

    def _render_board(self, highlight_cell: tuple[int, int] | None = None) -> None:
        for r in range(9):
            for c in range(9):
                if self.cell_text[r][c] is not None:
                    self.cell_text[r][c].remove()

                value = int(self.current_grid[r, c])
                if value == 0:
                    self.cell_text[r][c] = self.ax.text(
                        c + 0.5,
                        r + 0.5,
                        ".",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="gray",
                    )
                else:
                    # Given clues in black, solved/animated values in blue, current step in red.
                    color = "black" if self.initial_grid[r, c] != 0 else "royalblue"
                    if highlight_cell == (r, c):
                        color = "crimson"
                    self.cell_text[r][c] = self.ax.text(
                        c + 0.5,
                        r + 0.5,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=14,
                        color=color,
                        fontweight="bold" if highlight_cell == (r, c) else "normal",
                    )

        self.fig.canvas.draw_idle()

    def _on_solve_clicked(self, _event) -> None:
        if self.is_animating:
            return

        self.is_animating = True
        self.solve_button.label.set_text("Solving...")
        self.status_text.set_text(
            f"Difficulty: {self.difficulty} | Step 0/{len(self.empty_cells)}"
        )

        # Animate filling one cell at a time.
        self.timer = self.fig.canvas.new_timer(interval=180)
        self.timer.add_callback(self._advance_one_step)
        self.timer.start()

    def _advance_one_step(self) -> None:
        if self.step_index >= len(self.empty_cells):
            if self.timer is not None:
                self.timer.stop()
            self.solve_button.label.set_text("Solved")
            self.status_text.set_text(
                f"Difficulty: {self.difficulty} | Solved: {len(self.empty_cells)}/{len(self.empty_cells)} steps"
            )
            self.is_animating = False
            self._render_board()
            return

        r, c = self.empty_cells[self.step_index]
        self.current_grid[r, c] = self.solved_grid[r, c]
        self.step_index += 1

        self.status_text.set_text(
            f"Difficulty: {self.difficulty} | Step {self.step_index}/{len(self.empty_cells)}"
        )
        self._render_board(highlight_cell=(r, c))

    def show(self) -> None:
        backend = plt.get_backend().lower()
        if "agg" in backend:
            print("Interactive GUI is not available with Agg backend.")
            print("Could not switch to an interactive backend (TkAgg/QtAgg/WXAgg).")
            plt.close(self.fig)
            return

        plt.show()


def run_demo():
    # Generate a Sudoku puzzle with specified difficulty
    difficulty_level = "medium"
    grid = generate_sudoku(difficulty_level)

    print(f"\n{'=' * 50}")
    print(f"Generated {difficulty_level.upper()} Sudoku Puzzle (0 = empty)")
    print(f"{'=' * 50}\n")
    print_sudoku(grid)

    # Build binary-variable matrix system Ax = b (729 variables)
    A, b = sudoku_to_binary_matrix(grid)
    print()

    # Rank-nullity analysis on binary model
    rank, nullity, _ = compute_rank_nullity(A, grid)
    filled_cells = int(np.count_nonzero(grid))

    print("=== Solving ===")
    solved_grid, solver_stats = solve_binary_system(A, b, grid, return_stats=True)

    difficulty = classify_difficulty(
        rank,
        nullity,
        filled_cells,
        rref_resolved_cells=solver_stats["rref_resolved_cells"],
        backtracking_cells=solver_stats["backtracking_cells"],
        search_nodes=solver_stats["search_nodes"],
    )
    breakdown = get_difficulty_breakdown(
        rank,
        nullity,
        filled_cells,
        rref_resolved_cells=solver_stats["rref_resolved_cells"],
        backtracking_cells=solver_stats["backtracking_cells"],
        search_nodes=solver_stats["search_nodes"],
    )
    print(
        "Difficulty score breakdown: "
        f"base={breakdown['base_label']}({breakdown['base_points']}), "
        f"backtrack={breakdown['backtrack_points']}, "
        f"search={breakdown['search_points']}, "
        f"rref={breakdown['rref_points']}, "
        f"rank={breakdown['rank_points']}, "
        f"nullity={breakdown['nullity_points']} "
        f"=> total={breakdown['total_score']} => {difficulty}"
    )
    print()

    if solved_grid is not None:
        print(f"\n{'=' * 50}")
        print(f"SOLVED PUZZLE")
        print(f"{'=' * 50}\n")
        print(_format_grid(solved_grid))
        print()

        if not _ensure_interactive_backend():
            print("Matplotlib interactive backend unavailable, trying Tkinter fallback GUI...")
            if not _run_tk_fallback_gui(grid, solved_grid, difficulty):
                print("Interactive GUI disabled: no interactive backend available.")
                print("Install Tkinter or PyQt5, then run: python visualization.py")
            return

        # Single interactive window: unsolved first, then step-by-step solve on click.
        gui = SudokuInteractiveGUI(grid, solved_grid, difficulty)
        gui.show()
    else:
        print(f"\n{'=' * 50}")
        print(f"SOLVER FAILED")
        print(f"Difficulty classification: {difficulty}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    run_demo()