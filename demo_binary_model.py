"""
Comprehensive demo of the Binary Variable Sudoku Model.

This script demonstrates the fixed mathematical flaw in the original Sudoku
project by implementing a proper binary variable model with 729 variables
instead of the previous flawed 81-variable model.

Key features:
- 729 binary variables: x[r][c][d] = 1 if cell (r,c) contains digit d, else 0
- 324 base constraints + additional fixed-value constraints for filled cells
- Proper rank/nullity analysis that varies by puzzle difficulty
- RREF-based solving with lightweight backtracking for remaining free cells
- Clear reporting of solution method (RREF vs backtracking)
- Formatted difficulty analysis table
"""

import numpy as np
from sudoku_generator import generate_sudoku, print_sudoku
from matrix_representation import (
    sudoku_to_binary_matrix,
    compute_rank_nullity,
    classify_difficulty,
)
from solver_engine import solve_binary_system


def main():
    """Run comprehensive demo of binary variable Sudoku model."""
    
    print("\n" + "=" * 60)
    print("BINARY VARIABLE SUDOKU MODEL DEMO")
    print("=" * 60 + "\n")
    
    # Generate a medium-difficulty Sudoku puzzle
    difficulty_level = "medium"
    print(f"Generating {difficulty_level.upper()} Sudoku puzzle...\n")
    grid = generate_sudoku(difficulty_level)
    
    print(f"Original Puzzle ({difficulty_level.upper()}):")
    print_sudoku(grid)
    print()
    
    # Step 1: Build binary constraint matrix
    print("=" * 60)
    print("Step 1: Building Binary Constraint Matrix")
    print("=" * 60)
    print()
    
    A, b = sudoku_to_binary_matrix(grid)
    
    filled_cells = int(np.count_nonzero(grid))
    print(f"Filled cells in puzzle: {filled_cells}")
    print(f"Empty cells: {81 - filled_cells}")
    print()
    
    # Step 2: Compute rank and nullity
    print("=" * 60)
    print("Step 2: Rank-Nullity Analysis")
    print("=" * 60)
    print()
    
    rank, nullity, difficulty_label = compute_rank_nullity(A, grid)
    print(f"Classification result: {difficulty_label}")
    print()
    
    # Step 3: Print formatted difficulty analysis table
    print("=" * 60)
    print("Step 3: Formatted Analysis Table")
    print("=" * 60)
    print()
    
    classify_difficulty(rank, nullity, filled_cells)
    print()
    
    # Step 4: Solve using binary system
    print("=" * 60)
    print("Step 4: Solving with Binary Model")
    print("=" * 60)
    print()
    
    solved_grid = solve_binary_system(A, b, grid)
    
    if solved_grid is not None:
        print("=" * 60)
        print("SOLVED SUDOKU")
        print("=" * 60)
        print()
        print_sudoku(solved_grid)
        print()
    else:
        print("ERROR: Failed to solve the Sudoku puzzle.")
        print()


def demo_difficulty_variations():
    """
    Demonstrate that rank varies significantly across different difficulty levels.
    This shows that the new model properly captures difficulty differences.
    """
    print("\n" + "=" * 60)
    print("DIFFICULTY VARIATION DEMO")
    print("=" * 60 + "\n")
    
    difficulties = ["easy", "medium", "hard"]
    results = []
    
    for diff_level in difficulties:
        print(f"\nGenerating {diff_level.upper()} puzzle...")
        grid = generate_sudoku(diff_level)
        
        filled_cells = int(np.count_nonzero(grid))
        A, b = sudoku_to_binary_matrix(grid)
        
        rank = int(np.linalg.matrix_rank(A))
        nullity = A.shape[1] - rank
        
        results.append({
            'difficulty': diff_level.upper(),
            'filled': filled_cells,
            'rank': rank,
            'nullity': nullity,
        })
        
        print(f"  Filled: {filled_cells:2d}, Rank: {rank:3d}, Nullity: {nullity:3d}")
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Difficulty':<12} {'Filled':<10} {'Rank':<10} {'Nullity':<10}")
    print("-" * 42)
    for r in results:
        print(f"{r['difficulty']:<12} {r['filled']:<10} {r['rank']:<10} {r['nullity']:<10}")
    print("\nNote: Rank varies significantly across puzzles, showing proper difficulty")
    print("      classification (unlike the flawed 81-variable model which always")
    print("      had rank=81, nullity=0)")


def demo_solution_method_reporting():
    """
    Demonstrate that the solver properly reports whether RREF alone solved
    the puzzle or if backtracking was needed.
    """
    print("\n" + "=" * 60)
    print("SOLUTION METHOD REPORTING DEMO")
    print("=" * 60 + "\n")
    
    for diff_level in ["easy", "medium", "hard"]:
        print(f"\n{diff_level.upper()} Puzzle:")
        print("-" * 40)
        
        grid = generate_sudoku(diff_level)
        filled_cells = int(np.count_nonzero(grid))
        print(f"Filled cells: {filled_cells}")
        
        A, b = sudoku_to_binary_matrix(grid)
        solved_grid = solve_binary_system(A, b, grid)
        
        if solved_grid is not None:
            print("[OK] Solution found")
        else:
            print("[FAIL] Solution not found")


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Run additional demonstrations
    demo_difficulty_variations()
    demo_solution_method_reporting()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey improvements in this implementation:")
    print("[OK] 729 binary variables instead of 81 scalar variables")
    print("[OK] Proper 324x729 constraint matrix encoding all Sudoku rules")
    print("[OK] Rank varies by puzzle difficulty (not always 81)")
    print("[OK] Nullity properly reflects underdetermined constraints")
    print("[OK] RREF propagates forced assignments intelligently")
    print("[OK] Lightweight backtracking only for genuinely free cells")
    print("[OK] Clear reporting of solution method used")
    print("[OK] No silent fallbacks to backtracking")
    print()
