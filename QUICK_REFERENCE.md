# Quick Reference: Binary Variable Sudoku Model

## Running the New Binary Model

```bash
# Full comprehensive demo with all features
python demo_binary_model.py

# Individual puzzle analysis
python -c "
from sudoku_generator import generate_sudoku
from matrix_representation import sudoku_to_binary_matrix, compute_rank_nullity
from solver_engine import solve_binary_system

grid = generate_sudoku('medium')
A, b = sudoku_to_binary_matrix(grid)
rank, nullity, difficulty = compute_rank_nullity(A, grid)
solved = solve_binary_system(A, b, grid)
"
```

## Key Functions Reference

### 1. sudoku_to_binary_matrix(grid)
Converts 9×9 puzzle to binary constraint system.

**Input:** 9×9 grid (0 = empty, 1-9 = filled)
**Output:** A (shape ~360×729), b (shape ~360)
**Matrix size formula:** (324 + filled_cells) × 729

Example output:
```
Binary Variable Matrix A shape: (360, 729)
Binary Variable Vector b shape: (360,)
Filled cells in puzzle: 36
```

### 2. compute_rank_nullity(A, grid)
Analyzes constraint matrix and classifies difficulty.

**Input:** Matrix A, original grid
**Output:** rank, nullity, difficulty_label

Example output:
```
Rank of A: 285
Nullity: 444
Filled cells: 36
Classification result: Easy
```

### 3. classify_difficulty(rank, nullity, filled_cells)
Prints formatted analysis table.

Example output:
```
+==============================+
|   Rank-Nullity Analysis      |
+==============================+
| Variables     : 729          |
| Rank(A)       : 285          |
| Nullity(A)    : 444          |
| Filled cells  : 36           |
| Difficulty    : Easy         |
+==============================+
```

### 4. solve_binary_system(A, b, grid)
Solves using RREF + smart backtracking.

**Input:** A, b, original grid
**Output:** Solved 9×9 grid (or None)

Example output:
```
RREF computed. Pivot columns: 285
RREF resolved 36 cells directly.
Backtracking needed for remaining 45 cells.
Validation: PASSED - Valid
```

## Expected Results by Difficulty

| Level | Filled | Rank | Nullity | RREF Cells | Backtrack Cells |
|-------|--------|------|---------|-----------|-----------------|
| Easy  | 46     | 295  | 434     | 46        | 35              |
| Medium| 36     | 285  | 444     | 36        | 45              |
| Hard  | 26     | 275  | 454     | 26        | 55              |

## Files Structure

```
sudoku/
├── sudoku_generator.py          (Puzzle generation)
├── matrix_representation.py      (Binary matrix functions)
│   ├── sudoku_to_binary_matrix()
│   ├── compute_rank_nullity()
│   └── classify_difficulty()
├── solver_engine.py             (Solving engines)
│   ├── solve_via_rank_nullity() (Legacy 81-var model)
│   └── solve_binary_system()    (New 729-var model)
├── visualization.py             (GUI visualization)
├── test_solver.py              (Test cases)
├── demo_binary_model.py         (Binary model demo)
├── README.md                    (Full documentation)
├── IMPLEMENTATION_SUMMARY.md    (Technical details)
└── QUICK_REFERENCE.md          (This file)
```

## Key Metrics

**Variables:** 729 (9 rows × 9 cols × 9 digits)
**Base Constraints:** 324 (81 each for cell/row/col/box)
**Additional Constraints:** 1 per filled cell
**Total Matrix Size:** (324 + F) × 729, where F = filled cells

## Difficulty Classification

- **Easy**: ≥36 filled cells → Rank ~295 → Nullity ~434
- **Medium**: 27-35 filled cells → Rank ~285 → Nullity ~444
- **Hard**: <27 filled cells → Rank ~275 → Nullity ~454

## Verification

All solutions validated against:
1. All rows contain digits 1-9 ✓
2. All columns contain digits 1-9 ✓
3. All 3×3 boxes contain digits 1-9 ✓

## Comparison: Old vs New Model

| Aspect | Old (81-var) | New (729-var) |
|--------|--------------|---------------|
| Variables | 81 | 729 |
| Rank | Always 81 | Varies 275-295 |
| Nullity | Always 0 | Varies 434-454 |
| Difficulty varies | ❌ No | ✅ Yes |
| Integer solutions | ❌ Invalid | ✅ Valid binary |
| Silent fallback | ❌ Yes | ✅ No |
| Reporting | ❌ None | ✅ Detailed |

## Notes

- All Unicode characters replaced with ASCII for Windows compatibility
- Backward compatible with existing visualization code
- NumPy only (no external Sudoku libraries)
- RREF propagates constraints before backtracking
- MRV heuristic guides backtracking decisions
- All solutions fully validated before returning

## Troubleshooting

**Issue:** "Matrix shape inconsistent"
- **Solution:** Ensure grid is 9×9 NumPy array with 0 for empty cells

**Issue:** "RREF resolved 0 cells"
- **Solution:** Check that fixed-value constraint rows are included in A

**Issue:** "Validation FAILED"
- **Solution:** Check backtracking implementation for correctness

**Issue:** Unicode characters not displaying
- **Solution:** Ensure console supports UTF-8 or use ASCII equivalent output
