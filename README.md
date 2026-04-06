# Sudoku Solver using Linear Algebra

## 📌 Overview

This project models a Sudoku puzzle as a linear system ( Ax = b ) and analyzes it using linear algebra concepts like Gaussian elimination, rank, and nullity.

---

## ⚙️ Files

* `sudoku_generator.py` – generates valid Sudoku puzzles
* `matrix_representation.py` – converts Sudoku to matrix form
* `solver_engine.py` – performs Gaussian elimination, rank & nullity
* `visualization.py` – displays Sudoku grid and difficulty
* `test_solver.py` – test cases

---

## 🧠 Concept

* 324 equations, 81 variables
* Rank → constraints
* Nullity → degrees of freedom

**Difficulty Logic:**

* Nullity = 0 → Easy
* Small nullity → Medium
* High nullity → Hard

---

## ▶️ Run

```bash
python visualization.py
```

---

## 🎯 Output

* Sudoku grid visualization
* Rank & nullity
* Difficulty level

---

## ✅ Conclusion

This project shows how Sudoku can be solved and analyzed using linear algebra, combining theory with practical implementation.
