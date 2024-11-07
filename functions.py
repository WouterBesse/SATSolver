def encode_sudoku(puzzle, size):
    """
    Encode the Sudoku puzzle into CNF (Conjunctive Normal Form) clauses.
    """
    clauses = []

    # Add constraints for cells that are already filled
    for r in range(size):
        for c in range(size):
            if puzzle[r][c] != 0:
                # Encode the pre-filled value
                num = puzzle[r][c]
                clauses.append([r * size * size + c * size + num])

    # Add constraints for row uniqueness
    for r in range(size):
        for num in range(1, size + 1):
            clauses.append([(r * size * size + c * size + num) for c in range(size)])

    # Add constraints for column uniqueness
    for c in range(size):
        for num in range(1, size + 1):
            clauses.append([(r * size * size + c * size + num) for r in range(size)])

    # Add constraints for subgrid uniqueness (2x2 subgrid)
    subgrid_size = int(size ** 0.5)
    for grid_r in range(0, size, subgrid_size):
        for grid_c in range(0, size, subgrid_size):
            for num in range(1, size + 1):
                clauses.append(
                    [(row * size * size + col * size + num) 
                     for row in range(grid_r, grid_r + subgrid_size)
                     for col in range(grid_c, grid_c + subgrid_size)]
                )
    
    return clauses
def faultytest_solve_sudoku(puzzle, size=4):
    """
    Solve the Sudoku puzzle using the DPLL algorithm.
    """
    clauses = encode_sudoku(puzzle, size)
    assignment = {}

    # Solve the Sudoku using the DPLL algorithm
    with tqdm(total=2**len([c for c in clauses if len(c) == 1]), desc="Solving Sudoku") as pbar:
        solution = dpll(clauses, assignment, size, pbar)
    
    if solution is None:
        print("No solution found.")
        return None

    # Convert the assignment (solution) into a 2D list format
    solved_grid = [[0] * size for _ in range(size)]
    for var, val in solution.items():
        if val:
            # Decode the variable index to row, col, and num
            row = (var - 1) // (size * size)
            col = ((var - 1) % (size * size)) // size
            num = (var - 1) % size + 1
            solved_grid[row][col] = num

    # Print the solved Sudoku grid
    print("Solved Sudoku:")
    for row in solved_grid:
        print(row)

    return solved_grid
#Test functions to use when needed
def old2_dpll(rules: list[list[int]], unassigned: dict[int, bool], assigned: dict[int, bool], size, pbar: tqdm, history: list[int] = [0]) -> dict[int, bool] | None:
    """
    Solves a set of clauses using the DPLL algorithm.

    Args:
        rules (list[list[int]]): A list of clauses, each as a list of literals.
        unassigned (dict[int, bool]): Variables that need to be solved.
        assigned (dict[int, bool]): Fixed variables that cannot be changed.
        size (int): Size of the grid (e.g., for Sudoku).
        pbar (tqdm): Progress bar for the solver.
        history (list[int]): List of literals representing the path of the solver.

    Returns:
        dict[int, bool] | None: Satisfying assignment if one exists, otherwise None.
    """
    print("Start DPLL with history:", history)

    # Simplify the clauses using unit propagation and pure literal elimination
    clauses = single_literal(rules, unassigned)
    if clauses is None:
        return None  # Conflict found
    if not clauses:
        return {**assigned, **unassigned}  # All clauses satisfied, return combined solution

    clauses = pure_literal(clauses, unassigned)
    if not clauses:
        return {**assigned, **unassigned}  # All clauses satisfied, return combined solution

    # Select an unassigned variable (choose from the unassigned variables, not the assigned ones)
    unassigned_vars = [var for clause in clauses for var in clause if var in unassigned]
    unique_vars = list(set(unassigned_vars))

    print('Unassigned variables:', unique_vars)
    if not unique_vars:
        print("No unassigned variables left")
        return None  # No unassigned variables left, puzzle is inconsistent

    literal = unique_vars[0]  # Pick the first unassigned literal
    print(f"Trying literal: {literal} with unassigned={unassigned}")

    # Try assigning True to the literal
    print(f"Assigning {literal} = True")
    unassigned[literal] = True
    updated_clauses = [c for c in clauses if literal not in c]  # Remove satisfied clauses
    result = dpll(updated_clauses, unassigned.copy(), assigned, size, pbar, history + [literal])
    if result:
        return result  # If assignment works, return it

    # If assigning True doesn't work, backtrack and try False
    unassigned[literal] = False
    updated_clauses = [c for c in clauses if -literal not in c]  # Remove satisfied clauses by -literal
    print(f"Assigning {literal} = False")
    result = dpll(updated_clauses, unassigned.copy(), assigned, size, pbar, history + [-literal])
    if result:
        return result  # If assignment works, return it

    # Backtrack
    print(f"No solution for {literal}, backtracking...")
    unassigned.pop(literal)  # Reset unassigned value
    return None

def solve_sudoku_spare(rules: list[list[int]], puzzle: list[list[int]], size) -> list[list[int]] | None:
    """
    Solves a Sudoku puzzle using given rules and an initial puzzle setup.

    Args:
        rules (list[list[int]]): A list of rules in CNF form for Sudoku constraints.
        puzzle (list[list[int]]): A grid of given size representing the puzzle, with `0` for empty cells.

    Returns:
        list[list[int]] | None: A solved Sudoku grid if a solution exists, otherwise None.
    """
    assignment = {} # the sudoku
    values2solve = {}
    cells_unknown = 0

    print(puzzle)
    for i, row in enumerate(puzzle):
        for j, value in enumerate(row):
            if value != 0:  # if 0, cell is empty, if not 0, it is a pre-filled cell
                # Make the value in row i, column j True if it is the value from the sudoku, all other numbers in that place False
                for k in range(size):
                    assignment[(i + 1) * 100 + (j + 1) * 10 + k + 1] = k + 1 == value
            else:
                for k in range(size):
                    values2solve[((i + 1) * 100 + (j + 1) * 10 + k + 1)] = k + 1 == value
                cells_unknown += 1
    print(f'The unknown values are: {values2solve}')
    # print(f'The known values are: {assignment}')

    with tqdm(total=2**cells_unknown, desc="Solving Sudoku") as pbar:
        solution = dpll(rules, values2solve, size, pbar)
    if not solution:
        return None

    return solution
def old_dpll(rules: list[list[int]], assignment: dict[int, bool], size,  pbar: tqdm, history: list[int] = [0]) -> dict[int, bool] | None:
    """
    Solves a set of clauses using the DPLL algorithm.

    Args:
        clauses (list[list[int]]): A list of clauses, each as a list of literals.
        assignment (dict[int, bool]): Current variable assignments, aka the sudoku.
        pbar (tqdm): Progress bar for the solver.
        history (list[int]): List of integers representing the path of the solver.

    Returns:
        dict[int, bool] | None: Satisfying assignment if one exists, otherwise None.
    """
    print("Start DPLL", history)
    # print('The rules are', rules)
    clauses = single_literal(rules, assignment)
    if clauses is None:
        return None  # conflict
    if not clauses:
        return assignment  # all clauses satisfied
    
    # print('After single_literal:', clauses)

    clauses = pure_literal(clauses, assignment)
    if not clauses:
        return assignment  # all clauses satisfied
    
    # print('After pure_literal:', clauses)

    # choose a literal from the first clause
    j = 0
    k = 0
    literal = clauses[0][0]
    print(clauses[0][0])
    var = str(abs(literal))[:2]
    print(var)
    vars_first_two_digits = {str(key)[:2] for key in assignment.keys()}
    print(var)
    
    # Make sure that we don't assign a variable that has already been assigned (i.e. if 113 is true, we can't make 114 true aswell)
    while str(var)[:2] in vars_first_two_digits:
        k += 1
        if k > len(clauses[j]) - 1:
            j += 1
            k = 0
        if j > len(clauses) - 1:
            return None
        literal = clauses[j][k]
        var = str(abs(literal))[:2]

    # Recursive DPLL call
    pbar.update(1)

    # Check all possible values for the cell, I'm not entirely sure this is necessary since it goes through all values in the clauses anyway
    for i in range(size):
        var = int(str(var)[:2] + str(i + 1))
        assignment[var] = True
        if dpll([c for c in clauses if literal not in c], assignment.copy(), size, pbar, history + [1 * var]): # if positive literal
            return assignment

        assignment[var] = False
        if dpll([c for c in clauses if -literal not in c], assignment.copy(), size, pbar, history + [-1 * var]): # if negative literal
            return assignment

    return None