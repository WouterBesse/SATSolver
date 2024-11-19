from literalmethods import *
from heuristics import *
from util import *
from tqdm.autonotebook import tqdm

def dpll(clauses: list[list[int]], assignment: dict[int, bool], heuristic: object, pbar: tqdm, history: list[int] = [0]) -> dict[int, bool] | None:
    """DPLL algorithm to solve logic problems and stuff.

    Args:
        clauses (list[list[int]]): List of clauses in dimacs format
        assignment (dict[int, bool]): Dictionary with variables that have a value assigned to them
        pbar (tqdm): Tqdm progress bar, to keep track of brrrrrr
        history (list[int], optional): What variables it has checked. Defaults to [0].

    Returns:
        dict[int, bool] | None: Returns solution if possible, else None
    """
    # Apply single literal rule
    clauses: list[list[int]] | None = single_literal(clauses, assignment)

    if clauses is None:
        # if using vsids update activities
        if isinstance(heuristic, (VSIDS, CHB)):
            heuristic.update(history[-1], conflict=True)
        return None  # conflict
    if not clauses:
        return assignment  # all clauses satisfied

    # Apply pure literal rule
    clauses = pure_literal(clauses, assignment)
    if not clauses:
        return assignment  # all clauses satisfied

    #new for heuristics
    literal = choose_literal(clauses, assignment, heuristic=heuristic)
    if literal is None: # conflict
        if isinstance(heuristic, (VSIDS, CHB)):
            heuristic.update(history[-1], conflict=True)
        return None
    literal = abs(literal)
    
    pbar.update(1)
    # Check both true and false assignments
    for booly in [True, False]:
        updated_assignment = assignment.copy()
        updated_assignment[abs(literal)] = booly
        updated_clauses = propagate(clauses, literal if booly else -literal)
        
        if isinstance(heuristic, CHB):
            # Update CHB heuristic for branching
            heuristic.update(literal, conflict=False)
        
        result = dpll(updated_clauses, updated_assignment, heuristic, pbar, history + [literal if booly else -literal])
        if result:  # If a satisfying assignment is found
            return result
        
    # conflict
    if isinstance(heuristic, (VSIDS, CHB)):
        heuristic.update(history[-1], conflict=True)
    return None  # backtrack

def solve_sudoku(rules: list[list[int]], puzzle: list[list[int]], size: int, heuristic: str = 'random') -> list[list[int]] | None:
    """
    Solves a Sudoku puzzle using given rules and an initial puzzle setup.

    Args:
        rules (list[list[int]]): A list of rules in CNF dimacs form for Sudoku constraints.
        puzzle (list[list[int]]): A 9x9 grid representing the puzzle, with `0` for empty cells.
        size (int): The size of the Sudoku puzzle (e.g. 9 for a 9x9 puzzle).

    Returns:
        list[list[int]] | None: A solved 9x9 Sudoku grid if a solution exists, otherwise None.
    """
    assignment = {} # the sudoku
    amt_unknown = 0
    rcmult = 17 if size == 16 else 10
    for i, row in enumerate(puzzle):
        for j, value in enumerate(row):
            if value != 0:  # if 0, cell is empty, if not 0, it is a pre-filled cell
                # Make the value in row i, column j True if it is the value from the sudoku, all other numbers in that place False
                varnum = (i + 1) * (rcmult**2) + (j + 1) * rcmult + value
                assignment[varnum] = True
                rules = propagate(rules, varnum)
                for k in range(1, size+1):
                    if k != value:
                        assignment[(i + 1) * (rcmult**2) + (j + 1) * rcmult + k] = False
            else:
                amt_unknown += 1
                
    # Choose the heuristic
    match heuristic:
        case 'random':
            heuristiccall = random_literal
        case 'vsids':
            heuristiccall = VSIDS(rules)
        case 'chb':
            heuristiccall = CHB(rules)
        case _:
            heuristiccall = random_literal
            
    with tqdm(total=amt_unknown * size,desc='Trying to find logic solution') as pbar:
        solution = dpll(rules, assignment, heuristiccall, pbar)
    if not solution:
        return None
    
    print('Found logic solution')
    print(solution)
    # parse the solution back to a 9x9 grid
    result = [[0] * size for _ in range(size)]
    for var, value in solution.items():
        if value:  # Ensure var corresponds to a Sudoku cell (1-9 for each row, column)
            
            
            num = var % rcmult
            j = (((var - value) // rcmult) % rcmult) - 1
            i = ((var - value - (j + 1) * rcmult) // (rcmult**2)) - 1
            result[i][j] = num
    return result