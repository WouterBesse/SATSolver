from literalmethods import *
from heuristics import *
from util import *
from tqdm.autonotebook import tqdm
from collections import defaultdict
import argparse
import wandb
import time

def forward_dpll(
    clauses: list[list[int]],
    assignment: dict[int, bool],
    heuristics: dict[str, HEURISTIC],
    conflict_counter: list[int],
    history: list[int]
    ) -> list[list[int]] | dict[int, bool] | None:
       
    # Apply single literal rule
    clauses: list[list[int]] | None = single_literal(clauses, assignment)
    
    if clauses is None:
        # Update both heuristics on conflict
        update_heuristics(history[-1], conflict=True, heuristics=heuristics)

        conflict_counter[0] += 1
        return None  # Conflict
    if not clauses:
        return assignment  # All clauses satisfied

    # Apply pure literal rule
    clauses = pure_literal(clauses, assignment)
    if not clauses:
        return assignment  # All clauses satisfied
    
    return clauses

    

def dpll_recursive(
    clauses: list[list[int]],
    assignment: dict[int, bool],
    current_heuristic: str,
    heuristics: dict[str, HEURISTIC],
    pbar: tqdm,
    history: list[int],
    conflict_counter: list[int],
    restart_threshold: int,
    first: bool = False,
    globaldecisions: list[int] = [0]
) -> dict[int, bool] | None:
    """ Recursive DPLL algorithm with restarts and heuristics.

    Args:
        clauses (list[list[int]]): List of clauses in DIMACS format.
        assignment (dict[int, bool]): Dictionary of assigned variables.
        current_heuristic (str): Name of the current heuristic to use.
        heuristics (dict[str, object]): Dictionary of heuristics to use.
        pbar (tqdm): Progress bar for tracking progress.
        history (list[int]): Track variable history.
        conflict_counter (int): amount of conflicts
        restart_threshold (int): amount of conflicts before restart.

    Returns:
        dict[int, bool] | None: Solution if satisfiable, else None.
    """
    if conflict_counter[0] >= restart_threshold:
        return None
    
    if not first:        
        forward_dpll_result = forward_dpll(clauses, assignment, heuristics, conflict_counter, history)
        if isinstance(forward_dpll_result, dict):
            return forward_dpll_result
        if forward_dpll_result is None:
            return None
        clauses = forward_dpll_result
    
    # Select a literal using the heuristic
    literal = choose_literal(clauses, assignment, heuristics, current_heuristic, conflict_counter, history, restart_threshold)
    if literal is None:
        return None
    
    pbar.update(1)
    if current_heuristic == 'ranran':
        pbar.set_description(f"Literal: {literal}")
    else:
        pbar.set_description(f"Literal: {literal}, {heuristics['vsids']} --- {heuristics['chb']}")

    # Check both true and false assignments
    for booly in [True, False]:
        updated_assignment = assignment.copy()
        updated_assignment[literal] = booly
        updated_clauses = propagate(clauses, literal if booly else -literal)

        # Update both heuristics during propagation
        update_heuristics(literal if booly else -literal, conflict=False, heuristics=heuristics)
        globaldecisions[0] += 1
        result = dpll_recursive(
            updated_clauses,
            updated_assignment,
            current_heuristic,
            heuristics,
            pbar,
            history + [literal if booly else -literal],
            conflict_counter,
            restart_threshold,
            globaldecisions=globaldecisions
        )
        # if result is int update conflict counter else return result
        if result:
            return result

    # Conflict occurred
    if history and history[-1] != 0:
        update_heuristics(history[-1], conflict=True, heuristics=heuristics)

    conflict_counter[0] += 1
    return None  # Backtrack

def dpll_with_restarts(
    clauses: list[list[int]],
    assignment: dict[int, bool],
    heuristics: dict[str, HEURISTIC],
    history: list[int] = [0],
    max_restarts: int = 10,
    restart_schedule: list[int] | None = None,
    schedule: str = 'robinre'
) -> tuple[dict[int, bool] | None, float, int, int, int]:
    """
    DPLL algorithm with restart capability, alternating between VSIDS and CHB heuristics.

    Args:
        clauses (list[list[int]]): List of clauses in DIMACS format.
        assignment (dict[int, bool]): Dictionary of assigned variables.
        heuristics (dict[str, object]): Dictionary of heuristics to use.
        pbar (tqdm): Progress bar for tracking progress.
        history (list[int], optional): Track variable history. Defaults to [0].
        max_restarts (int): Maximum number of restarts allowed. Defaults to 10.
        restart_schedule (list[int], optional): Schedule of conflicts to trigger restarts.
            If None, uses geometric progression.

    Returns:
        dict[int, bool] | None: Solution if satisfiable, else None.
    """
    # Default restart schedule: geometric progression
    starttime = time.time()
    if restart_schedule is None:
        restart_schedule = [5 * (2 ** i) for i in range(max_restarts)]

    conflict_counter = [0]
    globalconflics = 0
    globaldecisions = [0]
    restart_count = 0

    # Start with VSIDS
    heursticnames = list(heuristics.keys())

    with tqdm(range(max_restarts), desc=f"Starting with {heursticnames[0]} heuristic - {schedule}") as pbar1:
        
        # Apply single literal rule, we do this once here already because the first time is always the same and takes the longest
        forward_dpll_result = forward_dpll(clauses, assignment, heuristics, conflict_counter, history)
        if isinstance(forward_dpll_result, dict):
            return forward_dpll_result, time.time() - starttime, globalconflics, restart_count, globaldecisions[0]
        if forward_dpll_result is None:
            return None, time.time() - starttime, globalconflics, restart_count, globaldecisions[0]
        clauses = forward_dpll_result
        
        original_clauses = clauses.copy()    
        
        for restart_count in pbar1:
            # Initialize/reset the assignment and history for a new attempt
            if schedule == 'robinre':
                current_heuristic = heursticnames[restart_count % 2] # Rotate through heuristics
            elif schedule == 'snip':
                current_heuristic = heursticnames[0] if restart_count < max_restarts // 2 else heursticnames[1]
            elif schedule == 'random':
                current_heuristic = 'random'
            elif schedule == 'rr':
                current_heuristic = 'rr'
            elif schedule == 'vsids':
                current_heuristic = 'vsids'
            elif schedule == 'chb':
                current_heuristic = 'chb'
            elif schedule == 'sss':
                current_heuristic = 'sss'
            else:
                current_heuristic = 'ranran'
                heuristics = {}
            current_assignment = assignment.copy()
            
            with tqdm(total=restart_schedule[restart_count], desc=f"Recursive loop") as pbar:
                result = dpll_recursive(
                    clauses,
                    current_assignment,
                    current_heuristic,
                    heuristics,
                    pbar,
                    history,
                    conflict_counter,
                    restart_schedule[restart_count],
                    first=True,
                    globaldecisions=globaldecisions
                )
                # pbar.close()

            if isinstance(result, dict):  # Satisfiable
                globalconflics += conflict_counter[0]
                pbar1.update(0)
                return result, time.time() - starttime, globalconflics, restart_count, globaldecisions[0]

            # Restart: Reset the conflict counter and toggle the heuristic
            globalconflics += conflict_counter[0]
            conflict_counter = [0]
            restart_count += 1
            clauses = original_clauses.copy()  # Reset clauses for restart
            
            if current_heuristic == 'ranran':
                pbar1.set_description(f"Restarting")
            else:
                pbar1.set_description(f"Restarting with {heursticnames[restart_count % 2]} heuristic, {heuristics['vsids']} --- {heuristics['chb']}")

    return None, time.time() - starttime, globalconflics, restart_count, globaldecisions[0]  # If all restarts fail, return unsatisfiable

def solve_sudoku(
    rules: list[list[int]], 
    puzzle: list[list[int]] | list[int], 
    size: int, 
    heuristiclist: list[str] = ['vsids', 'chb'],
    schedulere: str = 'robinre',
    dimacs: bool = False,
    max_restarts: int = 10
) -> tuple[list[list[int]] | None,float,int,int,int, list[int] | None]:
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
    rcmult = 17 if size == 16 else 10
    if not dimacs:
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
        for lit in puzzle:
            if lit > 0:
                assignment[lit] = True
                rules = propagate(rules, lit)
                
                num, i, j = decode_dimacs(lit, rcmult)
                for k in range(1, size+1):
                    if k != num:
                        assignment[(i + 1) * (rcmult**2) + (j + 1) * rcmult + k] = False
                
    # Choose the heuristic
    heuristics = defaultdict()
    for heuristic in heuristiclist:
        if heuristic.lower() == 'vsids':
            heuristics['vsids'] = VSIDS(rules)
        elif heuristic.lower() == 'chb':
            heuristics['chb'] = CHB(rules)
        else:
            raise ValueError(f"Invalid heuristic: {heuristic}")

    solution, takentime, conflicts, restarts, decisions = dpll_with_restarts(rules, assignment, heuristics, schedule=schedulere, max_restarts=max_restarts)
    if not solution:
        return None, takentime, conflicts, restarts, decisions, None
    
    print(f'Found logic solution in {takentime:.2f} seconds with {conflicts} conflicts, {restarts} restarts and {decisions} decisions')
    # print(solution)
    solutiondimacs = []
    for var, value in solution.items():
        if value:
            solutiondimacs.append(var)
    # parse the solution back to a 9x9 grid
    result = [[0] * size for _ in range(size)]
    for var, value in solution.items():
        if value:  # Ensure var corresponds to a Sudoku cell (1-9 for each row, column)
            num, i, j = decode_dimacs(var, rcmult)
            result[i][j] = num
    return result, takentime, conflicts, restarts, decisions, solutiondimacs

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-S1', action='store_true', help='Use the first heuristic strategy.')
    argparser.add_argument('-S2', action='store_true', help='Use the second heuristic strategy.')
    argparser.add_argument('-S3', action='store_true', help='Use the third heuristic strategy.')
    argparser.add_argument('-S4', action='store_true', help='Use the fourth heuristic strategy.')
    argparser.add_argument('-S5', action='store_true', help='Use the fifth heuristic strategy.')
    argparser.add_argument('-S6', action='store_true', help='Use the sixth heuristic strategy.')
    argparser.add_argument('-S7', action='store_true', help='Use the seventh heuristic strategy.')
    argparser.add_argument('-S8', action='store_true', help='Use the eighth heuristic strategy.')
    argparser.add_argument('puzzle', help='Path to the file containing the Sudoku puzzle.')
    argparser.add_argument('--rules', help='Path to the file containing the rules in DIMACS format.', default='./sudoku/sudoku-rules-9x9.txt')
    argparser.add_argument('--batch', action='store_true', help='Batch mode: solve all puzzles in the file.')
    argparser.add_argument('--schedule', default='robinre', help='Schedule for the restarts.')
    args = argparser.parse_args()
    
    # convert s1, s2 etc to heuristic names make dict
    schedulelist = ['ranran', 'vsids', 'chb', 'rr', 'snip', 'sss', 'random', 'robinre']
    
    if not args.batch:
        if args.puzzle is None:
            raise ValueError("Please provide a path to the Sudoku puzzle.")
        elif args.puzzle[-3:].upper() == 'CNF':
            puzzles, rules, size = load_sudoku_dimacs(args.puzzle)
            slist = [args.S1, args.S2, args.S3, args.S4, args.S5, args.S6, args.S7, args.S8]
            assert sum(slist) == 1, "Please select exactly one heuristic strategy."
            heuristic = slist.index(True) + 1
            solution, takentime, conflicts, restarts, decisions, dimacs = solve_sudoku(rules, puzzles, size, schedulere=schedulelist[heuristic], dimacs=True, max_restarts=20)
            
            if solution is not None:
                print("Solution:")
                for row in solution:
                    print(row)
            else:
                print("No solution found.")
            
            if dimacs is not None:
                export_sudoku_dimacs(dimacs, args.puzzle[:-4] + '.out')
        else:
            rules = load_rules(args.rules)
            puzzles, size = load_sudoku(args.puzzle)
            solution, _, _, _, _, _ = solve_sudoku(rules, puzzles[0], size)
            
            if solution is not None:
                print("Solution:")
                for row in solution:
                    print(row)
            else:
                print("No solution found.")        
            
    else:
        sched = args.schedule
        wandb.init(project='sudoku-solver', name='9x9' + sched)
        rules = load_rules('./sudoku/sudoku-rules-9x9.txt')
        puzzles, size = load_sudoku('./sudoku/1000 sudokus.txt')
        for i, puzzle in enumerate(puzzles):
            print(f"Solving puzzle {i + 1}...")
            solution, takentime, conflicts, restarts, decisions, _  = solve_sudoku(rules, puzzle, size, schedulere=sched) 
            if solution:
                print("Solution:")
                for row in solution:
                    print(row)
            else:
                print("No solution found.")
                
            wandb.log({'success': 1 if solution is not None else 0 ,'time': takentime, 'conflicts': conflicts, 'decisions': decisions, 'restarts': restarts})
            print()
            
        wandb.finish()
        wandb.init(project='sudoku-solver', name='16x16' + sched)
        rules = load_rules('./sudoku/sudoku-rules-16x16.txt')
        puzzles, size = load_sudoku('./sudoku/16x16.txt')
        for i, puzzle in enumerate(puzzles):
            print(f"Solving puzzle {i + 1}...")
            solution, takentime, conflicts, restarts, decisions, _  = solve_sudoku(rules, puzzle, size, schedulere=sched)
            if solution:
                print("Solution:")
                for row in solution:
                    print(row)
            else:
                print("No solution found.")
                
            wandb.log({'success': 1 if solution is not None else 0 ,'time': takentime, 'conflicts': conflicts, 'decisions': decisions, 'restarts': restarts})
            print()
    