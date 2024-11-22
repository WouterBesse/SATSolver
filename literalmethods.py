from tqdm.autonotebook import tqdm
from collections import deque
from heuristics import HEURISTIC, update_heuristics, random_literal
import random

def printinclause(clauses: list[list[int]], literal: int, extra: str) -> None:
    for clause in clauses:
        for lit in clause:
            if lit//10 == literal:
                print('In clause!', end=extra + '\n')
                return

def propagate(clauses: list[list[int]], literal: int) -> list[list[int]]:
    """Propagates a literal and simplifies the clauses."""
    new_clauses = []
    for clause in clauses:
        if literal in clause:
            continue  # Clause is satisfied, so skip it
        elif -literal in clause:
            new_clause = [l for l in clause if l != -literal]
        else:
            new_clause = clause
        # If new_clause is empty, we should add it as an empty clause to signal unsatisfiability.
        new_clauses.append(new_clause)
    return new_clauses

def single_literal(clauses: list[list[int]], assignment: dict[int, bool]) -> list[list[int]] | None:
    """Check for clauses that have only one literal in them, if so we can assign that literal to True and remove the clause.

    Args:
        clauses (list[list[int]]): List of clauses in dimacs format
        assignment (dict[int, bool]): Dictionary of assignments

    Returns:
        list[list[int]] | None: Simplified clauses
    """
    
    unit_clauses = [c[0] for c in clauses if len(c) == 1]
    queue = deque(unit_clauses)
    with tqdm(total=len(queue), desc="Doing single literal", leave=False) as pbar:
        while queue:
            literal = queue.popleft()
            assignment[abs(literal)] = literal > 0
            clauses = propagate(clauses, literal)
            extension = [c[0] for c in clauses if len(c) == 1 and c[0] not in queue]
            queue.extend(extension)
            pbar.update(1)
            pbar.total += len(extension)
        
    return clauses

def pure_literal(clauses: list[list[int]], assignment: dict[int, bool]) -> list[list[int]]:
    """Check for pure literals, meaning that if one literal is in the clauses but it's negative is never we can assign it.
    Turns out this isn't used often for sudoku and mainly slows down the solver.

    Args:
        clauses (list[list[int]]): Lists of clauses in dimacs format
        assignment (dict[int, bool]): Dictionary of assignments

    Returns:
        list[list[int]]: Simplified clauses
    """
    while True:
        literals = {lit for clause in clauses for lit in clause}
        pure_literals = [l for l in literals if -l not in literals]
        if not pure_literals:
            break
        for literal in pure_literals:
            assignment[abs(literal)] = literal > 0
            # Remove all clauses containing this pure literal
            clauses = [clause for clause in clauses if literal not in clause]
    return clauses


def choose_literal(
    clauses: list[list[int]], 
    assignment: dict[int, bool], 
    heuristics: dict[str, HEURISTIC], 
    current_heuristic: str, 
    conflict_counter: list[int], 
    history: list[int],
    restart_threshold: int
) -> int | None:
    """
    Chooses the next literal to assign based on the given heuristic.

    Args:
        clauses (list[list[int]]): List of clauses in DIMACS format.
        assignment (dict[int, bool]): Dictionary of current variable assignments.
        heuristics (dict[str, HEURISTIC]): Dictionary of heuristics to use.
        current_heuristic (str): Name of the current heuristic to use.
        conflict_counter (list[int]): Amount of conflicts.
        history (list[int]): Track variable history.

    Returns:
        int | None: The chosen literal, or None if no unassigned literals remain.
    """
    unassigned = {abs(lit) for clause in clauses for lit in clause if abs(lit) not in assignment}
    if not unassigned:
        return None
    
    if current_heuristic in ['vsids', 'chb']:
        literal = heuristics[current_heuristic](clauses, unassigned)
    elif current_heuristic == 'random':
        literal = heuristics[random.choice(list(heuristics))](clauses, unassigned)
    elif current_heuristic == 'rr':
        if heuristics['vsids'].lastactivated == 0:
            literal = heuristics['vsids'](clauses, unassigned)
            heuristics['vsids'].lastactivated = 1
            heuristics['chb'].lastactivated = 0
        else:
            literal = heuristics['chb'](clauses, unassigned)
            heuristics['chb'].lastactivated = 1
            heuristics['vsids'].lastactivated = 0
    elif current_heuristic == 'ranran':
        literal = random_literal(clauses, unassigned)
    elif current_heuristic == 'sss':
        if conflict_counter[0] >= restart_threshold // 2:
            literal = heuristics['chb'](clauses, unassigned)
        else:
            literal = heuristics['vsids'](clauses, unassigned)
    else:
        raise ValueError(f"Unknown heuristic: {current_heuristic}")
        
    
    if literal is None:  # Conflict
        update_heuristics(history[-1], conflict=True, heuristics=heuristics)
        conflict_counter[0] += 1
        return None
    
    return abs(literal)
