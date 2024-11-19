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
        new_clause = [l for l in clause if l != -literal]
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
    changed = True
    while changed:
        changed = False
        unit_clauses = [c for c in clauses if len(c) == 1]
        for unit in unit_clauses:
            literal = unit[0]
            assignment[abs(literal)] = literal > 0
            clauses = propagate(clauses, literal)
            changed = True
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


def choose_literal(clauses: list[list[int]], assignment: dict[int, bool], heuristic) -> int | None:
    """
    Chooses the next literal to assign based on the given heuristic.

    Args:
        clauses (list[list[int]]): List of clauses in DIMACS format.
        assignment (dict[int, bool]): Dictionary of current variable assignments.
        heuristic (function): Function to apply the heuristic logic.

    Returns:
        int | None: The chosen literal, or None if no unassigned literals remain.
    """
    unassigned = {abs(lit) for clause in clauses for lit in clause if abs(lit) not in assignment}
    if not unassigned:
        return None
    return heuristic(clauses, unassigned)
