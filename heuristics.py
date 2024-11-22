# Most frequent literal
import random

def random_literal(clauses: list[list[int]], unassigned: set[int]) -> int:
    return random.choice(list(unassigned))



class HEURISTIC:
    def __init__(self, clauses: list[list[int]]) -> None:
        self.activities: dict[int, float] = {}
        self.lastactivated = 0
        pass
    
    def __call__(self, clauses: list[list[int]], unassigned: set[int]) -> int | None:
        pass
    
    def update(self, literal: int, conflict: bool) -> None:
        pass
    
    def __str__(self) -> str:
        # return highest 5 literals as a string
        activitiesiterable = self.activities.items()
        sortedactivities = sorted(activitiesiterable, key = lambda item: item[-1], reverse=True)[:5]
        returnstring = ''
        for item in sortedactivities:
            returnstring += f'{item[0]}: {item[1]:.2f}, '
        return str()

class VSIDS(HEURISTIC):
    def __init__(self, clauses: list[list[int]], alpha: float = 0.95) -> None:
        self.activities: dict[int, float] = {}
        self.literals = set()
        self.alpha = alpha
        self.decay_factor = 1.0
        self.lastactivated = 0

        # Initialize the counts
        for clause in clauses:
            for lit in clause:
                if lit not in self.activities:
                    self.activities[lit] = 0
                    self.literals.add(lit)

    def __call__(self, clauses: list[list[int]], unassigned: set[int]) -> int | None:
        # Check the available literals still in the clauses
        available = {lit for clause in clauses for lit in clause}
        
        # Choose the literal with the highest activity
        max_activity = -1
        best_literal = None
        for lit in available:
            if self.activities[lit] > max_activity:
                max_activity = self.activities[lit]
                best_literal = lit
        
        return best_literal

    def update(self, literal: int, conflict: bool) -> None:
        # Increase the activity of the literal
        self.activities[literal] += 1
        
        self.decay()

    def decay(self) -> None:
        # Decay all activities, not used currently as it seemed to slow down the program
        self.decay_factor *= self.alpha
        for lit in self.activities:
            self.activities[lit] *= self.alpha
            
            
class CHB(HEURISTIC):
    def __init__(self, clauses: list[list[int]], alpha: float = 0.4, alpha_decay: float = 1e-6, alpha_min: float = 0.06) -> None:
        self.activities: dict[int, float] = {}
        self.literals = set()
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.conflicts = 0
        self.last_conflict = {}
        self.lastactivated = 0

        # Initialize the scores and last conflict times
        for clause in clauses:
            for lit in clause:
                if lit not in self.activities:
                    self.activities[lit] = 0.0  # Initial activity
                    self.literals.add(lit)
                    self.last_conflict[lit] = -1  # Last conflict time

    def __call__(self, clauses: list[list[int]], unassigned: set[int]) -> int | None:
        # Choose the literal with the highest activity among unassigned literals
        max_activity = -1
        best_literal = None
        for lit in unassigned:
            if self.activities[lit] > max_activity:
                max_activity = self.activities[lit]
                best_literal = lit
        return best_literal

    def update(self, literal: int, conflict: bool) -> None:
        self.conflicts += 1  # Increment the global conflict counter

        # Calculate the reward
        multiplier = 1.0 if conflict else 0.9
        recency = self.conflicts - self.last_conflict[literal] + 1
        reward = multiplier / recency

        # Update activity using ERWA
        self.activities[literal] = (1 - self.alpha) * self.activities[literal] + self.alpha * reward

        # Update last conflict time for the literal
        self.last_conflict[literal] = self.conflicts

        # Decay alpha
        self.alpha = max(self.alpha - self.alpha_decay, self.alpha_min)
        
def update_heuristics(literal: int, conflict: bool, heuristics: dict[str, HEURISTIC]) -> None:
    for heuristic in heuristics.values():
        if isinstance(heuristic, (VSIDS, CHB)):
            heuristic.update(literal, conflict=conflict)