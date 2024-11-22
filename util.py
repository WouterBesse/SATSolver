import math

def load_rules(filename: str) -> list[list[int]]:
    with open(filename) as f:
        raw_rules = f.read().splitlines()

    rules = [list(map(int, clause.split()))[:-1] for clause in raw_rules[1:]]
    return rules

def load_sudoku(filename: str) -> tuple[list[list[list[int]]], int]:
    with open(filename) as f:
        sudoku = f.read().splitlines()

    puzzles: list[list[list[int]]] = []
    
    puzzledim: float = math.sqrt(len(sudoku[0]))
    assert puzzledim.is_integer(), "The sudoku is not square"
    puzzledim = int(puzzledim)
    print(f"puzzledim: {puzzledim}")

    for puzzle in sudoku:
        newpuzzle = []
        for r in range(puzzledim):
            row = []
            for c in range(puzzledim):
                val = puzzle[r * puzzledim + c]
                if val == '.': 
                    val = 0
                elif puzzledim == 16 and val.isalpha():
                    # encode 1 to g values to 1 to 16
                    val = int(ord(val) - 55)
                else:
                    val = int(val)
                row.append(val)
            newpuzzle.append(row)
        puzzles.append(newpuzzle)
    return puzzles, puzzledim

def load_sudoku_dimacs(filename: str, puzzledim: int = 9) -> tuple[list[int], list[list[int]], int]:
    with open(filename) as f:
        sudokuandrules = f.read().splitlines()
    
    sudokuandrules = sudokuandrules[1:]
    puzzledimacs: list[int] = []
    rules: list[list[int]] = []
    for i, line in enumerate(sudokuandrules):
        if line == '111 112 113 114 115 116 117 118 119  0':
            rules = [list(map(int, clause.split()))[:-1] for clause in sudokuandrules[i:]]
            break
        
        puzzledimacs.append(int(line[:-1]))
    
    puzzles: list[list[list[int]]] = []
    
    return puzzledimacs, rules, puzzledim
    
def decode_dimacs(var: int, rcmult: int) -> tuple[int, int, int]:
    num = var % rcmult
    j = (((var - 1) // rcmult) % rcmult) - 1
    i = ((var - 1 - (j + 1) * rcmult) // (rcmult**2)) - 1
    
    return num, i, j

def export_sudoku_dimacs(puzzledimacs: list[int], filename: str) -> None:
    with open(filename, 'w') as f:
        f.write(f"p cnf {len(puzzledimacs)} {len(puzzledimacs)}\n")
        for dimac in puzzledimacs:
            f.write(f"{dimac} 0\n")

def print_ass(assignment: dict) -> None:
    for key in assignment.keys():
        dimac = key if assignment[key] else -key
        print(dimac, end=' ')
    print('')