import math

def load_rules(filename: str) -> list[list[int]]:
    with open(filename) as f:
        raw_rules = f.read().splitlines()

    rules = [list(map(int, clause.split()))[:-1] for clause in raw_rules[1:]]
    return rules

def load_sudoku(filename: str) -> tuple[list[list[list[int | str]]], int]:
    with open(filename) as f:
        sudoku = f.read().splitlines()

    puzzles: list[list[list[int | str]]] = []
    
    puzzledim: int = math.sqrt(len(sudoku[0]))
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