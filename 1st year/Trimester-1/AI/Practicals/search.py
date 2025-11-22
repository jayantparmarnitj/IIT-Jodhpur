
import numpy as np
from collections import deque

class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 0]]
        )
        self.max_row = 3
        self.max_col = 3
    
    def get_possible_moves(self, state):
        moves = []
        row, col = np.where(state==0)
        row, col = int(row[0]), int(col[0])
        directions = {
            "Up" : (-1, 0), # ("Up", (-1, 0))
            "Down" : (1, 0),
            "Left" : (0, -1),
            "Right" : (0, 1)
        }
        for move, (dr, dc) in directions.items():
            new_row, new_col = row+dr, col+dc
            if 0 <= new_row < self.max_row and 0 <= new_col < self.max_col:
                new_state = state.copy()
                new_state[row, col], new_state[new_row, new_col] = new_state[new_row, new_col], new_state[row, col]
                moves.append((new_state, move))
        return moves

    def bfs(self):
        frontier = deque([(self.initial_state, [])])
        explored = set()
        while frontier:
            current_state, path = frontier.popleft()
            if np.array_equal(current_state, self.goal_state):
                return path
            
            state_tuple = tuple(current_state.flatten())
            if state_tuple not in explored:
                explored.add(state_tuple)
                for next_state, move in self.get_possible_moves(current_state):
                    next_state_tuple = tuple(next_state.flatten())
                    if next_state_tuple not in explored:
                        frontier.append((next_state, path+[move]))
        return None
    

# initial_state = [
#     [1, 2, 3],
#     [4, 8, 5],
#     [7, 6, 0]
# ]

# initial_state = [
#     [1, 3, 8],
#     [4, 2, 5],
#     [7, 6, 0]
# ]

initial_state = [
    [1, 3, 8],
    [4, 2, 5],
    [7, 6, 0]
]

puzzle = EightPuzzle(np.array(initial_state))
solution = puzzle.bfs()

assert solution is not None
print(solution)
print(f"Initial state : \n{np.array(initial_state)}\n")

curr = (2, 2)
init_state = np.array(initial_state)
for idx, step in enumerate(solution):
    if step == 'Left':
        init_state[curr[0], curr[1]], init_state[curr[0], curr[1]-1] = init_state[curr[0], curr[1]-1], init_state[curr[0], curr[1]]
        curr = (curr[0], curr[1]-1)
    elif step == 'Right' :
        init_state[curr[0], curr[1]], init_state[curr[0], curr[1]+1] = init_state[curr[0], curr[1]+1], init_state[curr[0], curr[1]]
        curr = (curr[0], curr[1]+1)
    elif step == 'Up':
        init_state[curr[0], curr[1]], init_state[curr[0]-1, curr[1]] = init_state[curr[0]-1, curr[1]], init_state[curr[0], curr[1]]
        curr = (curr[0]-1, curr[1])
    elif step == 'Down':
        init_state[curr[0], curr[1]], init_state[curr[0]+1, curr[1]] = init_state[curr[0]+1, curr[1]], init_state[curr[0], curr[1]]
        curr = (curr[0]+1, curr[1])
    print(f"Step {idx+1} : {step}\n{init_state}\n")