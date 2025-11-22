
import random
import time
import heapq
import math
import argparse
from collections import deque

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 0 stands for blank (B)

def pretty_print(state):
    s = []
    for i in range(9):
        v = state[i]
        s.append('B' if v == 0 else str(v))
    return f"{s[0]} {s[1]} {s[2]}\n{s[3]} {s[4]} {s[5]}\n{s[6]} {s[7]} {s[8]}"

def is_solvable(state):
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv % 2 == 0

def random_solvable_state():
    while True:
        lst = list(GOAL)
        random.shuffle(lst)
        tup = tuple(lst)
        if tup != GOAL and is_solvable(tup):
            return tup

def neighbors(state):
    i = state.index(0)
    row, col = divmod(i, 3)
    succs = []
    for drow, dcol in [(-1,0),(1,0),(0,-1),(0,1)]:
        r, c = row + drow, col + dcol
        if 0 <= r < 3 and 0 <= c < 3:
            j = r * 3 + c
            new = list(state)
            new[i], new[j] = new[j], new[i]
            succs.append(tuple(new))
    return succs

def h_misplaced(state):
    return sum(1 for i, v in enumerate(state) if v != 0 and v != GOAL[i])

def h_manhattan(state):
    dist = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        goal_idx = v - 1
        cur_row, cur_col = divmod(i, 3)
        goal_row, goal_col = divmod(goal_idx, 3)
        dist += abs(cur_row - goal_row) + abs(cur_col - goal_col)
    return dist

# ---------- A* implementation ----------
def a_star(start, heuristic_fn, max_nodes=500000):

    t0 = time.time()
    open_heap = []  # entries: (f, g, counter, state)
    counter = 0
    g_score = {start: 0}
    h0 = heuristic_fn(start)
    heapq.heappush(open_heap, (h0, 0, counter, start))
    came_from = {}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        f, g, _, current = heapq.heappop(open_heap)
        if g_score.get(current, math.inf) != g:
            continue
        nodes_expanded += 1

        if current == GOAL:
            path = deque()
            s = current
            while s in came_from:
                path.appendleft(s)
                s = came_from[s]
            path.appendleft(s)
            return {
                "solution": list(path),
                "cost": g_score[current],
                "nodes_expanded": nodes_expanded,
                "time": time.time() - t0
            }

        closed.add(current)

        for nei in neighbors(current):
            tentative_g = g_score[current] + 1  # uniform cost
            if nei in closed and tentative_g >= g_score.get(nei, math.inf):
                continue
            if tentative_g < g_score.get(nei, math.inf):
                came_from[nei] = current
                g_score[nei] = tentative_g
                h = heuristic_fn(nei)
                f_nei = tentative_g + h
                counter += 1
                heapq.heappush(open_heap, (f_nei, tentative_g, counter, nei))

        if nodes_expanded > max_nodes:
            return {
                "solution": None,
                "cost": None,
                "nodes_expanded": nodes_expanded,
                "time": time.time() - t0
            }

    return {"solution": None, "cost": None, "nodes_expanded": nodes_expanded, "time": time.time() - t0}

def run_once(seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
    start_state = random_solvable_state()
    if verbose:
        print("Start state:\n", pretty_print(start_state))
        print("\nGoal state:\n", pretty_print(GOAL))

    if verbose:
        print("\nRunning A* with h1 (misplaced tiles)...")
    res1 = a_star(start_state, h_misplaced)
    if res1["solution"] is None:
        print("h1: No solution found within node limit. Nodes expanded:", res1["nodes_expanded"])
    else:
        if verbose:
            print(f"h1 (misplaced) solved: cost={res1['cost']}, nodes_expanded={res1['nodes_expanded']}, time={res1['time']:.4f}s")
            print("Solution steps:")
            for idx, s in enumerate(res1["solution"]):
                print(f"\nStep {idx}:\n{pretty_print(s)}")

    if verbose:
        print("\nRunning A* with h2 (Manhattan distance)...")
    res2 = a_star(start_state, h_manhattan)
    if res2["solution"] is None:
        print("h2: No solution found within node limit. Nodes expanded:", res2["nodes_expanded"])
    else:
        if verbose:
            print(f"h2 (Manhattan) solved: cost={res2['cost']}, nodes_expanded={res2['nodes_expanded']}, time={res2['time']:.4f}s")
            print("Solution steps:")
            for idx, s in enumerate(res2["solution"]):
                print(f"\nStep {idx}:\n{pretty_print(s)}")

    return start_state, res1, res2

def run_trials(trials=30):
    stats = {"h1": [], "h2": []}
    for i in range(trials):
        _, r1, r2 = run_once(seed=None, verbose=False)
        stats["h1"].append(r1)
        stats["h2"].append(r2)

    def summarize(list_of_results):
        solved = [r for r in list_of_results if r["solution"] is not None]
        success_rate = len(solved) / len(list_of_results)
        avg_nodes = sum(r["nodes_expanded"] for r in list_of_results) / len(list_of_results)
        avg_time = sum(r["time"] for r in list_of_results) / len(list_of_results)
        avg_cost = (sum(r["cost"] for r in solved) / len(solved)) if solved else None
        return {"success_rate": success_rate, "avg_nodes": avg_nodes, "avg_time": avg_time, "avg_cost": avg_cost}

    return {"h1": summarize(stats["h1"]), "h2": summarize(stats["h2"])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A* search for 8-puzzle with two heuristics.")
    parser.add_argument("--trials", type=int, default=0, help="Run multiple random starts and summarize (default: 0, run single)")
    parser.add_argument("--max_nodes", type=int, default=500000, help="Max nodes expanded per A* (default 500000)")
    args = parser.parse_args()

  
    if args.trials <= 0:
        run_once()
    else:
        print(f"Running {args.trials} random solvable starts (no verbose per-run)...")
        summary = run_trials(trials=args.trials)
        print("\n=== Trials Summary ===")
        for hname, s in summary.items():
            print(f"{hname}: success_rate={s['success_rate']*100:.1f}%, avg_nodes={s['avg_nodes']:.1f}, avg_time={s['avg_time']:.4f}s, avg_cost={s['avg_cost']}")
