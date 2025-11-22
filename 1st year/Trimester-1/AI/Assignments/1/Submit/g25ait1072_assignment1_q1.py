import random
import math
import time
from collections import Counter
import statistics

def print_board_from_state(state, q_char='q', empty_char='.', top_to_bottom=True):
    n = len(state)
    rows = range(n) if top_to_bottom else reversed(range(n))
    for r in rows:
        row_elems = []
        for c in range(n):
            row_elems.append(q_char if state[c] == r else empty_char)
        print(" ".join(row_elems))


def cost(state):
    n = len(state)
    conflicts = 0
    diag1, diag2 = Counter(), Counter()
    for c, r in enumerate(state):
        diag1[r - c] += 1
        diag2[r + c] += 1
    for cnt in diag1.values():
        if cnt > 1:
            conflicts += cnt * (cnt - 1) // 2
    for cnt in diag2.values():
        if cnt > 1:
            conflicts += cnt * (cnt - 1) // 2
    return conflicts

def random_permutation(n, rng):
    s = list(range(n))
    rng.shuffle(s)
    return s

def neighbor_by_swap(state, rng):
    n = len(state)
    i, j = rng.sample(range(n), 2)
    new = state.copy()
    new[i], new[j] = new[j], new[i]
    return new, i, j


def simulated_annealing(n, max_iters=200000, T0=1.0, alpha=0.9999, initial_state=None, rng=None):

    if rng is None:
        rng = random.Random()
    if initial_state is None:
        state = random_permutation(n, rng)
    else:
        state = initial_state.copy()
    current_cost = cost(state)
    T = T0
    iters = 0
    start = time.time()

    while iters < max_iters and current_cost > 0:
        new_state, _, _ = neighbor_by_swap(state, rng)
        new_cost = cost(new_state)
        delta = new_cost - current_cost

        # Acceptance rule
        if delta <= 0:
            accept = True
        else:
            # If temperature T is zero, this becomes False (no worse moves accepted)
            accept = (rng.random() < math.exp(-delta / T)) if T > 0 else False

        if accept:
            state, current_cost = new_state, new_cost

        T *= alpha
        iters += 1

    elapsed = time.time() - start
    return {"state": state, "cost": current_cost, "iters": iters, "time": elapsed}


def hill_climbing_from_sa(n, max_iters=200000, initial_state=None, rng=None):

    if rng is None:
        rng = random.Random()
    if initial_state is None:
        state = random_permutation(n, rng)
    else:
        state = initial_state.copy()
    current_cost = cost(state)
    iters = 0
    start = time.time()

    while iters < max_iters and current_cost > 0:
        new_state, _, _ = neighbor_by_swap(state, rng)
        new_cost = cost(new_state)
        delta = new_cost - current_cost
        if delta <= 0:
            state, current_cost = new_state, new_cost
        iters += 1

    elapsed = time.time() - start
    return {"state": state, "cost": current_cost, "iters": iters, "time": elapsed}


def summarize_results(results):
    costs = [r["cost"] for r in results]
    iters = [r["iters"] for r in results]
    times = [r["time"] for r in results]
    successes = [r for r in results if r["cost"] == 0]
    success_rate = len(successes) / len(results) if results else 0.0
    success_count = len(successes)
    avg_iters = sum(iters) / len(iters) if iters else 0.0
    median_iters = int(statistics.median(iters)) if iters else 0
    avg_time = sum(times) / len(times) if times else 0.0
    example = next((r for r in results if r["cost"] == 0), results[0] if results else None)
    best_cost = min(costs) if costs else None
    worst_cost = max(costs) if costs else None
    mean_cost = statistics.mean(costs) if costs else None
    stdev_cost = statistics.pstdev(costs) if len(costs) > 1 else 0.0
    return {
        "success_rate": success_rate,
        "success_count": success_count,
        "avg_iters": avg_iters,
        "median_iters": median_iters,
        "avg_time": avg_time,
        "example": example,
        "best_cost": best_cost,
        "worst_cost": worst_cost,
        "mean_cost": mean_cost,
        "stdev_cost": stdev_cost
    }


def collect_unique(results):
    seen = {}
    for idx, r in enumerate(results):
        tup = tuple(r["state"])
        if tup not in seen:
            seen[tup] = {"first_trial": idx, "cost": r["cost"], "iters": r["iters"], "time": r["time"]}
    return seen

if __name__ == "__main__":
    N = 20
    TRIALS = 20
    MAX_ITERS = 200000
    SA_T0 = 0.5         # standard SA temperature
    SA_ALPHA = 0.9995
    SEED = 2025         # global seed for reproducibility

    # Create RNGs:
    master_rng = random.Random(SEED)

    initial_state = random_permutation(N, master_rng)

    print("=== INITIAL SETUP ===")
    print(f"N = {N}, TRIALS = {TRIALS}, MAX_ITERS = {MAX_ITERS}")
    print("Initial (shared) state:", initial_state)
    print()

    print(" EQUIVALENCE DEMO: SINGLE RUNS STARTING FROM SAME INITIAL STATE ")
    rng_for_sa_as_hc = random.Random(1000)   # arbitrary seed
    rng_for_hc = random.Random(1000)        # same seed to synchronize neighbor draws

    # Standard SA (allows uphill moves)
    rng_sa_standard = random.Random(2000)
    sa_standard = simulated_annealing(
        N, max_iters=MAX_ITERS, T0=SA_T0, alpha=SA_ALPHA,
        initial_state=initial_state, rng=rng_sa_standard
    )

    # SA with T0 = 0 (should behave as HC)
    sa_as_hc = simulated_annealing(
        N, max_iters=MAX_ITERS, T0=0.0, alpha=SA_ALPHA,
        initial_state=initial_state, rng=rng_for_sa_as_hc
    )

    # Explicit Hill Climbing, with RNG seeded same as sa_as_hc to use identical neighbor sequence
    hc = hill_climbing_from_sa(N, max_iters=MAX_ITERS, initial_state=initial_state, rng=rng_for_hc)

    def fmt_res(name, res):
        return f"{name}: cost={res['cost']}, iters={res['iters']}, time={res['time']:.6f}, state={res['state']}"

    print("Initial state (shared):", initial_state)
    print(fmt_res("SA (standard, T0=0.5)", sa_standard))
    print(fmt_res("SA-as-HC (T0=0.0)", sa_as_hc))
    print(fmt_res("HillClimb (explicit)", hc))
    same_state = tuple(sa_as_hc["state"]) == tuple(hc["state"])
    print("SA-as-HC matches HillClimb exactly?:", same_state)
    print()

    print("=== MULTI-TRIAL COMPARISON (machine-friendly metrics) ===")
    sa_results = []
    hc_results = []
    for t in range(TRIALS):
        seed_sa = master_rng.randint(1, 10**9)
        seed_hc = master_rng.randint(1, 10**9)
        rng_sa = random.Random(seed_sa)
        rng_hc = random.Random(seed_hc)

        sa_res = simulated_annealing(N, max_iters=MAX_ITERS, T0=SA_T0, alpha=SA_ALPHA, initial_state=None, rng=rng_sa)
        hc_res = hill_climbing_from_sa(N, max_iters=MAX_ITERS, initial_state=None, rng=rng_hc)

        sa_results.append(sa_res)
        hc_results.append(hc_res)

    sa_summary = summarize_results(sa_results)
    hc_summary = summarize_results(hc_results)

    # Machine-friendly metrics
    print(" SIMULATED ANNEALING METRICS ")
    print(f"*sa_success_rate {sa_summary['success_rate']:.6f}")
    print(f"*sa_success_count {sa_summary['success_count']}")
    print(f"*sa_avg_iters {int(sa_summary['avg_iters'])}")
    print(f"*sa_median_iters {sa_summary['median_iters']}")
    print(f"*sa_avg_time_sec {sa_summary['avg_time']:.6f}")
    print(f"*sa_best_cost {sa_summary['best_cost']}")
    print(f"*sa_worst_cost {sa_summary['worst_cost']}")
    print(f"*sa_mean_cost {sa_summary['mean_cost'] if sa_summary['mean_cost'] is not None else 'nan'}")
    print(f"*sa_stdev_cost {sa_summary['stdev_cost']:.6f}")

    ex = sa_summary["example"]
    if ex is not None:
        print(f"*sa_example_cost {ex['cost']}")
        print(f"*sa_example_iters {ex['iters']}")
        print(f"*sa_example_time_sec {ex['time']:.6f}")
        print(f"*sa_example_state {','.join(str(x) for x in ex['state'])}")
    else:
        print("*sa_example None")

    print()

    print(" HILL CLIMBING METRICS ")
    print(f"*hc_success_rate {hc_summary['success_rate']:.6f}")
    print(f"*hc_success_count {hc_summary['success_count']}")
    print(f"*hc_avg_iters {int(hc_summary['avg_iters'])}")
    print(f"*hc_median_iters {hc_summary['median_iters']}")
    print(f"*hc_avg_time_sec {hc_summary['avg_time']:.6f}")
    print(f"*hc_best_cost {hc_summary['best_cost']}")
    print(f"*hc_worst_cost {hc_summary['worst_cost']}")
    print(f"*hc_mean_cost {hc_summary['mean_cost'] if hc_summary['mean_cost'] is not None else 'nan'}")
    print(f"*hc_stdev_cost {hc_summary['stdev_cost']:.6f}")

    ex2 = hc_summary["example"]
    if ex2 is not None:
        print(f"*hc_example_cost {ex2['cost']}")
        print(f"*hc_example_iters {ex2['iters']}")
        print(f"*hc_example_time_sec {ex2['time']:.6f}")
        print(f"*hc_example_state {','.join(str(x) for x in ex2['state'])}")
    else:
        print("*hc_example None")

    print("\n")  

    sa_success_disp = f"{sa_summary['success_rate']*100:.1f}% ({sa_summary['success_count']}/{TRIALS})"
    print("SIMULATED ANNEALING SUMMARY:")
    print(f"Success rate: {sa_success_disp}")
    print(f"Avg iterations: {int(sa_summary['avg_iters'])}, median iterations: {sa_summary['median_iters']}")
    print(f"Average time per trial: {sa_summary['avg_time']:.6f} s\n")

    hc_success_disp = f"{hc_summary['success_rate']*100:.1f}% ({hc_summary['success_count']}/{TRIALS})"
    print("HILL CLIMBING SUMMARY:")
    print(f"Success rate: {hc_success_disp}")
    print(f"Avg iterations: {int(hc_summary['avg_iters'])}, median iterations: {hc_summary['median_iters']}")
    print(f"Average time per trial: {hc_summary['avg_time']:.6f} s\n")

    print(" EXAMPLE BOARDS (unique solutions) ")
    PRINT_UNIQUE = True
    TOP_TO_BOTTOM = True

    print("SIMULATED ANNEALING example solution board:")
    if ex is not None:
        print_board_from_state(ex['state'], top_to_bottom=TOP_TO_BOTTOM)
    else:
        print("No example solution.")

    print("\nHILL CLIMBING example solution board:")
    if ex2 is not None:
        print_board_from_state(ex2['state'], top_to_bottom=TOP_TO_BOTTOM)
    else:
        print("No example solution.")

    print("\nDemo complete. You can adjust N, TRIALS, seeds, T0, alpha, and MAX_ITERS as needed.")
