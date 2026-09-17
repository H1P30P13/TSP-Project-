
 
# # AI Programming Group Project — Clean Submission Notebook
# 
# This version contains one consistent implementation of the TSP classes and algorithms.
 
# ## Core implementation
# 
# The distance matrix is pre-calculated once. Insert and Swap neighbourhood moves update the objective using only the affected edges.
 
import itertools
import math
import random
import time
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
 


def display(x):
    print(x)
 
EPS = 1e-12
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
class Instance:
    # Has precalculated distance matrix

    def __init__(self, n):
        if n < 4:
            raise ValueError("A TSP instance must contain at least four cities.")
 
        self.coordinates = [
            (random.random(), random.random())
            for _ in range(n)
        ]
 
        self.distances = [
            [0.0 for _ in range(n)]
            for _ in range(n)
        ]
 
        for i in range(n):
            for j in range(i + 1, n):
                x_i, y_i = self.coordinates[i]
                x_j, y_j = self.coordinates[j]
                value = math.hypot(x_i - x_j, y_i - y_j)
                self.distances[i][j] = value
                self.distances[j][i] = value
 
    def distance(self, cityA, cityB):
        return self.distances[cityA][cityB]
 
    def get_n(self):
        return len(self.coordinates)
 
    def get_coordinates(self, city):
        return self.coordinates[city]
 
 
class Solution:
    #Stored objective value
 
    def __init__(self, instance, sequence):
        self.instance = instance
        self.sequence = list(sequence)
        self.check_feasibility()
        self.objective = self.calculate_full_objective()
 
    def calculate_full_objective(self):
        sequence = self.sequence
        distance = self.instance.distance
 
        total = distance(sequence[-1], sequence[0])
        for index in range(len(sequence) - 1):
            total += distance(sequence[index], sequence[index + 1])
        return total
 
    def calculate_objective(self):
        return self.objective
 
    def clone(self):
        cloned = Solution(self.instance, self.sequence)
        cloned.objective = self.objective
        return cloned
 
    def check_feasibility(self):
        n = self.instance.get_n()
        if len(self.sequence) != n:
            raise ValueError(
                f"Expected {n} cities, but the sequence contains {len(self.sequence)} entries."
            )
        if set(self.sequence) != set(range(n)):
            raise ValueError(
                "The sequence must contain every city exactly once."
            )
        return True
 
    def resynchronise_objective(self):
        """Remove any accumulated floating-point error."""
        self.objective = self.calculate_full_objective()
        return self.objective
 
    def __str__(self):
        route = " -> ".join(str(city) for city in self.sequence)
        return f"{route}; objective {self.objective:.6f}"
 
 
def random_initialisation(instance):
    sequence = list(range(instance.get_n()))
    random.shuffle(sequence)
    return Solution(instance, sequence)
 
 
def plot_solution(solution, title=None):
    solution.check_feasibility()
    closed_tour = solution.sequence + solution.sequence[:1]
    coordinates = [
        solution.instance.get_coordinates(city)
        for city in closed_tour
    ]
    x_values, y_values = zip(*coordinates)
 
    plt.plot(x_values, y_values, marker="o")
    plt.title(
        title
        or f"Objective value = {solution.calculate_objective():.4f}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
 
 
def insert_move_delta(solution, remove_index, insert_index):
    """
    Return the objective change caused by:
        city = sequence.pop(remove_index)
        sequence.insert(insert_index, city)
 
    insert_index is interpreted in the reduced sequence after removal. The full tour objective is not recalculated.
    """
    sequence = solution.sequence
    n = len(sequence)
 
    if not 0 <= remove_index < n:
        raise IndexError("remove_index is outside the tour.")
    if not 0 <= insert_index < n:
        raise IndexError("insert_index is outside the valid insertion range.")
 
    if remove_index == insert_index:
        return 0.0
 
    distance = solution.instance.distance
    moved_city = sequence[remove_index]
    previous_city = sequence[(remove_index - 1) % n]
    next_city = sequence[(remove_index + 1) % n]
 
    def reduced_city(index):
        if index < remove_index:
            return sequence[index]
        return sequence[index + 1]
 
    reduced_n = n - 1
 
    if insert_index == 0 or insert_index == reduced_n:
        insert_previous = reduced_city(reduced_n - 1)
        insert_next = reduced_city(0)
    else:
        insert_previous = reduced_city(insert_index - 1)
        insert_next = reduced_city(insert_index)
 
    removed_cost = (
        distance(previous_city, moved_city)
        + distance(moved_city, next_city)
        + distance(insert_previous, insert_next)
    )
    added_cost = (
        distance(previous_city, next_city)
        + distance(insert_previous, moved_city)
        + distance(moved_city, insert_next)
    )
    return added_cost - removed_cost
 
 
def apply_insert_move(solution, remove_index, insert_index, change=None):
    if change is None:
        change = insert_move_delta(
            solution, remove_index, insert_index
        )
 
    city = solution.sequence.pop(remove_index)
    solution.sequence.insert(insert_index, city)
    solution.objective += change
 
 
def swap_move_delta(solution, index_a, index_b):
    
    # Only  affected edges are calculated. not incliude cyclic pairs 
    
    sequence = solution.sequence
    n = len(sequence)
 
    if not 0 <= index_a < n or not 0 <= index_b < n:
        raise IndexError("Swap index is outside the tour.")
    if index_a == index_b:
        return 0.0
 
    i, j = sorted((index_a, index_b))
    distance = solution.instance.distance
 
    city_i = sequence[i]
    city_j = sequence[j]
    previous_i = sequence[(i - 1) % n]
    next_i = sequence[(i + 1) % n]
    previous_j = sequence[(j - 1) % n]
    next_j = sequence[(j + 1) % n]
 
    if j == i + 1:
        removed_cost = (
            distance(previous_i, city_i)
            + distance(city_i, city_j)
            + distance(city_j, next_j)
        )
        added_cost = (
            distance(previous_i, city_j)
            + distance(city_j, city_i)
            + distance(city_i, next_j)
        )
    elif i == 0 and j == n - 1:
        # Cyclic adjacency
        removed_cost = (
            distance(previous_j, city_j)
            + distance(city_j, city_i)
            + distance(city_i, next_i)
        )
        added_cost = (
            distance(previous_j, city_i)
            + distance(city_i, city_j)
            + distance(city_j, next_i)
        )
    else:
        removed_cost = (
            distance(previous_i, city_i)
            + distance(city_i, next_i)
            + distance(previous_j, city_j)
            + distance(city_j, next_j)
        )
        added_cost = (
            distance(previous_i, city_j)
            + distance(city_j, next_i)
            + distance(previous_j, city_i)
            + distance(city_i, next_j)
        )
 
    return added_cost - removed_cost
 
 
def apply_swap_move(solution, index_a, index_b, change=None):
    if change is None:
        change = swap_move_delta(solution, index_a, index_b)
 
    solution.sequence[index_a], solution.sequence[index_b] = (
        solution.sequence[index_b],
        solution.sequence[index_a],
    )
    solution.objective += change
 
 
# First-improvement and greedy/best-improvement local moves
 
def first_improving_insert(solution, deadline=None):
    n = len(solution.sequence)
    checks = 0
 
    for remove_index in range(n):
        for insert_index in range(n):
            if remove_index == insert_index:
                continue
 
            checks += 1
            if (
                deadline is not None
                and checks % 128 == 0
                and time.perf_counter() >= deadline
            ):
                return False, True
 
            change = insert_move_delta(
                solution, remove_index, insert_index
            )
            if change < -EPS:
                apply_insert_move(
                    solution,
                    remove_index,
                    insert_index,
                    change,
                )
                return True, False
 
    return False, False
 
 
def first_improving_swap(solution, deadline=None):
    n = len(solution.sequence)
    checks = 0
 
    for i, j in itertools.combinations(range(n), 2):
        checks += 1
        if (
            deadline is not None
            and checks % 128 == 0
            and time.perf_counter() >= deadline
        ):
            return False, True
 
        change = swap_move_delta(solution, i, j)
        if change < -EPS:
            apply_swap_move(solution, i, j, change)
            return True, False
 
    return False, False
 
 
def insert_iterative_improvement(solution):
    # GII-Insert: first improving insertion move.
    improved, _ = first_improving_insert(solution)
    return improved
 
 
def swap_iterative_improvement(solution):
    # GII-Swap: first improving swap move.
    improved, _ = first_improving_swap(solution)
    return improved
 
 
def greedy_insert_iterative_improvement(solution):
    # best improvng insertion move 
    n = len(solution.sequence)
    best_change = 0.0
    bestMove = None
 
    for remove_index in range(n):
        for insert_index in range(n):
            if remove_index == insert_index:
                continue
 
            change = insert_move_delta(
                solution, remove_index, insert_index
            )
            if change < best_change - EPS:
                best_change = change
                bestMove = (remove_index, insert_index)
 
    if bestMove is None:
        return False
 
    apply_insert_move(
        solution,
        bestMove[0],
        bestMove[1],
        best_change,
    )
    return True
 
 
def greedy_swap_iterative_improvement(solution):
    # best improving swap move 
    n = len(solution.sequence)
    best_change = 0.0
    best_pair = None
 
    for i, j in itertools.combinations(range(n), 2):
        change = swap_move_delta(solution, i, j)
        if change < best_change - EPS:
            best_change = change
            best_pair = (i, j)
 
    if best_pair is None:
        return False
 
    apply_swap_move(
        solution,
        best_pair[0],
        best_pair[1],
        best_change,
    )
    return True
 
 
def run_single_neighbourhood_to_local_optimum(
    solution,
    improvement_operator,
):
    moves = 0
    while improvement_operator(solution):
        moves += 1
    solution.resynchronise_objective()
    solution.check_feasibility()
    return moves
 
 

# Initialisation heuristics
 
def nearest_neighbour_heuristic(instance, start_city):
    n = instance.get_n()
    if not 0 <= start_city < n:
        raise ValueError("start_city is outside the valid range.")
 
    sequence = [start_city]
    unvisited = set(range(n))
    unvisited.remove(start_city)
    current_city = start_city
 
    while unvisited:
        next_city = min(
            unvisited,
            key=lambda city: instance.distance(
                current_city, city
            ),
        )
        sequence.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
 
    return Solution(instance, sequence)
 
 
def best_nearest_neighbour_heuristic(instance):
    return min(
        (
            nearest_neighbour_heuristic(instance, start_city)
            for start_city in range(instance.get_n())
        ),
        key=lambda solution: solution.calculate_objective(),
    )
 
 
def edge_heuristic(instance, longest=False):

    n = instance.get_n()
    edges = [
        (instance.distance(i, j), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    edges.sort(key=lambda edge: edge[0], reverse=longest)
 
    parent = list(range(n))
    rank = [0] * n
    degree = [0] * n
    selected_edges = []
 
    def find(city):
        while parent[city] != city:
            parent[city] = parent[parent[city]]
            city = parent[city]
        return city
 
    def union(cityA, cityB):
        root_a = find(cityA)
        root_b = find(cityB)
 
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1
 
    for _, cityA, cityB in edges:
        if degree[cityA] >= 2 or degree[cityB] >= 2:
            continue
 
        same_component = find(cityA) == find(cityB)
 
        if same_component and len(selected_edges) < n - 1:
            continue
 
        selected_edges.append((cityA, cityB))
        degree[cityA] += 1
        degree[cityB] += 1
 
        if not same_component:
            union(cityA, cityB)
 
        if len(selected_edges) == n:
            break
 
    if (
        len(selected_edges) != n
        or any(city_degree != 2 for city_degree in degree)
    ):
        raise RuntimeError(
            "The edge heuristic failed to construct a complete cycle."
        )
 
    adjacency = [[] for _ in range(n)]
    for cityA, cityB in selected_edges:
        adjacency[cityA].append(cityB)
        adjacency[cityB].append(cityA)
 
    sequence = [0]
    previous_city = None
    current_city = 0
 
    for _ in range(n - 1):
        first, second = adjacency[current_city]
        next_city = (
            first if first != previous_city else second
        )
        sequence.append(next_city)
        previous_city, current_city = current_city, next_city
 
    result = Solution(instance, sequence)
    result.check_feasibility()
    return result
 
 
def shortest_edge_heuristic(instance):
    return edge_heuristic(instance, longest=False)
 
 
def longest_edge_heuristic(instance):
    return edge_heuristic(instance, longest=True)
 
 

# Perturbation and time-limited combined local search

 
def double_bridge_perturbation(solution):
    n = len(solution.sequence)
    if n < 4:
        return solution
 
    cut_1, cut_2, cut_3 = sorted(
        random.sample(range(1, n), 3)
    )
    segment_a = solution.sequence[:cut_1]
    segment_b = solution.sequence[cut_1:cut_2]
    segment_c = solution.sequence[cut_2:cut_3]
    segment_d = solution.sequence[cut_3:]
 
    solution.sequence = (
        segment_a + segment_c + segment_b + segment_d
    )
    solution.resynchronise_objective()
    return solution
 
 
def record_trace(trace_time, trace_value, start_time, solution):
    trace_time.append(time.perf_counter() - start_time)
    trace_value.append(solution.calculate_objective())
 
 
def combined_local_search(
    solution,
    deadline,
    start_time=None,
    trace_time=None,
    trace_current=None,
):
    
    # Repeatedly apply Insert and Swap first-improvement until neitherc neighbourhood improvs the tour or the deadline is reached.
    
    accepted_moves = 0
    reached_local_optimum = False
 
    while time.perf_counter() < deadline:
        improved_in_round = False
 
        while time.perf_counter() < deadline:
            improved, timed_out = first_improving_insert(
                solution, deadline
            )
            if timed_out:
                return accepted_moves, False
            if not improved:
                break
 
            accepted_moves += 1
            improved_in_round = True
            if trace_time is not None:
                record_trace(
                    trace_time,
                    trace_current,
                    start_time,
                    solution,
                )
 
        while time.perf_counter() < deadline:
            improved, timed_out = first_improving_swap(
                solution, deadline
            )
            if timed_out:
                return accepted_moves, False
            if not improved:
                break
 
            accepted_moves += 1
            improved_in_round = True
            if trace_time is not None:
                record_trace(
                    trace_time,
                    trace_current,
                    start_time,
                    solution,
                )
 
        if not improved_in_round:
            reached_local_optimum = True
            break
 
    return accepted_moves, reached_local_optimum
 
 
def iterative_improvement_search(initial_solution, time_limit):
    start_time = time.perf_counter()
    deadline = start_time + time_limit
    current = initial_solution.clone()
 
    trace_time = [0.0]
    trace_current = [current.calculate_objective()]
 
    accepted_moves, reached_local_optimum = combined_local_search(
        current,
        deadline,
        start_time,
        trace_time,
        trace_current,
    )
 
    current.resynchronise_objective()
    record_trace(
        trace_time, trace_current, start_time, current
    )
    current.check_feasibility()
 
    runtime = time.perf_counter() - start_time
    return {
        "best_solution": current.clone(),
        "current_solution": current,
        "initial_objective":
            initial_solution.calculate_objective(),
        "best_objective": current.calculate_objective(),
        "final_current_objective":
            current.calculate_objective(),
        "iterations": accepted_moves,
        "runtime": runtime,
        "trace_time": trace_time,
        "trace_current": trace_current,
        "trace_best": list(trace_current),
        "reached_local_optimum": reached_local_optimum,
    }
 
 
def iterated_local_search(initial_solution, time_limit):
    start_time = time.perf_counter()
    deadline = start_time + time_limit
 
    current = initial_solution.clone()
    trace_time = [0.0]
    trace_current = [current.calculate_objective()]
    trace_best = [current.calculate_objective()]
 
    local_moves, _ = combined_local_search(
        current,
        deadline,
        start_time,
        trace_time,
        trace_current,
    )
    best = current.clone()
    trace_best = list(trace_current)
 
    perturbation_iterations = 0
 
    while time.perf_counter() < deadline:
        candidate = best.clone()
        double_bridge_perturbation(candidate)
 
        candidate_moves, _ = combined_local_search(
            candidate, deadline
        )
        local_moves += candidate_moves
 
        if (
            candidate.calculate_objective()
            < best.calculate_objective() - EPS
        ):
            best = candidate.clone()
            current = best.clone()
            record_trace(
                trace_time,
                trace_current,
                start_time,
                current,
            )
            trace_best.append(best.calculate_objective())
 
        perturbation_iterations += 1
 
    best.resynchronise_objective()
    best.check_feasibility()
    record_trace(trace_time, trace_current, start_time, best)
 
    if len(trace_best) < len(trace_time):
        trace_best.extend(
            [best.calculate_objective()]
            * (len(trace_time) - len(trace_best))
        )
 
    runtime = time.perf_counter() - start_time
    return {
        "best_solution": best,
        "current_solution": best.clone(),
        "initial_objective":
            initial_solution.calculate_objective(),
        "best_objective": best.calculate_objective(),
        "final_current_objective":
            best.calculate_objective(),
        "iterations": perturbation_iterations,
        "local_moves": local_moves,
        "runtime": runtime,
        "trace_time": trace_time,
        "trace_current": trace_current,
        "trace_best": trace_best,
    }
 
 
# Threshold Acceptance
 
def random_insert_move(solution):
    n = len(solution.sequence)
    while True:
        remove_index = random.randrange(n)
        insert_index = random.randrange(n)
        if remove_index != insert_index:
            break
 
    change = insert_move_delta(
        solution, remove_index, insert_index
    )
    return ("insert", remove_index, insert_index, change)
 
 
def random_swap_move(solution):
    i, j = sorted(
        random.sample(range(len(solution.sequence)), 2)
    )
    change = swap_move_delta(solution, i, j)
    return ("swap", i, j, change)
 
 
def generate_random_move(solution, operator):
    if operator == "insert":
        return random_insert_move(solution)
    if operator == "swap":
        return random_swap_move(solution)
    if operator == "mixed":
        if random.random() < 0.5:
            return random_insert_move(solution)
        return random_swap_move(solution)
 
    raise ValueError(
        "operator must be 'insert', 'swap', or 'mixed'."
    )
 
 
def apply_random_move(solution, move):
    move_type, i, j, change = move
 
    if move_type == "insert":
        apply_insert_move(solution, i, j, change)
    elif move_type == "swap":
        apply_swap_move(solution, i, j, change)
    else:
        raise ValueError("Unknown move type.")
 
 
def threshold_acceptance(
    initial_solution,
    initial_delta,
    operator="mixed",
    time_limit=15.0,
    decrease_delta=True,
    trace_every=100,
):
    if initial_delta < 0:
        raise ValueError("initial_delta must be non-negative.")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive.")
    if trace_every < 1:
        raise ValueError("trace_every must be at least one.")
 
    start_time = time.perf_counter()
    current = initial_solution.clone()
    best = current.clone()
 
    iterations = 0
    accepted_moves = 0
    accepted_worse_moves = 0
 
    trace_time = [0.0]
    trace_current = [current.calculate_objective()]
    trace_best = [best.calculate_objective()]
 
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit:
            break
 
        if decrease_delta:
            remaining_fraction = max(
                0.0, 1.0 - elapsed / time_limit
            )
            current_delta = (
                initial_delta * remaining_fraction
            )
        else:
            current_delta = initial_delta
 
        move = generate_random_move(current, operator)
        deterioration = move[3]
 
        if deterioration <= current_delta:
            apply_random_move(current, move)
            accepted_moves += 1
 
            if deterioration > EPS:
                accepted_worse_moves += 1
 
            if (
                current.calculate_objective()
                < best.calculate_objective() - EPS
            ):
                best = current.clone()
 
        iterations += 1
 
        if iterations % trace_every == 0:
            trace_time.append(elapsed)
            trace_current.append(
                current.calculate_objective()
            )
            trace_best.append(best.calculate_objective())
 
    current.resynchronise_objective()
    best.resynchronise_objective()
    current.check_feasibility()
    best.check_feasibility()
 
    runtime = time.perf_counter() - start_time
    trace_time.append(runtime)
    trace_current.append(current.calculate_objective())
    trace_best.append(best.calculate_objective())
 
    return {
        "best_solution": best,
        "current_solution": current,
        "initial_objective":
            initial_solution.calculate_objective(),
        "best_objective": best.calculate_objective(),
        "final_current_objective":
            current.calculate_objective(),
        "operator": operator,
        "initial_delta": initial_delta,
        "iterations": iterations,
        "accepted_moves": accepted_moves,
        "accepted_worse_moves": accepted_worse_moves,
        "runtime": runtime,
        "trace_time": trace_time,
        "trace_current": trace_current,
        "trace_best": trace_best,
    }
 
 
def percentile(values, proportion):
    ordered = sorted(values)
    if not ordered:
        return 1e-6
 
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
 
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )
 
 
def make_delta_candidates(
    initial_solution,
    operator,
    samples=500,
):
    worsening_changes = []
 
    for _ in range(samples):
        change = generate_random_move(
            initial_solution, operator
        )[3]
        if change > EPS:
            worsening_changes.append(change)
 
    return [
        percentile(worsening_changes, 0.10),
        percentile(worsening_changes, 0.25),
        percentile(worsening_changes, 0.50),
    ]
 
 
# Reproducible benchmark instances used by all tasks.
random.seed(123)
PROBLEM_SIZES = [10, 20, 50, 100, 200]
benchmark_instances = [
    Instance(size) for size in PROBLEM_SIZES
]
instances_by_size = {
    instance.get_n(): instance
    for instance in benchmark_instances
}
 
print("Core TSP classes and algorithms loaded.")
 
# ## Validation checks
# 
# These checks compare the \(O(1)\) objective-change formulas with an exact full-objective recalculation and verify the construction heuristics produce feasible tours.
 
# Quick correctness checks for the O(1) move-delta formulas.
random.seed(20260805)
validation_instance = Instance(20)
validation_solution = random_initialisation(validation_instance)
 
maximum_delta_error = 0.0
 
for _ in range(200):
    i = random.randrange(20)
    j = random.randrange(20)
    if i == j:
        continue
 
    candidate = validation_solution.clone()
    predicted_change = insert_move_delta(candidate, i, j)
    old_objective = candidate.calculate_objective()
    apply_insert_move(candidate, i, j, predicted_change)
    exact_objective = candidate.calculate_full_objective()
 
    maximum_delta_error = max(
        maximum_delta_error,
        abs(
            exact_objective
            - (old_objective + predicted_change)
        ),
    )
    if not math.isclose(
        candidate.calculate_objective(),
        exact_objective,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise AssertionError("Insert delta validation failed.")
 
for _ in range(200):
    i, j = random.sample(range(20), 2)
 
    candidate = validation_solution.clone()
    predicted_change = swap_move_delta(candidate, i, j)
    old_objective = candidate.calculate_objective()
    apply_swap_move(candidate, i, j, predicted_change)
    exact_objective = candidate.calculate_full_objective()
 
    maximum_delta_error = max(
        maximum_delta_error,
        abs(
            exact_objective
            - (old_objective + predicted_change)
        ),
    )
    if not math.isclose(
        candidate.calculate_objective(),
        exact_objective,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise AssertionError("Swap delta validation failed.")
 
for constructor in (
    nearest_neighbour_heuristic,
    shortest_edge_heuristic,
    longest_edge_heuristic,
):
    if constructor is nearest_neighbour_heuristic:
        candidate = constructor(validation_instance, 0)
    else:
        candidate = constructor(validation_instance)
    candidate.check_feasibility()
 
print(f"Validation passed. Maximum objective-delta error: {maximum_delta_error:.3e}")
 
# # Task 1 — Tuesday 28 July
# 
# ## Task 1A — 20 random tours for each problem size
 
random.seed(1001)
task1a_rows = []
task1a_best_solutions = {}
 
for instance in benchmark_instances:
    n = instance.get_n()
    random_solutions = [
        random_initialisation(instance)
        for _ in range(20)
    ]
 
    best_solution = min(
        random_solutions,
        key=lambda solution:
            solution.calculate_objective(),
    )
    worst_solution = max(
        random_solutions,
        key=lambda solution:
            solution.calculate_objective(),
    )
    objective_values = [
        solution.calculate_objective()
        for solution in random_solutions
    ]
 
    task1a_best_solutions[n] = best_solution.clone()
    task1a_rows.append({
        "N": n,
        "Best": best_solution.calculate_objective(),
        "Worst": worst_solution.calculate_objective(),
        "Average": (
            sum(objective_values)
            / len(objective_values)
        ),
    })
 
    plt.figure(figsize=(5, 4))
    plot_solution(
        best_solution,
        f"Task 1A best random tour (N={n})",
    )
    plt.show()
 
    plt.figure(figsize=(5, 4))
    plot_solution(
        worst_solution,
        f"Task 1A worst random tour (N={n})",
    )
    plt.show()
 
task1a_df = pd.DataFrame(task1a_rows)
display(task1a_df.round(4))
 
# ### Task 1B — GII-Swap
# 
 
# ## Task 1C — Best Random vs GII-Insert vs GII-Swap
 
task1c_rows = []
task1c_solutions = {}
 
for instance in benchmark_instances:
    n = instance.get_n()
    best_random = task1a_best_solutions[n].clone()
 
    insert_solution = best_random.clone()
    insert_moves = run_single_neighbourhood_to_local_optimum(
        insert_solution,
        insert_iterative_improvement,
    )
 
    swap_solution = best_random.clone()
    swap_moves = run_single_neighbourhood_to_local_optimum(
        swap_solution,
        swap_iterative_improvement,
    )
 
    random_value = best_random.calculate_objective()
    insert_value = insert_solution.calculate_objective()
    swap_value = swap_solution.calculate_objective()
 
    task1c_rows.append({
        "N": n,
        "Best Random": random_value,
        "GII-Insert": insert_value,
        "GII-Swap": swap_value,
        "Random - Insert": random_value - insert_value,
        "Random - Swap": random_value - swap_value,
        "Insert - Swap": insert_value - swap_value,
        "Insert moves": insert_moves,
        "Swap moves": swap_moves,
    })
 
    task1c_solutions[n] = {
        "Random": best_random,
        "GII-Insert": insert_solution,
        "GII-Swap": swap_solution,
    }
 
task1c_df = pd.DataFrame(task1c_rows)
display(task1c_df.round(4))
 
# # Task 2 — Wednesday 29 July
# 
# ## Task 2A — Time-efficiency improvements
# 
# - Distances between every pair of cities are pre-calculated in `Instance.distances`.
# - Swap pairs use combinations, so `(i, j)` and `(j, i)` are not checked twice.
# - Insert and Swap moves calculate only the change in the affected edges instead of recalculating the whole tour.
# - The cached objective is updated by the calculated change.
 
# ## Task 2B — Greedy/Best-Improvement GII-Insert and GII-Swap
 
task2b_rows = []
task2b_solutions = {}
 
for instance in benchmark_instances:
    n = instance.get_n()
    starting_solution = task1a_best_solutions[n].clone()
 
    greedy_insert_solution = starting_solution.clone()
    greedy_insert_moves = (
        run_single_neighbourhood_to_local_optimum(
            greedy_insert_solution,
            greedy_insert_iterative_improvement,
        )
    )
 
    greedy_swap_solution = starting_solution.clone()
    greedy_swap_moves = (
        run_single_neighbourhood_to_local_optimum(
            greedy_swap_solution,
            greedy_swap_iterative_improvement,
        )
    )
 
    task2b_rows.append({
        "N": n,
        "Initial objective":
            starting_solution.calculate_objective(),
        "Greedy GII-Insert":
            greedy_insert_solution.calculate_objective(),
        "Greedy GII-Swap":
            greedy_swap_solution.calculate_objective(),
        "Greedy Insert moves":
            greedy_insert_moves,
        "Greedy Swap moves":
            greedy_swap_moves,
    })
 
    task2b_solutions[n] = {
        "Greedy GII-Insert": greedy_insert_solution,
        "Greedy GII-Swap": greedy_swap_solution,
    }
 
task2b_df = pd.DataFrame(task2b_rows)
display(task2b_df.round(4))
 
# ## Task 2C — NNH and SEH initial solutions
 
task2c_rows = []
task2c_solutions = {}
 
for instance in benchmark_instances:
    n = instance.get_n()
 
    nnh_solutions = [
        nearest_neighbour_heuristic(
            instance, start_city
        )
        for start_city in range(n)
    ]
    nnh_objectives = [
        solution.calculate_objective()
        for solution in nnh_solutions
    ]
 
    best_nnh = min(
        nnh_solutions,
        key=lambda solution:
            solution.calculate_objective(),
    )
    worst_nnh = max(
        nnh_solutions,
        key=lambda solution:
            solution.calculate_objective(),
    )
    seh_solution = shortest_edge_heuristic(instance)
 
    task2c_rows.append({
        "N": n,
        "NNH Best": best_nnh.calculate_objective(),
        "NNH Worst": worst_nnh.calculate_objective(),
        "NNH Average": (
            sum(nnh_objectives)
            / len(nnh_objectives)
        ),
        "SEH": seh_solution.calculate_objective(),
    })
 
    task2c_solutions[n] = {
        "Best NNH": best_nnh,
        "Worst NNH": worst_nnh,
        "SEH": seh_solution,
    }
 
task2c_df = pd.DataFrame(task2c_rows)
display(task2c_df.round(4))
 
# # Task 3 — Thursday 30 July
# 
# ## Task 3A — Longest Edge Heuristic
 
task3a_rows = []
task3a_solutions = {}
 
for instance in benchmark_instances:
    leh_solution = longest_edge_heuristic(instance)
    leh_solution.check_feasibility()
 
    task3a_rows.append({
        "N": instance.get_n(),
        "LEH objective":
            leh_solution.calculate_objective(),
    })
    task3a_solutions[
        instance.get_n()
    ] = leh_solution
 
task3a_df = pd.DataFrame(task3a_rows)
display(task3a_df.round(4))
 
# ## Task 3B — Double-Bridge perturbation
#  
# ## Task 3C — Iterated Local Search
# 
# Initialisation: SEH. Local search: repeated first-improvement Insert and Swap. Perturbation: Double-Bridge.
 
# Larger instances receive more time.
TASK3_TIME_LIMITS = {
    10: 3.0,
    20: 3.0,
    50: 5.0,
    100: 15.0,
    200: 15.0,
}
 
task3c_rows = []
task3c_results = {}
 
for instance in benchmark_instances:
    n = instance.get_n()
    initial_solution = shortest_edge_heuristic(instance)
 
    random.seed(30000 + n)
    result = iterated_local_search(
        initial_solution,
        TASK3_TIME_LIMITS[n],
    )
    result["best_solution"].check_feasibility()
 
    task3c_results[n] = result
    task3c_rows.append({
        "N": n,
        "Initial objective":
            initial_solution.calculate_objective(),
        "ILS objective": result["best_objective"],
        "Perturbation iterations":
            result["iterations"],
        "Local-search moves":
            result["local_moves"],
        "Runtime (s)": result["runtime"],
    })
 
task3c_df = pd.DataFrame(task3c_rows)
display(task3c_df.round(4))
 
# # Task 4 — Tuesday 4 August
# 
# ## Task 4A — Threshold Acceptance
 
# Tune operator and initial Delta on the existing N=200 instance.
instance_200 = instances_by_size[200]
 
random.seed(2026)
task4a_initial_solution = random_initialisation(
    instance_200
)
 
TASK4A_TUNING_SECONDS = 1.0
TASK4A_FINAL_SECONDS = 15.0
task4a_trials = []
 
for operator_number, operator in enumerate(
    ("insert", "swap", "mixed")
):
    random.seed(3000 + operator_number)
    delta_candidates = make_delta_candidates(
        task4a_initial_solution,
        operator,
        samples=500,
    )
 
    for delta_number, delta in enumerate(
        delta_candidates
    ):
        random.seed(
            4000
            + 100 * operator_number
            + delta_number
        )
        trial_result = threshold_acceptance(
            initial_solution=
                task4a_initial_solution,
            initial_delta=delta,
            operator=operator,
            time_limit=TASK4A_TUNING_SECONDS,
            decrease_delta=True,
            trace_every=500,
        )
 
        task4a_trials.append({
            "Operator": operator,
            "Initial Delta": delta,
            "Best objective":
                trial_result["best_objective"],
            "Accepted moves":
                trial_result["accepted_moves"],
            "Worse accepted":
                trial_result[
                    "accepted_worse_moves"
                ],
        })
 
task4a_trials_df = pd.DataFrame(task4a_trials)
display(task4a_trials_df.round(6))
 
best_trial = min(
    task4a_trials,
    key=lambda row: row["Best objective"],
)
selected_operator = best_trial["Operator"]
selected_delta = best_trial["Initial Delta"]
 
print("Selected operator:", selected_operator)
print("Selected initial Delta:", f"{selected_delta:.6f}")
 
random.seed(5000)
task4a_result = threshold_acceptance(
    initial_solution=task4a_initial_solution,
    initial_delta=selected_delta,
    operator=selected_operator,
    time_limit=TASK4A_FINAL_SECONDS,
    decrease_delta=True,
    trace_every=100,
)
task4a_result["best_solution"].check_feasibility()
 
task4a_summary_df = pd.DataFrame([{
    "N": 200,
    "Initial objective":
        task4a_result["initial_objective"],
    "Operator": task4a_result["operator"],
    "Initial Delta":
        task4a_result["initial_delta"],
    "Best objective":
        task4a_result["best_objective"],
    "Final current objective":
        task4a_result[
            "final_current_objective"
        ],
    "Iterations": task4a_result["iterations"],
    "Accepted moves":
        task4a_result["accepted_moves"],
    "Worse accepted":
        task4a_result[
            "accepted_worse_moves"
        ],
    "Runtime (s)": task4a_result["runtime"],
}])
display(task4a_summary_df.round(6))
 
# ## Task 4B — Comparison of Iterative Improvement, ILS, and Threshold Acceptance
 
TASK4B_TIME_LIMIT_SECONDS = 15.0
TRACE_INITIALISATION = "SEH"
 
# Build one shared initial solution per N and initialisation method.
initial_solutions = {}
initialisation_rows = []
 
for n in (100, 200):
    instance = instances_by_size[n]
 
    random.seed(1000 + n)
    starts = {
        "Random": random_initialisation(instance),
        "NNH":
            best_nearest_neighbour_heuristic(instance),
        "SEH": shortest_edge_heuristic(instance),
        "LEH": longest_edge_heuristic(instance),
    }
 
    for name, solution in starts.items():
        solution.check_feasibility()
        initial_solutions[(n, name)] = solution
        initialisation_rows.append({
            "N": n,
            "Initialisation": name,
            "Initial objective":
                solution.calculate_objective(),
        })
 
initialisation_df = pd.DataFrame(
    initialisation_rows
)
print("INITIAL SOLUTIONS")
display(initialisation_df.round(4))
 
 
def task4b_ta(initial_solution, time_limit):
    return threshold_acceptance(
        initial_solution=initial_solution,
        initial_delta=selected_delta,
        operator=selected_operator,
        time_limit=time_limit,
        decrease_delta=True,
        trace_every=100,
    )
 
 
algorithm_runners = {
    "Iterative Improvement":
        iterative_improvement_search,
    "Iterated Local Search":
        iterated_local_search,
    "Threshold Acceptance":
        task4b_ta,
}
 
task4b_rows = []
trace_results = {}
 
for n in (100, 200):
    for initialisation_index, initialisation in enumerate(
        ("Random", "NNH", "SEH", "LEH")
    ):
        starting_solution = initial_solutions[
            (n, initialisation)
        ]
 
        for algorithm_index, (
            algorithm_name,
            runner,
        ) in enumerate(algorithm_runners.items()):
            random.seed(
                50000
                + n * 100
                + initialisation_index * 10
                + algorithm_index
            )
 
            print(f"Running N={n}, initialisation={initialisation}, algorithm={algorithm_name}")
 
            result = runner(
                starting_solution.clone(),
                TASK4B_TIME_LIMIT_SECONDS,
            )
            result["best_solution"].check_feasibility()
 
            initial_objective = (
                starting_solution.calculate_objective()
            )
            best_objective = result["best_objective"]
            improvement = (
                100.0
                * (
                    initial_objective
                    - best_objective
                )
                / initial_objective
            )
 
            task4b_rows.append({
                "N": n,
                "Initialisation": initialisation,
                "Algorithm": algorithm_name,
                "Initial objective":
                    initial_objective,
                "Best objective": best_objective,
                "Improvement (%)": improvement,
                "Runtime (s)": result["runtime"],
                "Iterations": result["iterations"],
            })
 
            if (
                n == 200
                and initialisation
                    == TRACE_INITIALISATION
            ):
                trace_results[
                    algorithm_name
                ] = result
 
task4b_results_df = pd.DataFrame(task4b_rows)
task4b_results_df = task4b_results_df.sort_values(
    ["N", "Initialisation", "Best objective"]
).reset_index(drop=True)
 
print("TASK 4B — FULL RESULTS")
display(task4b_results_df.round(4))
 
task4b_comparison_df = task4b_results_df.pivot_table(
    index=["N", "Initialisation"],
    columns="Algorithm",
    values="Best objective",
    aggfunc="first",
).reset_index()
 
print("TASK 4B — OBJECTIVE COMPARISON")
display(task4b_comparison_df.round(4))
 
best_per_algorithm_df = task4b_results_df.loc[
    task4b_results_df.groupby(
        ["N", "Algorithm"]
    )["Best objective"].idxmin()
].sort_values(["N", "Algorithm"])
 
print("BEST INITIALISATION FOR EACH ALGORITHM")
display(
    best_per_algorithm_df[
        [
            "N",
            "Algorithm",
            "Initialisation",
            "Best objective",
            "Runtime (s)",
        ]
    ].round(4)
)
 
best_overall_df = task4b_results_df.loc[
    task4b_results_df.groupby(
        "N"
    )["Best objective"].idxmin()
].sort_values("N")
 
print("BEST OVERALL RESULT FOR EACH PROBLEM SIZE")
display(
    best_overall_df[
        [
            "N",
            "Algorithm",
            "Initialisation",
            "Best objective",
            "Runtime (s)",
        ]
    ].round(4)
)
 
task4b_results_df.to_csv(
    OUTPUT_DIR / "task4b_full_results.csv",
    index=False,
)
task4b_comparison_df.to_csv(
    OUTPUT_DIR / "task4b_objective_comparison.csv",
    index=False,
)
 
for algorithm_name in algorithm_runners:
    result = trace_results[algorithm_name]
 
    plt.figure(figsize=(10, 6))
    plt.plot(
        result["trace_time"],
        result["trace_current"],
    )
    plt.xlabel("Execution time (seconds)")
    plt.ylabel(
        "Current objective function value"
    )
    plt.title(
        f"{algorithm_name} Execution Trace "
        f"(N=200, "
        f"initialisation={TRACE_INITIALISATION})"
    )
    plt.grid(True)
    plt.tight_layout()
 
    file_name = (
        "task4b_trace_"
        + algorithm_name.lower().replace(" ", "_")
        + ".png"
    )
    plt.savefig(
        OUTPUT_DIR / file_name,
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
 
print("Saved files:")
print("- task4b_full_results.csv")
print("- task4b_objective_comparison.csv")
for algorithm_name in algorithm_runners:
    print(
        "- task4b_trace_"
        + algorithm_name.lower().replace(" ", "_")
        + ".png"
    )
 