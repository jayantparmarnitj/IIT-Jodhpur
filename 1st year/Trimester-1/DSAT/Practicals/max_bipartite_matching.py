from collections import deque, defaultdict
from typing import Dict, List, Tuple, Iterable, Hashable, Set

Node = Hashable

class MaxFlow:
    def __init__(self):
        self.cap = defaultdict(lambda: defaultdict(int))  # capacity[u][v]
        self.adj = defaultdict(set)                      # neighbor list for BFS

    def add_edge(self, u: Node, v: Node, c: int = 1):
        self.cap[u][v] += c
        self.cap[v][u] += 0          # ensure reverse exists
        self.adj[u].add(v)
        self.adj[v].add(u)

    def edmonds_karp(self, s: Node, t: Node) -> int:
        flow = 0
        while True:
            parent = {s: None}
            bottleneck = {s: float('inf')}
            q = deque([s])
            # BFS on residual graph
            while q and t not in parent:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in parent and self.cap[u][v] > 0:
                        parent[v] = u
                        bottleneck[v] = min(bottleneck[u], self.cap[u][v])
                        q.append(v)
                        if v == t:
                            break
            if t not in parent:
                break  # no augmenting path
            # augment
            delta = bottleneck[t]
            flow += delta
            v = t
            while v != s:
                u = parent[v]
                self.cap[u][v] -= delta
                self.cap[v][u] += delta
                v = u
        return flow

def max_bipartite_matching(U: Iterable[Node],
                           V: Iterable[Node],
                           edges: Iterable[Tuple[Node, Node]]):
    """
    U: nodes in left partition
    V: nodes in right partition
    edges: iterable of (u in U, v in V) edges
    Returns: (max_matching_size, matching_set_of_pairs)
    """
    g = MaxFlow()
    s, t = "__SRC__", "__SNK__"

    U = set(U)
    V = set(V)

    # Build network
    for u in U:
        g.add_edge(s, u, 1)
    for v in V:
        g.add_edge(v, t, 1)
    for u, v in edges:
        if u in U and v in V:
            g.add_edge(u, v, 1)

    maxflow = g.edmonds_karp(s, t)

    # Recover matching: edges u->v with flow 1 (i.e., residual reverse cap > 0)
    matching = set()
    for u in U:
        for v in g.adj[u]:
            if v in V and g.cap[v][u] == 1:   # reverse residual carries the unit flow
                matching.add((u, v))

    return maxflow, matching

# Example
if __name__ == "__main__":
    U = {"u1", "u2", "u3"}
    V = {"v1", "v2"}
    E = {("u1", "v1"), ("u1", "v2"), ("u2", "v1"), ("u3", "v2")}
    size, match = max_bipartite_matching(U, V, E)
    print("Max matching size:", size)     # -> 2
    print("Matched pairs:", match)        # e.g., {('u1','v1'), ('u3','v2')}
