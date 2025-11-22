import heapq
from typing import Dict, List, Tuple, Any, Optional

def dijkstra(adj: Dict[Any, List[Tuple[Any, float]]], s: Any):
    """
    Follows the pseudocode provided.
    adj: adjacency list mapping u -> list of (v, weight)
    s: source node
    Returns: (d, pi)
      d: dict of shortest distances (float('inf') if unreachable)
      pi: dict of predecessors (None if NIL)
    """
    # 2. Create arrays (dicts in Python)
    d = {}
    pi = {}
    Q = []               # priority queue of (distance, node)
    in_queue = set()     # optional: track nodes ever inserted (not required)

    # 3-5. for each vertex
    # Ensure all nodes appear (also include targets that might not be keys)
    nodes = set(adj.keys())
    for u in adj.values():
        for v, _w in u:
            nodes.add(v)

    for v in nodes:
        d[v] = float('inf')
        pi[v] = None
        heapq.heappush(Q, (d[v], v))   # Insert (Q, v, d[v])
        in_queue.add(v)

    # 6-7. set source distance and Decrease-Key (push new pair)
    d[s] = 0.0
    heapq.heappush(Q, (0.0, s))       # Decrease-Key(Q, s, 0)

    visited = set()  # corresponds to S in pseudocode

    # 8. while Q != empty
    while Q:
        dist_u, u = heapq.heappop(Q)      # 9. Extract-Min(Q)
        # skip stale heap entries
        if dist_u != d[u]:
            continue

        # 10. S = S ∪ {u}
        visited.add(u)

        # 11. for each v in Adj[u]
        for (v, w_uv) in adj.get(u, []):
            # 12. if d[v] > d[u] + w(u,v)
            if d[v] > d[u] + w_uv:
                # 13-14. relax
                d[v] = d[u] + w_uv
                pi[v] = u
                # 15. Decrease-Key(Q, v, d[v]) -> push new key
                heapq.heappush(Q, (d[v], v))

    return d, pi

def reconstruct_path(pi: Dict[Any, Optional[Any]], s: Any, t: Any) -> List[Any]:
    """Reconstruct path s -> t using predecessor map pi. Returns [] if unreachable."""
    if t not in pi:
        return []
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        if cur == s:
            break
        cur = pi[cur]
    path.reverse()
    return path if path and path[0] == s else []

# Example usage
if __name__ == "__main__":
    graph = {
        'A': [('B', 2), ('C', 5)],
        'B': [('C', 1), ('D', 4)],
        'C': [('D', 1)],
        'D': [('E', 3)],
        'E': []
    }

    d, pi = dijkstra(graph, 'A')
    print("Distances:", d)
    print("Predecessors:", pi)
    print("Shortest A -> E:", reconstruct_path(pi, 'A', 'E'), "cost =", d['E'])
