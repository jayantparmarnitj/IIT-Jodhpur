from collections import defaultdict, deque

class FlowNetwork:
    def __init__(self):
        self.cap = defaultdict(lambda: defaultdict(float))  # capacity[u][v]

    def add_edge(self, u, v, c):
        self.cap[u][v] += c         # allow parallel edges by summing
        self.cap[v]                  # ensure keys exist

    def _dfs_path(self, s, t):
        """Find any s->t path in residual graph via DFS; return (path, bottleneck)."""
        stack = [(s, float('inf'))]
        parent = {s: None}
        while stack:
            u, f = stack.pop()
            if u == t:
                # reconstruct
                path = []
                cur = t
                bottleneck = f
                while parent[cur] is not None:
                    p = parent[cur]
                    path.append((p, cur))
                    bottleneck = min(bottleneck, self.res[p][cur])
                    cur = p
                path.reverse()
                return path, bottleneck
            for v, rc in self.res[u].items():
                if rc > 0 and v not in parent:
                    parent[v] = u
                    stack.append((v, min(f, rc)))
        return None, 0.0

    def max_flow(self, s, t):
        # initialize residual graph with forward + reverse edges
        self.res = defaultdict(lambda: defaultdict(float))
        for u in list(self.cap.keys()):
            for v, c in self.cap[u].items():
                self.res[u][v] += c
                self.res[v]       # ensure keys
                # reverse residual initially 0

        flow_value = 0.0
        while True:
            path, delta = self._dfs_path(s, t)
            if not path:
                break
            # augment along the path
            for u, v in path:
                self.res[u][v] -= delta
                self.res[v][u] += delta
            flow_value += delta
        return flow_value
if __name__ == "__main__":
    g = FlowNetwork()
    # classic toy network
    g.add_edge('s', 'a', 3)
    g.add_edge('s', 'b', 2)
    g.add_edge('a', 'b', 1)
    g.add_edge('a', 't', 2)
    g.add_edge('b', 't', 3)

    print("Max flow:", g.max_flow('s', 't'))  # e.g., 4 or 5 depending on capacities
