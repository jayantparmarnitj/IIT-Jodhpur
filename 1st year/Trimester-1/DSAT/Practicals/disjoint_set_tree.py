class DisjointSet:
    def __init__(self):
        # parent[x] = parent of x; if parent[x] == x => x is representative
        # rank[x] = approx. tree height (used for union by rank)
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        """Create a new set containing x (if not already present)."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        """Find representative of x with path compression.
           Raises KeyError if x not present."""
        if x not in self.parent:
            raise KeyError(f"{x} is not found. Call make_set({x}) first.")
        # Path compression (recursive)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y using union by rank.
           Returns True if a union happened (they were separate), False if already in same set."""
        # Ensure both elements exist
        if x not in self.parent or y not in self.parent:
            raise KeyError("Both elements must be present. Call make_set for each before union.")

        rx = self.find(x)
        ry = self.find(y)

        if rx == ry:
            return False  # already in same set

        # attach smaller rank tree under larger rank tree
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            # equal rank: choose one as new root and increment its rank
            self.parent[ry] = rx
            self.rank[rx] += 1

        return True

    def connected(self, x, y):
        """Check if x and y are in same set."""
        if x not in self.parent or y not in self.parent:
            raise KeyError("Both elements must be present.")
        return self.find(x) == self.find(y)

    def get_all_sets(self):
        """Return a dict mapping representative -> list of members (useful for inspection)."""
        groups = {}
        for x in list(self.parent.keys()):
            r = self.find(x)
            groups.setdefault(r, []).append(x)
        return groups

ds = DisjointSet()
for i in range(1, 8):
    ds.make_set(i)

ds.union(1, 2)
ds.union(2, 3)
ds.union(4, 5)
ds.union(6, 7)
ds.union(5, 6)   # merges sets {4,5} and {6,7}
print(ds.get_all_sets())
print(ds.connected(1, 3))   # True
print(ds.connected(1, 4))   # False

ds.union(3, 4)   # merge the two big components
print(ds.connected(1, 7))   # True

print(ds.get_all_sets())
# Possible output:
# {1: [1,2,3,4,5,6,7]}  (representatives may differ but elements will be grouped)
