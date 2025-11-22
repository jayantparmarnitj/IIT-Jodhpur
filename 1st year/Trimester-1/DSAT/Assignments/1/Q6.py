class AugmentedNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.size = 1


class AugmentedBST:
    def __init__(self):
        self.root = None

    def _update_size(self, node):
        if node:
            left_size = node.left.size if node.left else 0
            right_size = node.right.size if node.right else 0
            node.size = 1 + left_size + right_size

    def _insert(self, node, value):
        if node is None:
            return AugmentedNode(value)
        if value < node.value:
            node.left = self._insert(node.left, value)
        elif value > node.value:
            node.right = self._insert(node.right, value)
        self._update_size(node)
        return node

    def insert(self, value):
        self.root = self._insert(self.root, value)

    def _find_kth(self, node, k):
        if not node:
            return None
        left_size = node.left.size if node.left else 0
        if k == left_size + 1:
            return node.value
        elif k <= left_size:
            return self._find_kth(node.left, k)
        else:
            return self._find_kth(node.right, k - left_size - 1)

    def find_kth(self, k):
        return self._find_kth(self.root, k)

    def _rank(self, node, key):
        if not node:
            return 0
        if key < node.value:
            return self._rank(node.left, key)
        elif key > node.value:
            left_size = node.left.size if node.left else 0
            return left_size + 1 + self._rank(node.right, key)
        else:
            left_size = node.left.size if node.left else 0
            return left_size + 1

    def rank(self, key):
        return self._rank(self.root, key)

    def kth_successor(self, x, k):

        r = self.rank(x)
        if r == 0:
            print(f"Element {x} not found in the tree.")
            return None
        target_rank = r + k
        if not self.root or target_rank > self.root.size:
            return None  # No such successor
        return self.find_kth(target_rank)

if __name__ == "__main__":
    tree = AugmentedBST()
    for val in [20, 8, 22, 4, 12, 10, 14]:
        tree.insert(val)

    x, k = 10, 2
    print(f"The {k}-th successor of {x} is:", tree.kth_successor(x, k))
