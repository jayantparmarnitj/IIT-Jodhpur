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

    # Find rank of a given key (number of elements ≤ key)
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

    def inorder(self):
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.value)
                traverse(node.right)
        traverse(self.root)
        return result


# Example usage
if __name__ == "__main__":
    tree = AugmentedBST()
    for val in [20, 8, 22, 4, 12, 10, 14]:
        tree.insert(val)

    print("Inorder traversal:", tree.inorder())
    print("Tree size (root):", tree.root.size)
    print("Rank of 14:", tree.rank(14))
    print("3rd smallest element:", tree.find_kth(3))
