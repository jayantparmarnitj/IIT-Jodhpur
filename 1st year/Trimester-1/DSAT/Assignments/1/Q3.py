class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
def insert(root, key):
    if root is None:
        return Node(key)
    if root.val == key:
        return root
    if root.val < key:
        root.right = insert(root.right, key)
    else:
        root.left = insert(root.left, key)
    return root
def min_abs_diff(n1, n2, diff):
    new_diff = abs(n1 - n2)
    if new_diff < diff:
        return new_diff
    return diff
def inorder(root, pre_val, diff):
    if root:
        pre_val, diff = inorder(root.left, pre_val, diff)
        if pre_val is not None:
            diff = min_abs_diff(root.val, pre_val, diff)
        pre_val = root.val
        print(pre_val)
        pre_val, diff = inorder(root.right, pre_val, diff)

    return pre_val, diff
if __name__ == '__main__':
    r = Node(50)
    r = insert(r, 30)
    r = insert(r, 29)
    r = insert(r, 40)
    r = insert(r, 70)
    r = insert(r, 60)
    r = insert(r, 80)
    r = insert(r, 8)
    print("Inorder Traversal : ")
    _, diff = inorder(r, None, float('inf'))
    print(f"Minimum absolute difference is {diff} ")




