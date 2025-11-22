# Python program to demonstrate all core operations in a Binary Search Tree

class Node:
    """A class to represent a single node in a BST."""
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# =============================================================================
# 1. CREATE and DISPLAY (Recursive Implementation)
# =============================================================================

def insert(root, key):
    """A utility function to insert a new node with the given key using recursion."""
    if root is None:
        return Node(key)
    if root.val == key:
        return root  # Ignore duplicate keys
    if root.val < key:
        root.right = insert(root.right, key)
    else:
        root.left = insert(root.left, key)
    return root

def inorder(root):
    """A utility function to do in-order tree traversal using recursion."""
    if root:
        inorder(root.left)
        print(root.val, end=" ")
        inorder(root.right)

# =============================================================================
# 2. SEARCH (Declared)
# =============================================================================

def search(root, key):
    """
    Recursively searches for a key in the BST.
    Should return the node containing the key if found, otherwise None.
    """
    pass

# =============================================================================
# 3. DELETE (Declared)
# =============================================================================

def deleteNode(root, key):
    """
    Recursively deletes a node with the given key from the BST.
    This is the most complex operation, as it needs to handle three cases:
    1. The node is a leaf node.
    2. The node has one child.
    3. The node has two children (requires finding the in-order successor).
    """
    pass

# =============================================================================
# 4. FIND MIN and MAX (Declared)
# =============================================================================

def find_min(root):
    """
    Finds the node with the minimum value in a given tree/subtree.
    This would typically be implemented by traversing to the leftmost node.
    """
    pass

def find_max(root):
    """
    Finds the node with the maximum value in a given tree/subtree.
    This would typically be implemented by traversing to the rightmost node.
    """
    pass

def get_min(node):
    if node == None:
        return None
    while(node.left != None):
        node = node.left
    return node

def get_max(node):
    if node == None:
        return None
    while(node.right != None):
        node = node.right
    return node
# =============================================================================
# 5. FIND SUCCESSOR and PREDECESSOR (Declared)
# =============================================================================

def find_successor(root, key):
    # Case-0
    if root is None:
        return None
    # Case-1
    if root.right is not None and root.val == key:
        return get_min(root.right)
    
    # Case-2
    y = None
    x = root
    while x is not None:
        if key < x.val:
            y = x
            x = x.left
        else:
            x = x.right
    return y


def find_predecessor(root, key):
    # Case-0
    if root is None:
        return None
    # Case-1
    if root.left is not None and root.val == key:
        return get_max(root.left)
    
    # Case-2
    y = None
    x = root
    while x is not None:
        if key > x.val:
            y = x
            x = x.right
        else:
            x = x.left
    return y


# =============================================================================
# Main execution block
# =============================================================================
if __name__ == '__main__':
    # Creating the following BST
    #      50
    #     /  \
    #    30   70
    #   / \   / \
    #  20 40 60 80
    #  /
    # 8
    r = Node(50)
    r = insert(r, 30)
    r = insert(r, 20)
    r = insert(r, 40)
    r = insert(r, 70)
    r = insert(r, 60)
    r = insert(r, 80)
    r = insert(r, 8)

    print("Initial in-order traversal of the BST:")
    inorder(r)
    print("\n\n" + "="*40)
    key = 8
    successor = find_successor(r, key)
    if successor:
        print("Successor of {} is ".format(key),successor.val)
    else:
        print("No Successor")

    predecessor = find_predecessor(r, key)
    if predecessor:
        print("Predecessor of {} is ".format(key),predecessor.val)
    else:
        print("No Predecesssor")


