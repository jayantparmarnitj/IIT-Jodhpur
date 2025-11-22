class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.head = self  # initially, head points to itself


class DisjointSetLinkedList:
    def __init__(self):
        self.members = {}  # maps value -> node
        self.heads = {}    # maps head node -> tail node (for efficient append)

    def make_set(self, value):
        """Create a new set with one element."""
        node = Node(value)
        self.members[value] = node
        self.heads[node] = node  # head = tail at start

    def find_set(self, value):
        """Find the representative (head value) of the set containing the element."""
        node = self.members.get(value)
        if node:
            return node.head.value
        return None

    def union(self, value1, value2):
        """Merge the two sets containing value1 and value2."""
        node1 = self.members[value1]
        node2 = self.members[value2]

        head1 = node1.head
        head2 = node2.head

        # If they are already in the same set
        if head1 == head2:
            print(f"{value1} and {value2} are already in the same set.")
            return

        # Get tails
        tail1 = self.heads[head1]
        tail2 = self.heads[head2]

        # Append list2 to list1
        tail1.next = head2

        # Update head pointers for all nodes in second list
        temp = head2
        while temp:
            temp.head = head1
            temp = temp.next

        # Update tail reference
        self.heads[head1] = tail2
        del self.heads[head2]

    def display_sets(self):
        """Print all disjoint sets."""
        printed_heads = set()
        for node in self.heads.keys():
            if node not in printed_heads:
                printed_heads.add(node)
                elements = []
                temp = node
                while temp:
                    elements.append(temp.value)
                    temp = temp.next
                print(f"Set Representative {node.value}: {elements}")
# Create Disjoint Sets
ds = DisjointSetLinkedList()
for i in range(1, 6):
    ds.make_set(i)

ds.display_sets()
# Perform unions
ds.union(1, 2)
ds.union(3, 4)
ds.union(2, 3)

# Display all sets
ds.display_sets()

# Find representatives
print("Find(4):", ds.find_set(4))
print("Find(5):", ds.find_set(5))
