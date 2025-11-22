def count_inversion(lst):
    mid = len(lst) // 2
    if len(lst) <= 1:
        return lst, 0, []
    left, left_inv, left_pairs = count_inversion(lst[:mid])
    right, right_inv, right_pairs = count_inversion(lst[mid:])
    sorted_lst, inv, inv_pairs = merge(left, right)
    total_inv = left_inv + right_inv + inv
    total_pairs = left_pairs + right_pairs + inv_pairs
    return sorted_lst, total_inv, total_pairs

def merge(left, right):
    sorted_list = []
    count = 0
    inversion_pairs = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            count += len(left) - i
            for k in range(i, len(left)):
                inversion_pairs.append((left[k], right[j]))
            j += 1
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list, count, inversion_pairs

if __name__ == "__main__":
    lst = [1, 2, 4, 3, 5, 9, 7, 8, 6]
    print("Original list:", lst)
    _, total_inversions, inversion_pairs = count_inversion(lst)
    print("Total inversions:", total_inversions)
    print("Inversion pairs:")
    for pair in inversion_pairs:
        print(pair)
