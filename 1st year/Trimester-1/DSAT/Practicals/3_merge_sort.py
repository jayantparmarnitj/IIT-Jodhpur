def merge_sort(lst):
    mid = len(lst)//2
    if len(lst)<=1:
        return lst
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left,right)
def merge(left, right):
    sorted_list = []
    i = j = 0
    while i < len(left) and j < len(right) :
        if left[i] <= right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list


if __name__ == "__main__":
    n = int(input("Enter number of elements to be sorted : "))
    lst = [int(input("Enter number ")) for i in range(n)]
    print("Input list : ", lst)
    print("Sorted list :", merge_sort(lst))
