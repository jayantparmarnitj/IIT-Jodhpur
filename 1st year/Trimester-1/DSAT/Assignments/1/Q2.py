def mearge_sort(s):
    mid = len(s) // 2
    if len(s) <= 1:
        return s
    left = mearge_sort(s[:mid])
    right  = mearge_sort(s[mid:])
    sorted_lst  = merge(left, right)
    return sorted_lst

def merge(left, right):
    sorted_list = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list
def find_sum(lst, i, j, x):
    sum = 0
    while(i<j):
        sum = lst[i] + lst[j]
        if sum == x:
            return True, (lst[i],lst[j])
        elif sum < x:
            i += 1
        else:
            j -= 1
    return False, []
    
if __name__ == "__main__":
    input_set = set([8, 20, 3, 14, 6])
    x = 11
    print("Original Set:", input_set)
    input_list = list(input_set) 
    sorted_list = mearge_sort(input_list)
    status, sum_pairs = find_sum(sorted_list, 0, len(input_list)-1, x)
    print("Sum exist status :", status)
    print("Sum pairs:", sum_pairs)

