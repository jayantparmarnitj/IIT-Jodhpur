def insertion_sort(n, lst):
    for i in range(1, n):
        key = lst[i]
        j = i - 1
        # Shift elements that are greater than key
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst


if __name__ == "__main__":
    n = int(input("Enter number of elements to be sorted : "))
    lst = [int(input("Enter number ")) for i in range(n)] 
    print("Input list : ", lst)
    print("Sorted list :", insertion_sort(n, lst))
