def find_kunique(lst, k):
    d = {}
    for index,  value in enumerate(lst):
        if value in d and abs(index - d[value]) <= k:
            print(f"value at index {index} and index {d[value]} is same and |i-j| is {abs(index - d[value])} which is less than k = {k}")
            return False
        d[value] = index
    return True
if __name__ == "__main__":
    lst = [ 1, 2, 4, 3, 5, 7, 8, 6, 7,9,7]
    k = 4
    print(f"Input list: {lst} with k={k}")
    status = find_kunique(lst, k)
    print("k-unique status : ",status)
    

