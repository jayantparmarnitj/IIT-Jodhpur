def cut_rod(price, n):
    if n == 0:
        return 0
    result = 0
    for i in range(1, n + 1):
        result = max(result, price[i - 1] + cut_rod(price, n - i))
    return result

if __name__ == "__main__":
    price = [1, 5, 8, 9, 10, 17, 17, 20]
    n = len(price)
    print(cut_rod(price, n))