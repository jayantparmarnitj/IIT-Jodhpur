def cut_rod_dp(price, n):
    dp = [0] * (n + 1)  # dp[i] will store max revenue for rod length i

    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            max_val = max(max_val, price[j] + dp[i - j - 1])
        dp[i] = max_val

    return dp[n]

if __name__ == "__main__":
    price = [1, 5, 8, 9, 10, 17, 17, 20]
    n = len(price)
    print(cut_rod_dp(price, n))
