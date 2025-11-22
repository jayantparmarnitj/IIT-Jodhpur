def lcs(X: str, Y: str):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Reconstruct the LCS string
    i, j = m, n
    lcs_seq = []

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_seq.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_seq.reverse()

    return L[m][n], ''.join(lcs_seq)

# Example
if __name__ == "__main__":
    X = "AGGTAB"
    Y = "GXTXAYB"
    length, sequence = lcs(X, Y)
    print("LCS length:", length)
    print("LCS sequence:", sequence)
