def is_prime(n):
    if n==1:
        return "No"
    else:
        for i in range(2,n):
            if (n%i) == 0:
                return "No"
    return "Yes"

if __name__ == "__main__":
    n = int(input("Enter a number: "))
    print(f"{n} is prime? {is_prime(n)}")