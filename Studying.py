import sys
from collections import defaultdict

input = lambda: sys.stdin.readline().rstrip("\r\n")
MII = lambda: map(int, input().split())

n, m = MII()

a = [0] + [int(i) for i in input().split()]
Sum = [0] * (m + 1)

for i in range(1, m + 1):
    Sum[i] = Sum[i - 1] + a[i]

f = [[0] * (m + 2) for _ in range(n + 2)]

for i in range(1, n + 1):
    for j in range(1, m):
        f[i] = max(f[i], f[i - j])