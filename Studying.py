import sys
from collections import defaultdict

input = lambda: sys.stdin.readline().rstrip("\r\n")
MII = lambda: map(int, input().split())

n = int(input())

def can_make_equal(n, grid):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            center = grid[i][j]
            neighbors = [grid[i - 1][j], grid[i + 1][j], grid[i][j - 1], grid[i][j + 1]]
            total_diff = sum(neighbors) - center * 4
            if total_diff % 4 != 0:
                return "NO"
            adjustment = total_diff // 4
            grid[i - 1][j] -= adjustment
            grid[i + 1][j] -= adjustment
            grid[i][j - 1] -= adjustment
            grid[i][j + 1] -= adjustment
    return "YES"


# 读取输入
n = int(input())
grid = [[0] * (n + 2)]  # 加入边界行
for _ in range(n):
    grid.append([0] + list(map(int, input().split())) + [0])  # 加入边界列
grid.append([0] * (n + 2))  # 加入边界行

# 输出结果
print(can_make_equal(n, grid))
