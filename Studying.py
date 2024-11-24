import sys
from collections import defaultdict
from typing import List

input = lambda: sys.stdin.readline().rstrip("\r\n")
MII = lambda: map(int, input().split())


def solve(a):
    def get(x):
        by_9 = (x % 9 == 0)
        by_5 = (x % 5 == 0)
        by_11 = (x % 11 == 0)
        return by_9, by_5, by_11

    counts = defaultdict(int)
    n = len(a)
    for x in a:
        counts[get(x)] += 1

    def cb(count):
        beauty = 0
        type1 = count[(True, True, True)]
        type9 = count[(True, False, False)]
        type5 = count[(False, True, False)]
        type11 = count[(False, False, True)]

        beauty += type1 * (n - type1)
        beauty += type9 * type5
        beauty += type9 * type11
        beauty += type5 * type11
        return beauty

    ans = cb(counts)

    for i in range(n):
        old_type = get(a[i])
        new_type = get(a[i] + 1)
        if old_type == new_type:
            continue
        counts[old_type] -= 1
        counts[new_type] += 1

        ans = max(ans, cb(counts))
        counts[new_type] -= 1
        counts[old_type] += 1

    return ans


n = int(input())
a = [int(i) for i in input().split()]

print(solve(a))
