---
title: Binary Search - Valid Perfect Square
categories: [algorithm, two-pointers]
tags: [datastructure]
hidden: true
---

> **Difficulty** : Easy
{: .prompt-tip }

> Binary Search , Same solution for Search Insert Position
{: .prompt-info }


## Problem

Given a positive integer num, return true if num is a perfect square or false otherwise.

- **Input** : `16, 14`
- **Output** :  `True, False`

## Solution

1. This is a type of problem which can be solved by **Binary Search** in `O(log n)` time.
2. The square root will be any number between `0 - num`. Thechnically it will be much lower than `num` so the **Binary Search** solution can certainly be optimized, however the intention is to identify that the problem can be solved using Binary Search.

## Code

```python
def is_perfect_square(num):
    l, r = 0, num

    # Binary Search
    while l <= r:
        mid = (l+r)//2
        square = mid*mid
        if square == num:
            return True
        elif square > num:
            r = mid-1
        else:
            l = mid+1
    return False

print(is_perfect_square(15))
print(is_perfect_square(16))
```

```
False
True
```



