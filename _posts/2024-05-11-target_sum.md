---
title: Backtracking - Target Sum
categories: [algorithm, matrices]
tags: [datastructure]
hidden: true
---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> DFS, Backtracking, Caching
{: .prompt-info }

## Problem

You are given an integer array `nums` and an integer `target`.

You want to build an **expression** out of nums by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1"`.

Return the number of different **expressions** that you can build, which evaluates to `target`.

**Example 1:**

```
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

**Example 2:**

```
Input: nums = [1], target = 1
Output: 1
```

## Solution

Since we need to find all different expressions which sums up to the target, we can use `dfs()` with backtracking to solve this. Every valid expressions will increment the `result` by `1`. Like other problems we have seen already, using a `cache` will help not to reevaluate same path again & again.

Define the cache. The `dfs()` function will take current index we are processing and the current sum using all the numbers we have already used. So the cache will be a map object and the key will be `(index, current_sum)`

```python
cache = {}
def dfs(index, current_sum):
  
```

The very first thing is to define the base condition when we can decide if a valid path has been found. When the `index` reaches end of the array we will know it's time to find out if we have `current_sum == target`. ( Since the problem states that we need to use all the elements in the array the base condition is checked once we reach the end of the array )

```python
  if current_sum == target:
    return 1
  else:
    return 0
```

Return `1` if we have found the `target`, otherwise return `0`.

So if we have not reached the end of the array, find if the `result` is already in `cache`. If it is, then just return that.

```python
  if (index, current_sum) in cache:
    return cache[(index, current_sum)]
```

If we haven't reached the end of the array and also the result is not in cache, then we shall add (`+`)and subtract (`-`) the value at current `index` and run `dfs()` again.

```python
  result = dfs(index+1,current_sum+nums[index])+dfs(index+1,current_sum-nums[index])
```

Save the result to `cache` and return result.

```python
  cache[(index, current_sum)]=result
  return result

return dfs(0,0)
```

## Final Code

Here is the full code.

```python
def findTargetSumWays(nums, target) -> int:
    cache = {}

    def dfs(index, current_sum):
        if index == len(nums):
            if current_sum == target:
                return 1
            else:
                return 0

        if (index, current_sum) in cache:
            return cache[(index, current_sum)]

        result = dfs(index+1, current_sum +
                     nums[index])+dfs(index+1, current_sum-nums[index])

        cache[(index, current_sum)] = result
        return result

    return dfs(0, 0)
```







