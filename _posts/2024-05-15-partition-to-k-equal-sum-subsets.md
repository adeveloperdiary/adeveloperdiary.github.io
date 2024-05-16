---
title: Backtracking - Partition to K Equal Sum Subsets
categories: [algorithm, backtracking]
tags: [datastructure]
hidden: true
---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> DFS, Backtracking
{: .prompt-info }

## Problem

Given an integer array `nums` and an integer `k`, return `true` if it is possible to divide this array into `k` non-empty subsets whose sums are all equal.

**Example 1:**

```
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: It is possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
```

**Example 2:**

```
Input: nums = [1,2,3,4], k = 3
Output: false
```

## Solution

The problem is very similar to [Combination Sum II](https://adeveloperdiary.com/algorithm/backtracking/combination-sum-ii/) however the difference is that we need to find `k` equal number of combinations and not less. Also in **Combination Sum II**, we can use one number again in a different combination, but here one number can only be used only once.

So this tells us that,

- We need to keep track of which number is already used and not to use that again (unless we backtrack). In the **Combination Sum I & II**, we had to keep track of the `path` only for one combination. However here it needs to be persisted across all traversals.

  ```python
  used=[False] * len(nums)
  ```

- Once we identify a solution `path_sum==target`, we need to run `dfs()` again by decrementing `k`. Then when `k==0`, we can return `True`

  ```python
  def dfs(index, path_sum, k):
    if k == 0:
      return True
  
    if path_sum==target:
      return dfs(0,0, k-1)
  ```

![image-20240515233025534](../assets/img/image-20240515233025534.jpg)

As we discuss in detail, this problem has similarities to the [N-Queens](https://adeveloperdiary.com/algorithm/backtracking/n-queens/) problem as well.

Now, here also we will implement using **template 2** that we have already discussed  [here](https://adeveloperdiary.com/algorithm/backtracking/combination-sum/).

![image-20240514221758079](../assets/img/image-20240514221758079.jpg)

The first thing to do is to find the `target`. Also define the `used` array for keeping track of the used numbers.

```python
target = sum(nums) //k
used=[False] * len(nums)
```

As discussed earlier, the `dfs()` will take three arguments, `index`, `path_sum` & `k`. Also define the conditions we have created earlier.

```python
def dfs(index, path_sum, k):
  if k == 0:
    return True

  if path_sum==target:
    return dfs(0,0, k-1)
```

Now as per the **template 2**, we will use a `for` loop till end of the `nums` array from current `index`. We have the condition to make sure current number is not `used` and the `path_sum+nums[j] <= target`. Then set the `used[j]=True`, run the `dfs()` function.

If the `dfs()` returns `True`, return `True` immediately. The `dfs()` will return `True` only if `if k == 0` and this will be `True` only if `dfs(0,0, k-1)` runs `k-1` times.

Finally backtrack by setting `used[j]=False`

```python
for j in range(index, len(nums)):
  if not used[j] and path_sum+nums[j] <= target:
    used[j]=True
    if dfs(j+1,path_sum+nums[j],k):
      return True
    used[j]=False
  
  return False
    
```

Finally, just invoke & return `dfs()` .

There are 

## Final Code

Here is the full code.

```python
def can_partition_k_subsets(nums, k):        
    
    target = sum(nums) //k
    used=[False] * len(nums)

    def dfs(index, path_sum, k):
        if k == 0:
            return True

        if path_sum==target:
            return dfs(0,0, k-1)

        for j in range(index, len(nums)):
            if used[j]==False and path_sum+nums[j] <= target:
                used[j]=True
                if dfs(j,path_sum+nums[j],k):
                    return True
                used[j]=False
        return False
    return dfs(0,0,k)
```
