---
title: Binary Search - Search in Rotated Sorted Array
categories: [algorithm, two-pointers]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> Binary Search, First find which **side** is sorted
{: .prompt-info }


## Problem

Given the array **nums** after the **possible** **rotation** and an integer **target**, return the index of target if it is in **nums**, or **-1** if it is not in **nums**.

### Example 1:

- **Input** :  nums = `[4,5,6,7,0,1,2]`, target = `0`    	
- **Output** : `4`

## Solution

- We need to run a modified binary search for solving this.
- First need to find which side of middle is sorted & then based on it check for the conditions.

![Search in Rotated Sorted Array](../assets/img/search_in_rotated_sorted_array.jpg)

Its very important to understand the above 4 diagrams.  They shows the logic on how to move `l` and `r` pointers.

##  Code 1

```python
def rotated_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2

        # Return if target is found
        if target == nums[mid]:
            return mid

        # If left is smaller than middle then
        # left of the middle is sorted.
        if nums[l] <= nums[mid]:
            # there are three loctions where
            # the trget could be present.
            # We need to check for all three

            if target > nums[mid]:
                # 1. Target is right of middle and
                # in the same partition
                # In this case move left to mid +1
                l = mid + 1
            elif target < nums[l]:
                # 2. target is in the right partition
                # In this case move the left to mid+1
                l = mid + 1
            else:
                # 3. Target is left of middle in the same
                # sorted partition
                # In this case move right to mid-1
                r = mid - 1
        else:
            # The right of the middle is sorted.
            # Again check for three conditions.
            if target < nums[mid]:
                # 1. Target is in the left of the
                # middle and in same partition
                # In this case move right to mid-1
                r = mid - 1
            elif target > nums[r]:
                # 2. Target is in the other partition
                # In this case move right to mid-1
                r = mid - 1
            else:
                # 3. Target is right of middle in the
                # same sorted partition
                # In this case move left to mid+1
                l = mid + 1

    return -1
print(rotated_search([4,5,6,7,0,1,2],0))
```

```
4
```

## Code 2

Here is the alternative code using the reversed condition in line `16` and `24`.

```python
def rotated_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2

        # Return if target is found
        if target == nums[mid]:
            return mid

        # If left is smaller than middle then
        # left of the middle is sorted.
        if nums[l] <= nums[mid]:
            # there are three loctions where
            # the trget could be present.
            # We need to check for all three
            if nums[l] <= target < nums[mid]:
                r = mid-1
            else:
                l = mid+1

        else:
            # The right of the middle is sorted.
            # Again check for three conditions.
            if nums[mid] < target <= nums[r]:
                l = mid+1
            else:
                r = mid-1
    return -1
  print(rotated_search([4,5,6,7,0,1,2],0))
```

```
4
```

## Runtime Complexity

The runtime will be `O(log n)` as we are simply running a binary search.
