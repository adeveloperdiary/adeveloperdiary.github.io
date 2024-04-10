---
title: Tree - Convert Sorted Array to Binary Search Tree
categories: [algorithm, tree]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> Find Middle, Split, Recursion
{: .prompt-info }

## Problem

Given an integer array `nums` where the elements are sorted in **ascending order**, convert *it to a* **height-balanced** *binary search tree*.

> A **height-balanced** binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

**Example 1:**

<img src="../assets/img/btree1.jpeg" alt="addtwonumber1"  />

```
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:
```

![btree2](../assets/img/btree2.jpeg)

**Example 2:**

![btree](../assets/img/btree.jpeg)

```
Input: nums = [1,3]
Output: [3,1]
Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
```

## Solution

Eventhough sounds complex, this is a fairly simple problem to solve. Since the array is already sorted, we need to find the middle (in case of even length, there will be two solutions), then assign the left partition nodes to the left tree and right partition nodes to right tree. We can do this recursively until reached the leaf nodes.

The structure of the solution is very similar to the [Merge Two Binary Trees](https://adeveloperdiary.com/algorithm/tree/merge-two-binary-trees/) problem.

We will start by desiging our recursive function. The function takes two parameters `left_index` & `right_index`.

```python
def binary_search_tree(left_index, right_index):
```

The base condition is we should stop when `left_index > right_index` as at that point we have already crossed the middle.

```python
def binary_search_tree(left_index, right_index):
  if left_index > right_index:
    return None
```

Now calculate the `middle_index`. 

```python
  middle_index = (left_index+right_index)//2
```

Create new node for the root node. `nums` is the original sorted array. 

```python
  root = TreeNode(nums[middle_index])
```

Then call the `binary_search_tree()` recursively by passing the new left and right index.

```python
  root.left = binary_search_tree(left_index, middle_index-1)
  root.right = binary_search_tree(middle_index+1, right_index)
```

Finally return `root`.

## Final Code

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def sorted_array_to_bst(nums):
  def binary_search_tree(left_index, right_index):
    if left_index > right_index:
      return None
    
    middle_index = (left_index+right_index)//2
    
    root = TreeNode(nums[middle_index])
    
    root.left = binary_search_tree(left_index,middle_index-1)
    root.right = binary_search_tree(middle_index+1,right_index)
    
    return root
  return binary_search_tree(0,len(nums)-1)
```

