---
title: Tree - Trim a Binary Search Tree
categories: [algorithm, tree]
tags: [datastructure]
hidden: true
mermaid: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> DFS
{: .prompt-info }

## Problem

Given the `root` of a binary search tree and the lowest and highest boundaries as `low` and `high`, trim the tree so that all its elements lies in `[low, high]`. Trimming the tree should **not** change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). It can be proven that there is a **unique answer**.

Return *the root of the trimmed binary search tree*. Note that the root may change depending on the given bounds.

**Example 1:**

<img src="../assets/img/trim1.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: root = [1,0,2], low = 1, high = 2
Output: [1,null,2]
```

**Example 2:**

<img src="../assets/img/trim2.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: root = [3,0,4,null,2,null,null,1], low = 1, high = 3
Output: [3,2,null,1]
```

## Solution

Since the tree is BST we know that the left subtree of a root will always have values lesser than it and the right subtree of the root will always have values greater than it. We can use recursion to just trim the tree.

> Since its already BST we just need to trim the tree and not remove specific nodes. This makes this problem having a lower complexity.

Start with the base case.

```python
def trim_bst(root, low, high):
  if not root:
    return None
```

Now consider a scenario where the root of the enter tree is less than the `low` value. In that case we need to send back a new root. We can simply ignore the `left` sub-tree completely and also the `root` node in this case. We can call `trim_bst` again to explore the right sub-tree.

```python
  if root.val < low:
    return trim_bst(root.right, low, high)
```

Conversely, do the same for `right` sub-tree (explore left sub-tree).

```python
  if root.val > high:
    return trim_bst(root.left, low, high)
```

If neither of these happens, then we know that the `root.val` with-in the `low`, `high` range. So `root` is definitely going to be returned. Now lets find if the same is true for its `left` and `right` subtree.

```python
  root.left = trim_bst(root.left, low, high)
  root.right = trim_bst(root.right, low, high)
  
  return root
```

## Final Code 

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def trim_bst(root, low, high):
  if not root:
    return None
  
  if root.left < low:
    return trim_bst(root.right,low,high)
  if root.right > high:
    return trim_bst(root.left,low,high)
  
  root.left = trim_bst(root.left,low,high)
  root.right = trim_bst(root.right,low,high)
  
  return root
```

