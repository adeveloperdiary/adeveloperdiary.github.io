---
title: Tree - Merge Two Binary Trees
categories: [algorithm, tree]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> PreOrder dfs()
{: .prompt-info }

## Problem

You are given two binary trees `root1` and `root2`. Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree. Return *the merged tree*.

**Note:** The merging process must start from the root nodes of both trees.

**Example 1:**

<img src="../assets/img/merge.jpeg" alt="addtwonumber1"  />

```
Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
```

**Example 2:**

```
Input: root1 = [1], root2 = [1,2]
Output: [2,2]
```

## Solution

A PreOrder DFS can be used to solve this. We nee to traverse the trees together and sum the node value.

Start with the base condition. In case both of the nodes are `None` return `None`.

```python
if not root1 and not root2:
  return None
```

Next, just like LinkedLink merge, we need to make sure if one of the root is `None` we set the `val` to `0`.

```python
val1 = root1.val if root1 else 0
val2 = root2.val if root2 else 0
```

 Create a new node with the summed value.

```python
root = TreeNode(val1+val2)
```

Create new `left` node by traversing through `left` node of both `root1` and `root2`. (same for `right` node). `merge_trees` is called recursively here.

```python
root.left = merge_trees(root1.left if root1 else None,root2.left if root2 else None)
root.right = merge_trees(root1.right if root1 else None,root2.right if root2 else None)

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

def merge_trees(root1, root2):
    if not root1 and not root2:
        return None

    val1 = root1.val if root1 else 0
    val2 = root2.val if root2 else 0

    root = TreeNode(val1+val2)

    root.left = merge_trees(root1.left if root1 else None,
                            root2.left if root2 else None)
    root.right = merge_trees(
        root1.right if root1 else None, root2.right if root2 else None)

    return root
```

