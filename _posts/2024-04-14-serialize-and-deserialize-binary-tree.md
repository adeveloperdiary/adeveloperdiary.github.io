---
title: Tree - Serialize and Deserialize Binary Tree
categories: [algorithm, tree]
tags: [datastructure]
hidden: true
mermaid: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> Recursion
{: .prompt-info }

## Problem

Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not. Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Example 1:**

<img src="../assets/img/ex1.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: p = [1,2,3], q = [1,2,3]
Output: true
```

**Example 2:**

<img src="../assets/img/ex2.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: p = [1,2], q = [1,null,2]
Output: false
```

**Example 3:**

<img src="../assets/img/ex3.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: p = [1,2,1], q = [1,1,2]
Output: false
```

## Final Code 

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def same_tree(p:TreeNode, q:TreeNode):
  # Return None if both the nodes
  # are None.
  if not p and not q:
    return True
  # If previous if condition is not 
  # True then its possible for one
  # of the node to be None while
  # other one is not None.
  if not p or not q:
    return False

  # Return False if the values 
  # do not match
  if p.val != q.val:
    return False

  #Traverse left and right node and return
  return same_tree(p.left, q.left) and same_tree(p.right, q.right)
     
```
