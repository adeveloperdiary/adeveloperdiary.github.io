---
title: Tree - Validate Binary Search Tree
categories: [algorithm, tree]
tags: [datastructure]
hidden: true
mermaid: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> PreOrder DFS, Set Min and Max Boundary
{: .prompt-info }

## Problem

Given the `root` of a binary tree, *determine if it is a valid binary search tree (BST)*. 

A **valid BST** is defined as follows:

- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

**Example 1:**

<img src="../assets/img/tree1.jpeg" alt="addtwonumber1"  />

```
Input: root = [2,1,3]
Output: true
```

**Example 2:**

![btree](../assets/img/tree2.jpeg)

```
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
```

## Solution

We will be solving this using PreOrder DFS. So the struture of the code will be.

```mermaid
flowchart LR
    A["Base Condition"]-->B["Validate Root Node"]-->C["Traverse Left and Right Nodes"]

```

### Base Condition

The base condition is all `None` nodes are valid, so need to return `True`. 

```python
if not root:
  return True
```

### Validate Root Node

Now based on the condition of the **valid BST** above,we need the boundaries for each node. As long as the node value is with in the boundary, its valid. For any left subtree, current root node value is the max value and for right substree the current root node value is the min value. So we need to pass this `min_val` and `max_val` to our `dfs()` function.

```python
def dfs(root, min_val, max_val):
  if not root:
    return True
  
  if not (root.val > min_val and root.val < max_val):
    return False
```

### Traverse

Now traverse through `left` and `right` children. The only logic here to set the `max_val` and `min_val` accordingly. In case any one children returns `False` this function call should return `False` as well.

```python
  return dfs(root.left, min_val, root.val) and dfs(root.right, root.val, max_val)
```

Finally we can call the `dfs()` function for the first time by passing the `root` node and `[-inf, +inf]` as the `min_val` and `max_val`

```python
return dfs(root, float('-inf'), float('inf'))
```



## Visualization

Here is the visualization of the problem.



<img src="../assets/img/image-20240411203833410.png" alt="image-20240411203833410" style="zoom: 50%;" />

## Final Code

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def is_valid_bst(root:TreeNode):
  def dfs(root, min_val, max_val):
    if not root:
      return True
    
    if not (root.val > min_val and root.val < max_val):
      return False
    
    return dfs(root.left, min_val, root.val) and dfs(root.right, root.val, max_val)
    
  return dfs(root, float('-inf'),float('inf'))  
  
  
  

```

