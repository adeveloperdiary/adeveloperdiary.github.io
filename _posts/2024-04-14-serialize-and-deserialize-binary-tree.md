---
title: Tree - Serialize and Deserialize Binary Tree
categories: [algorithm, tree]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> Recursion
{: .prompt-info }

## Problem

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example 1:**

<img src="../assets/img/serdeser.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
```

## Solution

### Serialize

The solution for `serialize` is very simple. We will perform PreOrder traversal and save the values in an array. The only important part is to append `NONE` for all the `left` and `right` child for leaf node. 

```python
def serialize(root):
  result = []
  def dfs(root):
    if not root:
      result.append("NONE")
      return
    
    result.append(str(root.val))
    
    dfs(root.left)
    dfs(root.right)
  print(result)
  return result  
```

The output is interesting to understand. The diagram of the tree is given above in **Example 1**. 

```
['1', '2', 'NONE', 'NONE', '3', '4', 'NONE', 'NONE', '5', 'NONE', 'NONE']
```

### Deserialize

The only important logic for `deserialize` is the base case. Once we see `NONE`, we will assume that it's the end and return `None`. Then move the pointer by `1`

```python
def deserialize(result_arr):
  pointer = 0
  
  def dfs():
    nonlocal pointer
    if result_arr[pointer] == 'NONE':
      pointer+=1
      return None
```

Create the `root` node. Increment the `pointer`.

```python
    root = TreeNode(val = int(result_arr[pointer]))
    pointer +=1
```

Create nodes for `left` and `right` sub-trees.

```python
    root.left = dfs()
    root.right = dfs()
    return root
  return dfs()
```

## Final Code 

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def serialize(root):
  result = []
  def dfs(root):
    if not root:
      result.append("NONE")
      return
    
    result.append(str(root.val))
    
    dfs(root.left)
    dfs(root.right)
  return result  

def deserialize(result_arr):
  pointer = 0
  
  def dfs():
    nonlocal pointer
    if result_arr[pointer] == 'NONE':
      pointer+=1
      return None
    root = TreeNode(val = int(result_arr[pointer]))
    pointer +=1
    root.left = dfs()
    root.right = dfs()
    return root
  return dfs()
```
