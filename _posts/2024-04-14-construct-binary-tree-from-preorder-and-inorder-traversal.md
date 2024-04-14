---
title: Tree - Construct Binary Tree from PreOrder and InOrder Traversal
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

Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return *the binary tree*.

**Example 1:**

<img src="../assets/img/tree123.jpeg" alt="addtwonumber1" style="zoom:67%;" />

```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

**Example 2:**

```
Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

## Solution

- The first element of the **PreOrder** array is always the root node.
- Find the `root` in the **InOrder** array. Every value of the left of the array is going to be in the `left` subtree and every value at the right will be in the `right` subtree. 
-  

## Final Code 

Here is the full code.

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


     
```
