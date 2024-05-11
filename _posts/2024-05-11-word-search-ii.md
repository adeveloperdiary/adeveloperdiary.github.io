---
title: Backtracking - Word Search II
categories: [algorithm, matrices]
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

Given an `m x n` `board` of characters and a list of strings `words`, return *all words on the board*.

Each word must be constructed from letters of sequentially adjacent cells, where **adjacent cells** are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

**Example 1:**

![queens](../assets/img/search1.jpeg)

```
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```

**Example 2:**

![queens](../assets/img/search2.jpeg)

```
Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
```

## Solution

This is same as Word Search.

## Final Code

Here is the full code.

```python
def word_search(board, word):
    ROWS, COLS = len(board), len(board[0])
    path_visited = set()

    def dfs(row, col, char_index):
        if char_index == len(word):
            return True

        if row < 0 or col < 0 or row == ROWS or col == COLS or (row, col) in path_visited or word[char_index] != board[row][col]:
            return False

        path_visited.add((row, col))

        found = dfs(row+1, col, char_index+1) or dfs(row-1, col, char_index +
                                                     1) or dfs(row, col+1, char_index+1) or dfs(row, col-1, char_index+1)

        path_visited.remove((row, col))
        return found

    for row in range(ROWS):
        for col in range(COLS):
            if dfs(row, col, 0):
                return True

    return False
```







