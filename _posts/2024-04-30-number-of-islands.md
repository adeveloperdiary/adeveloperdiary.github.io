---
title: Tree - Number of Islands
categories: [algorithm, graph]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> BFS
{: .prompt-info }

## Problem

Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return *the number of islands*. An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Example 1:**

```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**

```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

## Solution

This is a very simple BFS solution. We need to start at every cell of the grid (if its already not visited), then run bfs and flag each visited cells.

Start by defining the required variables.The `directions` variable is for traversing in 4 directions.

```python
ROWS, COLS =len(grid), len(grid[0])
visited = set()
directions = [[0,1], [0,-1], [1, 0], [-1, 0]]
```

Now define the `bfs()` function. Very important is to add to `visited` right after inserting into the `queue`.

```python
def bfs(row, col):
  queue = collections.deque()
	queue.append((row, col))
  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nei_row, nei_col = row+dr, col+dc
      
      if nei_row < 0 or nei_row == ROWS or nei_col < 0 or nei_col == COLS:
        or (nei_row,nei_col) in visited or grid[nei_row][nei_col]=='0':
          continue
      visited.add((nei_row, nei_col))
      queue.append((nei_row, nei_col))
```

Once the `bfs()` function is completed, we need to run it for all the cells.

```python
result = 0
for row in range(ROWS):
  for col in range(COLS):
    if (row,col) not in visited and grid[row, col]=='1':
      bfs(row,col)
      result+=1
return result
```

## Final Code

Here is the full code.

```python
def num_islands(grid) -> int:
    ROWS, COLS = len(grid), len(grid[0])
    visited = set()
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def bfs(row, col):
        queue = collections.deque()
        queue.append((row, col))
        
        while queue:               
            row, col = queue.popleft()
                
            for dr, dc in directions:
                r = row+dr
                c = col+dc

                if r < 0 or r == ROWS or c < 0 or c == COLS or (r, c) in visited or grid[r][c] == '0':
                    continue
                
                queue.append((r, c))
                visited.add((r, c))
    
    result = 0
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in visited and grid[r][c] == '1':
                bfs(r, c)
                result += 1

    return result

```



