---
title: Tree - Course Schedule II
categories: [algorithm, graph]
tags: [datastructure]
hidden: true
mermaid: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Medium
{: .prompt-warning }

> Backtracking, DFS
{: .prompt-info }

## Problem

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.

- For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return *the ordering of courses you should take to finish all courses*. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
```

**Example 2:**

```
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
```

**Example 3:**

```
Input: numCourses = 1, prerequisites = []
Output: [0]
```

## Solution

This is a much simpler problem to solve, if you have already solve the first one. The only main change is to track `cycle` and `visited` separately. This way whenever there is a cycle detected we return `False`, which will return back `[]`. If we have already visited a node, we simply return `True` as we do not have to add (and traverse) it again as its already part of the `result` array.

Same code for creating the `adjacency_list`.

```python
adjacency_list = collections.defaultdict(list)
    for course, prerequisite in prerequisites:
        adjacency_list[course].append(prerequisite)
```

Add three separate variables as discussed earlier.

```python
result = []
visited = set()
cycle = set()
```

Inside `dfs()`, return `False` if cycle is detected. 

```python
if course in cycle:
  return False
```

Return `True` :fire: if the node was already visited.

```python
if course in visited:
  return True
```

 Add the node to both `cycle` and `visited`.

```python
cycle.add(course)
visited.add(course)
```

Traverse the `prerequisite` (Same code).

```python
for prerequisite in adjacency_list[course]:
  if not dfs(prerequisite):
    return False
```

Remove from `cycle`, like we removed from `visited` in last example. This is the backtracking part.

```python
cycle.remove(course)
```

Add the node to the `result` array and return `True`.

```python
result.append(course)
return True
```

Finally, traverse through all the nodes and return `[]` if any time the `dfs()` returns `False`. Finally return the `result` array.

```python
for course in range(num_courses):
  if not dfs(course):
    return []

return result
```

## Final Code

Here is the full code.

```python
def can_finish(num_courses, prerequisites):
    adjacency_list = collections.defaultdict(list)
    for course, prerequisite in prerequisites:
        adjacency_list[course].append(prerequisite)

    result = []
    visited = set()
    cycle = set()

    def dfs(course):
        if course in cycle:
            return False

        if course in visited:
            return True

        cycle.add(course)
        visited.add(course)

        for prerequisite in adjacency_list[course]:
            if not dfs(prerequisite):
                return False

        cycle.remove(course)
        result.append(course)
        return True

    for course in range(num_courses):
        if not dfs(course):
            return []

    return result
```



