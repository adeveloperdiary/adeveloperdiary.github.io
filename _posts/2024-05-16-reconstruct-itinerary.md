---
title: Backtracking - Reconstruct Itinerary
categories: [algorithm, backtracking]
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

You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

- For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

**Example 1:**

![itinerary1-graph](../assets/img/itinerary1-graph.jpeg)

```
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
```

**Example 2:**

![itinerary1-graph](../assets/img/itinerary2-graph.jpeg)

```
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.
```

## Solution

The generic solution expected here is not a Hard one however the generic `dfs()` solution with backtracking will get TimeOut Error in LeetCode. Unless you memorize, it won't be easy to find the best solution which will pass LeetCode in 15-20mins.

The question completes the language by mentioning **lexical ordering**, which is actually the easier part to implement. Let's start without this requirement and add it once we have basic solution in place.

Like other graph problem, we need the adjacency list.

```python
adjacency_list= collections.defaultdict(list)
for src,dest in tickets:
  adjacency_list[src].append(dest)
```

We need an `output` array to store the itinerary.

```python
output = []
```

 Now start the `dfs()`, it takes the `src` as the input and traverse through all possible neighbors based on the `adjacency_list`. When the length of `output` matches with the `tickets` we know that we were able to traverse through all the `tickets`.

We have to add `+1` as we are stating from `JFK` and we are going to always have one less ticket than the number of destinations. 

```python
def dfs(src):
  if len(output)==len(tickets)+1:
    return True
```

In case at some point we find that a node is not traversable, we can return `False`

```python
if src not in adjacency_list:
  return False
```

We will use **template 2** that we have already discussed  [here](https://adeveloperdiary.com/algorithm/backtracking/combination-sum/).

![image-20240514221758079](../assets/img/image-20240514221758079.jpg)

This is kinda straight forward, we need to get all the neighbors of the current node. Then `pop()` is out from the `adjacency_list` so that we don't travel in cycles. Then add it to the `output` and recursively call `dfs()` by providing the `dest`.

If anytime the `dfs()` returns `True`, we return `True`. Otherwise the path was not successful and we backtrack from the current `dest` and add it back to the `adjacency_list`.

```python
for index, dest in enumerate(adjacency_list):
  adjacency_list[src].pop(index)
	outout.append(dest)
  
  if dsf(dest):
    return True
  
  adjacency_list[src].insert(index, dest)
  outout.pop()
```

Finally, call `dfs()` by passing `JSK` as the first node and return `outout`.

```python
dfs('JFK')
return output
```

If you have understood so far, then incorporating the  **lexical ordering** is very straightforward. During an interview, you can just sort the neighbors before traversing. This would be the most natural way of solving this. 

```python
neighbors=adj[src]
neighbors.sort()
for i, dst in enumerate(neighbors):
  ...
```

You can also then think about pre-sorting the `tickets` array so that the sorting is done only once. 

```python
tickets.sort()
adjacency_list= collections.defaultdict(list)
for src,dest in tickets:
  ...

```

## Final Code

Here is the full code.

```python
def find_itinerary(tickets):
  tickets.sort()
  
  adjacency_list= collections.defaultdict(list)
	for src,dest in tickets:
  	adjacency_list[src].append(dest)
  
  output = []
  
  def dfs(src):
    if len(output)==len(tickets) + 1:
      return True
    
    if src not in adjacency_list:
		  return False
    
    for index, dest in enumerate(adjacency_list):
      adjacency_list[src].pop(index)
      outout.append(dest)

      if dsf(dest):
        return True

      adjacency_list[src].insert(index, dest)
      outout.pop()
      
	dfs('JFK')
	return output
```
