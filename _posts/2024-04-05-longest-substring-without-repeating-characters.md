---
title: Sliding Window - Longest Substring Without Repeating Characters
categories: [algorithm, sliding-window]
tags: [datastructure]
hidden: true

---

> All diagrams presented herein are original creations, meticulously designed to enhance comprehension and recall. Crafting these aids required considerable effort, and I kindly request attribution if this content is reused elsewhere.
{: .prompt-danger }

> **Difficulty** :  Easy
{: .prompt-tip }

> Two Pointers , use set()
{: .prompt-info }


## Problem

Given a string `s`, find the length of the **longest** **substring** without repeating characters.

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

**Example 2:**

```
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Example 3:**

```
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

## Solution

1.	Store the values in a set() which will be our window. We can add from right and remove from left in the window.
1.	Keep moving right and add to set()
1.	If current char is already in the set() then keep removing from left and increment left pointer until the current char in removed from set()

I will first show a basic version and them optimize a bit. This is the code from the `for` loop where we increment the right `r` pointer.  If the char `c` is not in the map `char_map` add `c` to `char_map` then find if we have new `max_len`.

```python
if c not in char_map:
  char_map[c]=True
  if max_len < r+1-l:
    max_len=r+1-l
```

If `c` is already in `char_map`, keep moving left pointer `l` until `c` is removed from `char_map`.

```python
else:
  while c in char_map:
      del char_map[s[l]]
      l+=1
```

Finally add `c` back to `char_map`.

```python
	char_map[c]=True
```

Here is the code:

```python
def length_of_longest_substring(s: str) -> int:
    char_map = {}
    l = 0
    max_len = 0
    for r in range(len(s)):
        c = s[r]

        if c not in char_map:
            char_map[c] = True
            if max_len < r+1-l:
                max_len = r+1-l

        else:
            while c in char_map:
                del char_map[s[l]]
                l += 1
            char_map[c] = True

    return max_len
```

We really do not need the initial `if` condition as we can make sure `c` is not inside `char_map` when we add `c` and find new `max_len`. Here is the updated version, where we first run the while loop.

```python
def length_of_longest_substring(s: str) -> int:
    char_map = {}
    l = 0
    max_len = 0
    for r in range(len(s)):
        c = s[r]

        while c in char_map:
            del char_map[s[l]]
            l += 1
        char_map[c] = True

        if max_len < r+1-l:
            max_len = r+1-l
    return max_len
```

## Code

Here is the final version where few other lines were optimized, however the base code is still the same. 

```python
def length_of_longest_substring(s):
    # This will represent the window
    # A map can also be used
    char_set = set()
    l = 0
    result = 0

    for r in range(len(s)):

        while s[r] in char_set:
            # Keep deleting from left
            # until there is no r
            # in the window
            char_set.remove(s[l])
            l += 1
        # Keep adding r
        # Moving the window right
        char_set.add(s[r])

        # Keep track of max length
        result = max(result, (r+1)-l)
    return result
print(length_of_longest_substring("abcabcbb"))
```

```python
3
```

## Runtime Complexity

The runtime will be `O(n)` as we are simply scanning through the array max twice.
