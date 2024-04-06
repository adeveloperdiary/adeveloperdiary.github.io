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

## Code

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
