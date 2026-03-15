# 字节跳动面试高频 Top 30 算法题

> 基于 CodeTop 面经数据统计，按频次排序。每题包含原始题面、解题思路、标准代码。

---

## #1 [LC 3] 无重复字符的最长子串 | Medium | 频次 642

### 题目

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1：**
```
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2：**
```
输入: s = "bbbbb"
输出: 1
```

**示例 3：**
```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
```

### 解答

滑动窗口 + 哈希集合。右指针不断扩展窗口，当遇到重复字符时，左指针收缩直到窗口内无重复。每次更新最大长度。

### 代码

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
# 时间 O(n)，空间 O(min(n, 字符集大小))
```

---

## #2 [LC 206] 反转链表 | Easy | 频次 575

### 题目

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

**示例：**
```
输入: head = [1,2,3,4,5]
输出: [5,4,3,2,1]
```

### 解答

迭代法，维护三个指针 `prev`、`curr`、`nxt`。每步将 `curr.next` 指向 `prev`，然后三个指针各前进一步。

### 代码

```python
def reverseList(head: ListNode) -> ListNode:
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
# 时间 O(n)，空间 O(1)
```

---

## #3 [LC 146] LRU 缓存 | Medium | 频次 525

### 题目

请你设计并实现一个满足 LRU (最近最少使用) 缓存约束的数据结构。

实现 `LRUCache` 类：
- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value`；如果不存在，则向缓存中插入该组 `key-value`。如果插入操作导致关键字数量超过 `capacity`，则应该 **逐出** 最久未使用的关键字。

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

**示例：**
```
输入:
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出: [null, null, null, 1, null, -1, null, -1, 3, 4]
```

### 解答

哈希表 + 双向链表。哈希表提供 O(1) 查找，双向链表维护使用顺序（头部最近使用，尾部最久未使用）。用 dummy head/tail 简化边界。Node 里存 key 是因为淘汰尾部节点时需要从哈希表中删除对应的 key。

### 代码

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
# 时间 get/put 均 O(1)，空间 O(capacity)
```

---

## #4 [LC 215] 数组中的第K个最大元素 | Medium | 频次 420

### 题目

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

**示例 1：**
```
输入: nums = [3,2,1,5,6,4], k = 2
输出: 5
```

**示例 2：**
```
输入: nums = [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

### 解答

**解法1（小顶堆）：** 维护大小为 k 的小顶堆，遍历数组时如果当前值大于堆顶就替换，最终堆顶就是第 k 大。

**解法2（快速选择，面试加分）：** 类似快排的 partition，每次确定一个元素的最终位置，只需要递归其中一半，平均 O(n)。

### 代码

```python
# 解法1: 小顶堆 O(n*logk)
def findKthLargest(nums, k):
    import heapq
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

# 解法2: 快速选择 O(n) 平均
def findKthLargest_quickselect(nums, k):
    import random
    target = len(nums) - k

    def partition(left, right):
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]
        store = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store], nums[i] = nums[i], nums[store]
                store += 1
        nums[store], nums[right] = nums[right], nums[store]
        return store

    left, right = 0, len(nums) - 1
    while True:
        pos = partition(left, right)
        if pos == target:
            return nums[pos]
        elif pos < target:
            left = pos + 1
        else:
            right = pos - 1
```

---

## #5 [LC 25] K 个一组翻转链表 | Hard | 频次 332

### 题目

给你链表的头节点 `head`，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

**示例 1：**
```
输入: head = [1,2,3,4,5], k = 2
输出: [2,1,4,3,5]
```

**示例 2：**
```
输入: head = [1,2,3,4,5], k = 3
输出: [3,2,1,4,5]
```

### 解答

1. 先检查剩余节点是否 >= k 个，不够就直接返回。
2. 翻转前 k 个节点（复用 LC 206 的逻辑）。
3. 原来的 head 变成了这一段的尾部，将其 next 指向递归处理后续部分的结果。

### 代码

```python
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    # 检查是否有 k 个节点
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1
    if count < k:
        return head

    # 翻转前 k 个
    prev = None
    curr = head
    for _ in range(k):
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    # head 现在是翻转后的尾部，连接后续递归结果
    head.next = reverseKGroup(curr, k)
    return prev
# 时间 O(n)，空间 O(n/k) 递归栈
```

---

## #6 [LC 15] 三数之和 | Medium | 频次 310

### 题目

给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。请你返回所有和为 `0` 且不重复的三元组。

**示例：**
```
输入: nums = [-1,0,1,2,-1,-4]
输出: [[-1,-1,2],[-1,0,1]]
```

### 解答

排序 + 固定一个数 + 双指针。先排序，然后遍历数组固定第一个数 `nums[i]`，用双指针 `left`、`right` 从两端向中间找另外两个数使三数之和为 0。注意跳过重复元素。

### 代码

```python
def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
# 时间 O(n²)，空间 O(1)
```

---

## #7 [LC 53] 最大子数组和 | Medium | 频次 261

### 题目

给你一个整数数组 `nums`，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例 1：**
```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**示例 2：**
```
输入: nums = [5,4,-1,7,8]
输出: 23
```

### 解答

Kadane 算法。维护 `curr_sum` 表示以当前元素结尾的最大子数组和。如果 `curr_sum` 加上当前元素还不如当前元素本身大，就从当前元素重新开始。每步更新全局最大值。

### 代码

```python
def maxSubArray(nums):
    curr_sum = max_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
# 时间 O(n)，空间 O(1)
```

---

## #8 [LC 21] 合并两个有序链表 | Easy | 频次 242

### 题目

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入: l1 = [1,2,4], l2 = [1,3,4]
输出: [1,1,2,3,4,4]
```

### 解答

创建 dummy 头节点，逐个比较两个链表的当前节点，较小的接到结果链表后面。最后把未遍历完的链表直接接上。

### 代码

```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2
    return dummy.next
# 时间 O(n+m)，空间 O(1)
```

---

## #9 [LC 1] 两数之和 | Easy | 频次 232

### 题目

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** `target` 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

**示例：**
```
输入: nums = [2,7,11,15], target = 9
输出: [0,1]
解释: 因为 nums[0] + nums[1] == 9 ，返回 [0, 1]。
```

### 解答

哈希表。遍历数组，对每个元素计算 `complement = target - num`，在哈希表中查找是否存在。如果存在，返回两个下标；否则将当前元素存入哈希表。

### 代码

```python
def twoSum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []
# 时间 O(n)，空间 O(n)
```

---

## #10 [LC 5] 最长回文子串 | Medium | 频次 222

### 题目

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

**示例 1：**
```
输入: s = "babad"
输出: "bab"
解释: "aba" 同样是符合题意的答案。
```

**示例 2：**
```
输入: s = "cbbd"
输出: "bb"
```

### 解答

中心扩展法。遍历每个字符作为回文中心，分别考虑奇数长度（以单个字符为中心）和偶数长度（以两个相邻字符为中心），向两边扩展直到不再是回文。

### 代码

```python
def longestPalindrome(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result
# 时间 O(n²)，空间 O(1)
```

---

## #11 [LC 33] 搜索旋转排序数组 | Medium | 频次 220

### 题目

整数数组 `nums` 按升序排列，数组中的值 **互不相同**。在传递给函数之前，`nums` 在预先未知的某个下标 `k` 上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`。

给你旋转后的数组 `nums` 和一个整数 `target`，如果 `nums` 中存在这个目标值，则返回它的下标，否则返回 `-1`。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

**示例：**
```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
```

### 解答

二分查找。每次将数组分为两半，至少有一半是有序的。先判断哪半段有序，再判断 target 是否在有序的那半段中，从而决定搜索方向。

### 代码

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:  # 左半段有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半段有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
# 时间 O(logn)，空间 O(1)
```

---

## #12 [LC 102] 二叉树的层序遍历 | Medium | 频次 220

### 题目

给你二叉树的根节点 `root`，返回其节点值的 **层序遍历**（即逐层地，从左到右访问所有节点）。

**示例：**
```
输入: root = [3,9,20,null,null,15,7]
输出: [[3],[9,20],[15,7]]
```

### 解答

BFS。用队列逐层处理，每层开始前记录队列长度，弹出该层所有节点并将其子节点入队。

### 代码

```python
def levelOrder(root):
    from collections import deque
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
# 时间 O(n)，空间 O(n)
```

---

## #13 [LC 121] 买卖股票的最佳时机 | Easy | 频次 211

### 题目

给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

如果你不能获取任何利润，返回 `0`。

**示例 1：**
```
输入: prices = [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（价格 = 1）买入，在第 5 天（价格 = 6）卖出，利润 = 6-1 = 5。
```

**示例 2：**
```
输入: prices = [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

### 解答

一次遍历，维护到目前为止的最低价 `min_price`，对每个价格计算 `price - min_price`，取最大值。

### 代码

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
# 时间 O(n)，空间 O(1)
```

---

## #14 [LC 200] 岛屿数量 | Medium | 频次 211

### 题目

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或垂直方向上相邻的陆地连接形成。

**示例：**
```
输入: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出: 3
```

### 解答

遍历网格，遇到 `'1'` 就计数 +1，然后 DFS 将整个连通的岛全部标记为 `'0'`，避免重复计数。

### 代码

```python
def numIslands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
# 时间 O(m*n)，空间 O(m*n)
```

---

## #15 [LC 141] 环形链表 | Easy | 频次 210

### 题目

给你一个链表的头节点 `head`，判断链表中是否有环。

如果链表中存在环，则返回 `true`；否则，返回 `false`。

### 解答

快慢指针。慢指针每次走一步，快指针每次走两步。如果有环，快指针一定会追上慢指针；如果无环，快指针会先到达 null。

### 代码

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
# 时间 O(n)，空间 O(1)
```

---

## #16 [LC 20] 有效的括号 | Easy | 频次 207

### 题目

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：左括号必须用相同类型的右括号闭合；左括号必须以正确的顺序闭合；每个右括号都有一个对应的相同类型的左括号。

**示例：**
```
输入: s = "()[]{}"
输出: true

输入: s = "(]"
输出: false
```

### 解答

用栈。遇到左括号就压栈，遇到右括号就检查栈顶是否是对应的左括号。最后栈为空则有效。

### 代码

```python
def isValid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return len(stack) == 0
# 时间 O(n)，空间 O(n)
```

---

## #17 [LC 88] 合并两个有序数组 | Easy | 频次 204

### 题目

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n`，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你合并 `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0`，应忽略。

**示例：**
```
输入: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出: [1,2,2,3,5,6]
```

### 解答

从后往前填充，避免覆盖 nums1 中还未处理的元素。三个指针分别指向 nums1 有效部分末尾、nums2 末尾和填充位置。

### 代码

```python
def merge(nums1, m, nums2, n):
    p1, p2, p = m - 1, n - 1, m + n - 1
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    nums1[:p2 + 1] = nums2[:p2 + 1]
# 时间 O(m+n)，空间 O(1)
```

---

## #18 [LC 46] 全排列 | Medium | 频次 202

### 题目

给定一个不含重复数字的数组 `nums`，返回其 **所有可能的全排列**。你可以 **按任意顺序** 返回答案。

**示例：**
```
输入: nums = [1,2,3]
输出: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

### 解答

回溯法。维护当前路径 `path` 和剩余可选元素 `remaining`，每次选一个元素加入路径并递归，递归返回后撤销选择。

### 代码

```python
def permute(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result
# 时间 O(n! * n)，空间 O(n)
```

---

## #19 [LC 103] 二叉树的锯齿形层序遍历 | Medium | 频次 200

### 题目

给你二叉树的根节点 `root`，返回其节点值的 **锯齿形层序遍历**（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

**示例：**
```
输入: root = [3,9,20,null,null,15,7]
输出: [[3],[20,9],[15,7]]
```

### 解答

在普通层序遍历（LC 102）基础上，用一个布尔变量控制方向，偶数层将该层结果反转。

### 代码

```python
def zigzagLevelOrder(root):
    from collections import deque
    if not root:
        return []
    result = []
    queue = deque([root])
    left_to_right = True
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right
    return result
# 时间 O(n)，空间 O(n)
```

---

## #20 [LC 236] 二叉树的最近公共祖先 | Medium | 频次 199

### 题目

给定一个二叉树，找到该树中两个指定节点的最近公共祖先。

最近公共祖先的定义为：对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。

**示例：**
```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
```

### 解答

递归。如果当前节点是 null 或等于 p 或 q，直接返回当前节点。分别递归左右子树，如果左右两边都找到了（都不为 null），说明当前节点就是 LCA；否则返回不为 null 的那一边。

### 代码

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right
# 时间 O(n)，空间 O(h)
```

---

## #21 [LC 54] 螺旋矩阵 | Medium | 频次 186

### 题目

给你一个 `m` 行 `n` 列的矩阵 `matrix`，请按照 **顺时针螺旋顺序**，返回矩阵中的所有元素。

**示例：**
```
输入: matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出: [1,2,3,6,9,8,7,4,5]
```

### 解答

维护上下左右四个边界 `top`、`bottom`、`left`、`right`，按照 →↓←↑ 的顺序逐层收缩遍历。每走完一个方向就收缩对应的边界。

### 代码

```python
def spiralOrder(matrix):
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result
# 时间 O(m*n)，空间 O(1)
```

---

## #22 [LC 92] 反转链表 II | Medium | 频次 184

### 题目

给你单链表的头指针 `head` 和两个整数 `left` 和 `right`，其中 `left <= right`。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 **反转后的链表**。

**示例：**
```
输入: head = [1,2,3,4,5], left = 2, right = 4
输出: [1,4,3,2,5]
```

### 解答

先用 dummy 节点简化头部处理。走到 `left` 的前一个节点 `prev`，然后用"头插法"逐个将后面的节点插到 `prev` 后面，循环 `right - left` 次。

### 代码

```python
def reverseBetween(head, left, right):
    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    curr = prev.next
    for _ in range(right - left):
        nxt = curr.next
        curr.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next
# 时间 O(n)，空间 O(1)
```

---

## #23 [LC 160] 相交链表 | Easy | 频次 180+

### 题目

给你两个单链表的头节点 `headA` 和 `headB`，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null`。

### 解答

双指针。指针 A 从 headA 出发，走完后从 headB 继续；指针 B 从 headB 出发，走完后从 headA 继续。两者走的总长度相同，如果有交点一定会同时到达；如果没有交点会同时到达 null。

### 代码

```python
def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a
# 时间 O(n+m)，空间 O(1)
```

---

## #24 [LC 42] 接雨水 | Hard | 频次 180+

### 题目

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例：**
```
输入: height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

### 解答

双指针。左右各维护一个最大高度 `left_max` 和 `right_max`。较小的一侧可以确定能接多少水（由较矮的一侧决定水位），移动较小一侧的指针。

### 代码

```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
# 时间 O(n)，空间 O(1)
```

---

## #25 [LC 56] 合并区间 | Medium | 频次 180+

### 题目

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]`。请你合并所有重叠的区间，并返回一个不重叠的区间数组。

**示例：**
```
输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6]。
```

### 解答

按起点排序后，逐个检查当前区间的起点是否 <= 上一个区间的终点。如果是则合并（取终点的较大值），否则开始新区间。

### 代码

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
# 时间 O(n*logn)，空间 O(n)
```

---

## #26 [LC 142] 环形链表 II | Medium | 频次 175+

### 题目

给定一个链表的头节点 `head`，返回链表开始入环的第一个节点。如果链表无环，则返回 `null`。

### 解答

快慢指针找到相遇点后，再用两个指针：一个从链表头出发，一个从相遇点出发，都每次走一步，再次相遇的位置就是入环点。

数学推导：设头到入环点距离 a，入环点到相遇点 b，环长 c。快指针走 `a+b+c`，慢指针走 `a+b`。因为快 = 2 \* 慢，所以 `a+b+c = 2(a+b)`，得 `c = a+b`，即 `a = c-b`。从头走 a 步和从相遇点走 c-b 步都恰好到达入环点。

### 代码

```python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None
# 时间 O(n)，空间 O(1)
```

---

## #27 [LC 199] 二叉树的右视图 | Medium | 频次 170+

### 题目

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例：**
```
输入: root = [1,2,3,null,5,null,4]
输出: [1,3,4]
```

### 解答

DFS，先递归右子树再递归左子树。每层第一个被访问到的节点就是右视图看到的节点。用 `depth == len(ans)` 判断该层是否已有节点被记录。

### 代码

```python
def rightSideView(root):
    ans = []
    def dfs(node, depth):
        if not node:
            return
        if depth == len(ans):
            ans.append(node.val)
        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1)
    dfs(root, 0)
    return ans
# 时间 O(n)，空间 O(h)
```

---

## #28 [LC 300] 最长递增子序列 | Medium | 频次 165+

### 题目

给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

**示例 1：**
```
输入: nums = [10,9,2,5,3,7,101,18]
输出: 4
解释: 最长递增子序列是 [2,3,7,101]，因此长度为 4。
```

**示例 2：**
```
输入: nums = [0,1,0,3,2,3]
输出: 4
```

### 解答

**解法1（DP）：** `dp[i]` 表示以 `nums[i]` 结尾的 LIS 长度。对每个 `i`，遍历 `j < i`，如果 `nums[j] < nums[i]` 则 `dp[i] = max(dp[i], dp[j]+1)`。O(n²)。

**解法2（贪心+二分，面试加分）：** 维护 `tails` 数组，`tails[i]` 是长度为 `i+1` 的递增子序列的最小末尾元素。对每个新元素，用二分查找确定插入位置。O(n*logn)。

### 代码

```python
# 解法1: DP O(n²)
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 解法2: 贪心+二分 O(n*logn)
def lengthOfLIS_bs(nums):
    import bisect
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

---

## #29 [LC 23] 合并K个升序链表 | Hard | 频次 160+

### 题目

给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例：**
```
输入: lists = [[1,4,5],[1,3,4],[2,6]]
输出: [1,1,2,3,4,4,5,6]
```

### 解答

小顶堆。将每个链表的头节点放入堆中，每次取出最小的节点接到结果链表后面，如果该节点有 next 就把 next 放入堆中。堆中始终最多 k 个元素。

### 代码

```python
def mergeKLists(lists):
    import heapq
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    curr = dummy
    while heap:
        val, idx, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
# 时间 O(N*logk)，N 为总节点数，k 为链表数。空间 O(k)
```

---

## #30 [LC 415] 字符串相加 | Easy | 频次 155+

### 题目

给定两个字符串形式的非负整数 `num1` 和 `num2`，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 `BigInteger`），也不能直接将输入的字符串转换为整数形式。

**示例 1：**
```
输入: num1 = "11", num2 = "123"
输出: "134"
```

**示例 2：**
```
输入: num1 = "456", num2 = "77"
输出: "533"
```

### 解答

模拟竖式加法。从两个字符串的末尾开始逐位相加，处理进位 `carry`。直到两个字符串都遍历完且进位为 0。

### 代码

```python
def addStrings(num1, num2):
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = []
    while i >= 0 or j >= 0 or carry:
        x = int(num1[i]) if i >= 0 else 0
        y = int(num2[j]) if j >= 0 else 0
        total = x + y + carry
        result.append(str(total % 10))
        carry = total // 10
        i -= 1
        j -= 1
    return ''.join(reversed(result))
# 时间 O(max(m,n))，空间 O(max(m,n))
```

---

## 刷题优先级总结

| 优先级 | 题目 | 建议 |
|:---:|:---|:---|
| P0 | LC 3, 206, 146, 215, 25, 15, 42 | 闭眼写出来，bug free |
| P1 | LC 53, 21, 1, 5, 33, 102, 121, 200, 141, 20 | 熟练掌握，5分钟内写完 |
| P2 | LC 88, 46, 103, 236, 54, 92, 160, 142, 199, 300, 56, 23, 415 | 理解思路，能写出来即可 |
