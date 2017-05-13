[Edit raw](https://github.com/ChuangLan/Leetcode/edit/master/index.md)

# Pocket Gems
## TODO: 210 (topological sort)

### 5. Longest Palindromic Substring

**Description**
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
Example:
```
Input: "babad"

Output: "bab"

Note: "aba" is also a valid answer.
```
Example:
```
Input: "cbbd"

Output: "bb"
```
**Ideas**

1. At each position, iterate double ways both in odd and even
2. Record the max len

**Tag:** DP POCKET GEMS

```
public class Solution {
    public String longestPalindrome(String s) {
        if(s == null || s.isEmpty()) return "";
        int start = 0, len = 1;
        for(int i = 0; i < s.length(); i++){
            // odd
            int j = 1;
            while (i-j >= 0 && i+j < s.length() && s.charAt(i-j) == s.charAt(i+j)) j++;
            if(2*(j-1)+1 > len){
                start = i-(j-1);
                len = 2*(j-1)+1;
            }
            //even
            j = 1;
            while(i-j+1 >= 0 && i+j < s.length() && s.charAt(i-j+1) == s.charAt(i+j)) j++;
            if(2*(j-1) > len){
                start = i-(j-1)+1;
                len = 2*(j-1);
            }
        }
        return s.substring(start, start + len);
    }  
}
```


### 152. Maximum Product Subarray

**Description**
Find the contiguous subarray within an array (containing at least one number) which has the largest product.

For example, given the array [2,3,-2,4],
the contiguous subarray [2,3] has the largest product = 6.
**Ideas**

1. Record the curtMax and curtMin as 2 Integers
2. In each iteration, update the curtMax and curtMin, by comparing (preMax * num, preMin * num, num)
3. Update the max if curtMax > max
4. return max

**Tag:** DP POCKET GEMS

```
public class Solution {
    public int maxProduct(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int curtPos = nums[0], curtNeg = nums[0];
        int max = nums[0];
        for(int i = 1; i < nums.length; i++){
            int prePos = curtPos, preNeg = curtNeg;
            curtPos = Math.max(Math.max(prePos*nums[i], preNeg*nums[i]), nums[i]);
            curtNeg = Math.min(Math.min(prePos*nums[i], preNeg*nums[i]), nums[i]);
            max = Math.max(max, curtPos);
        }
        return max;
    }
}
```


### 285. Inorder Successor in BST

**Description**
Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

Note: If the given node has no in-order successor in the tree, return null.

**Ideas**

1. 如果root.val > p.val，说明next在root左边或自己
2. 于是以root.left为root，找结果。如果null，自己就是下一个，如果有值，就返回那个
3. 如果root.val <= p.val，说明next在root右边

**Tag:** dfs recursion POCKET GEMS

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if(root == null || p == null) return null;
        if(p.val < root.val){
            TreeNode left = inorderSuccessor(root.left, p);
            return left == null ? root : left;
        }
        return inorderSuccessor(root.right, p);
    }
}
```


### 239. Sliding Window Maximum

**Description**
Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

For example,
Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.
```
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```
Therefore, return the max sliding window as [3,3,5,5,6,7].

**Ideas**

1. Use a deque to store the maximums
2. The first one in deque is the curt max
3. Check if the first one expires.
4. 从队尾开始，把比当前小的都poll出来
5. 把当前加进去
6. 把队首计入result

**Tag:** deque POCKET GEMS

```
public class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null || nums.length == 0 || k <= 0) return new int[0];
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for(int i = 0; i < nums.length; i++){
            if(deque.peekFirst() != null && deque.peekFirst() <= i - k) deque.pollFirst();
            while(deque.peekLast() != null && nums[deque.peekLast()] < nums[i]) deque.pollLast();
            deque.offerLast(i);
            if(i + 1 >= k) res[i + 1 - k] = nums[deque.peekFirst()];
        }
        return res;
    }
}
```



### 210. Course Schedule II

**Description**
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

For example:
```
2, [[1,0]]
```
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]
```
4, [[1,0],[2,0],[3,1],[3,2]]
```
There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].

**Ideas**

1. Init graph with arraylist
2. Add neighbors to the list
3. Topologically sort by dfs

**Tag:** stack POCKET GEMS

```
public class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        List[] graph = new List[n];
        // init the graph
        for(int i = 0; i < n; i++) graph[i] = new ArrayList<Integer>();
        // fill the graph with courses
        for(int i = 0; i < prerequisites.length; i++){
            int node = prerequisites[i][1], neighbor = prerequisites[i][0];
            graph[node].add(neighbor);
        }
        // external visit set
        boolean[] visit = new boolean[n];
        // dfs visit set
        boolean[] temp = new boolean[n];
        int[] res = new int[n];
        // topologically sort by dfs
        int idx = n - 1;
        for(int i = 0; i < n; i++) if(!visit[i]) idx = dfs(graph, res, visit, temp, idx, i);
        // idx == -2 means this hasCycle
        if(idx == -2) return new int[0];
        return res;
    }
    
    private int dfs(List[] graph, int[] res, boolean[] visit, boolean[] temp, int idx, int node){
        if(idx == -2 || temp[node]) return -2;
        if(visit[node]) return idx;
        visit[node] = true;
        temp[node] = true;
        List<Integer> neighbors = graph[node];
        for(Integer neighbor : neighbors){
            idx = dfs(graph, res, visit, temp, idx, neighbor);
            if(idx == -2) return idx;
        }
        temp[node] = false;
        res[idx] = node;
        return idx - 1;
    }
}
```



### NAN Ternary Expression to Binary Tree

**Description**
Convert a ternary expression to a binary tree.
Say:
a?b:c to
```
  a
 /  \
b   c
```
a?b?c:d:e to
```

     a
    / \
   b   e
  / \
 c   d
```

**Ideas**

1. Each time we scan two characters, 
2. the first character is either ? or :, 
3. the second character holds the value of the tree node. 
4. When we see ?, we add the new node to left.
5. When we see :, we need to find out the ancestor node that doesn't have a right node, and make the new node as its right child.

**Tag:** stack POCKET GEMS

```
public class Solution{
	public TreeNode convert(char[] expr) {
	  if (expr.length == 0) {
	    return null;
	  }

	  TreeNode root = new TreeNode(expr[0]);

	  Stack<TreeNode> stack = new Stack<>();
	  
	  stack.push(root);

	  for (int i = 1; i < expr.length; i += 2) {
	    TreeNode node = new TreeNode(expr[i + 1]);

	    if (expr[i] == '?') {
	      stack.peek().left = node;
	    }

	    if (expr[i] == ':') {
	      stack.pop();
	      while (stack.peek().right != null) {
		stack.pop();
	      }
	      stack.peek().right = node;
	    }

	    stack.push(node);
	  }

	  return root;
	}
}
```


### NAN Top View of Binary Tree 变种 of 314. Binary Tree Vertical Order Traversal

**Description**
314. Binary Tree Vertical Order Traversal
Given a binary tree, return the first one of vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
If two nodes are in the same row and column, the order should be from left to right.
Examples:

Given binary tree [3,9,20,null,null,15,7],
```
 3
  /\
 /  \
 9  20
    /\
   /  \
  15   7
```
return its vertical order traversal as:
```
[9, 3, 20, 7]
```

**Ideas**

1. 先用computeRange求出min and max
2. 之后再用bfs得到每个colum
3. 用Integer[] 中的null 来代表没有访问过

注： 如果是bottom view 也是bfs，但是保留最后一个，每次直接覆盖之前的就可以了

**Tag:**  bfs-only POCKET GEMS

```
public class Solution{
	int min = 0, max = 0;
    public List<Integer> topView(TreeNode root){
    	List<Integer> res = new ArrayList<>();
    	if(root == null) return res;
    	computeRange(root, 0);
    	Integer[] array = new Integer[max - min + 1];
    	Queue<TreeNode> queue = new LinkedList<>();
    	Queue<Integer> cols = new LinkedList<>();
        queue.offer(root);
        cols.offer(-min);
        while(queue.peek() != null){
        	TreeNode node = queue.poll();
        	int col = cols.poll();
        	if(array[col] == null) array[col] = node.val;
        	if(node.left != null){
                queue.offer(node.left);
                cols.offer(col - 1);
            }
            if(node.right != null){
                queue.offer(node.right);
                cols.offer(col + 1);
            }
        }
        for(Integer val: array){
        	res.add(val);
        }
    	return res;
    }

    private void computeRange(TreeNode root, int col){
        if(root == null) return;
        min = Math.min(min, col);
        max = Math.max(max, col);
        computeRange(root.left, col - 1);
        computeRange(root.right, col + 1);
    }
}
```


### 200. Number of Islands

**Description**

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
Example 1:
```
11110
11010
11000
00000
```
Answer: 1
Example 2:
```
11000
11000
00100
00011
```
Answer: 3

**Ideas**

1. DFS recursive解
2. 遍历过的地方设为1，然后再遍历其他地方

**Tag:**  bfs dfs union-find POCKET GEMS

```
public class Solution {
    public int numIslands(char[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0) return 0;
        int rows = grid.length, cols = grid[0].length;
        int res = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                res += helper(grid, i, j);
            }
        }
        return res;
    }
    
    private int helper(char[][]grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return 0;
        grid[i][j] = '0';
        // this four will delete all the other neighbors
        helper(grid, i-1, j);
        helper(grid, i, j-1);
        helper(grid, i+1, j);
        helper(grid, i, j+1);
        return 1;
    }
}
```

### 133. Clone Graph

**Description**

Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

```
 1
      / \
     /   \
    0 --- 2
         / \
         \_/

```

**Ideas**

1. DFS/BFS都能解
2. 基本思路就是用一个map的key-value来link新老node
3. 之后就是普通的搜索

**Tag:**  bfs dfs POCKET GEMS

```
/**
 * Definition for undirected graph.
 * class UndirectedGraphNode {
 *     int label;
 *     List<UndirectedGraphNode> neighbors;
 *     UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
 * };
 */
public class Solution {
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if(node == null) return null;
        Map<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
        // bfs
          Queue<UndirectedGraphNode> queue = new LinkedList<>();
          queue.offer(node);
          UndirectedGraphNode root = node;
          while(queue.peek() != null){
              node = queue.poll();
              UndirectedGraphNode clone = new UndirectedGraphNode(node.label);
              map.put(node, clone);
              for(UndirectedGraphNode neighbor : node.neighbors){
                  if(map.containsKey(neighbor)) continue;
                  queue.offer(neighbor);
              }
          }
          for(UndirectedGraphNode key : map.keySet()){
              UndirectedGraphNode value = map.get(key);
              for(UndirectedGraphNode neighbor : key.neighbors){
                  value.neighbors.add(map.get(neighbor));
              }
          }
          return map.get(root);
        // bfs end
        // return cloneHelper(node, map);
    }
    // DFS
    // public UndirectedGraphNode cloneHelper(UndirectedGraphNode node, Map<UndirectedGraphNode, UndirectedGraphNode> map){
    //     if(map.containsKey(node)) return map.get(node);
    //     UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
    //     map.put(node, newNode);
    //     for(UndirectedGraphNode neighbor : node.neighbors){
    //         UndirectedGraphNode temp = cloneHelper(neighbor, map);
    //         newNode.neighbors.add(temp);
    //     }
    //     return newNode;
    // }
}
```

### 139. Word Break

**Description**

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code".

**Ideas**

1. 用一个boolean array 代表s每个idx之前是否能被word break
2. 如果减掉当前word长度是true，并且substring和word一样，说明这个也是true
3. 最后return dp[ls]

**Tag:**  dp POCKET GEMS

```
public class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        if(s == null || wordDict == null) return false;
        int ls = s.length();
        boolean[] dp = new boolean[ls+1];
        dp[0] = true;
        for(int i = 1; i <= ls; i++){
            for(String word : wordDict){
                if(word.length() <= i && dp[i - word.length()] && word.equals(s.substring(i - word.length(), i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[ls];
    }
}
```
### 78. Subsets

**Description**

Given a set of distinct integers, nums, return all possible subsets.

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,3], a solution is:

```
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

```

**Ideas**

1. 用bit manipulation
2. 每一bit都代表这个元素是否在set里,如有，就加在这次的list里
3. subset的总数就是2^n个

**Tag:**  array backtracking bit manipulation POCKET GEMS

```
public class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        // use bit manipulation to compute the result
        // each bit is means
        int size = 1 << nums.length;
        List<List<Integer>> res = new ArrayList<>(size);
        for(int i = 0; i < size; i++){
            int mask = i, j = 0;
            List<Integer> temp = new ArrayList<>();
            while(mask > 0){
                if((mask & 1) == 1) temp.add(nums[j]);
                j++;
                mask >>= 1;
            }
            res.add(temp);
        }
        return res;
    }
}
```

### 90. Subsets II

**Description**

Given a collection of integers that might contain duplicates, nums, return all possible subsets.

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,2], a solution is:

```
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

**Ideas**

1. backtracking
2. 处理重复: 如果add这个数，那后面照常，如果不，那后面一样的数字都别想
3. 这样处理保证永远是第一个dup在使用


**Tag:**  array backtracking POCKET GEMS

```
public class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        recurse(res, temp, nums, 0);
        return res;
    }
    
    private void recurse(List<List<Integer>> res, List<Integer> temp, int[] nums, int idx){
        // exit
        if(idx == nums.length){
            List<Integer> clone = new ArrayList<>(temp);
            res.add(clone);
            return;
        }
        // to add or not to add
        // adding: backtracking
        temp.add(nums[idx]);
        recurse(res, temp, nums, ++idx);
        temp.remove(temp.size() - 1);
        // no adding
        while(idx < nums.length && nums[idx-1] == nums[idx]) idx++;
        recurse(res, temp, nums, idx);
    }
}
```


### 10. Regular Expression Matching

**Description**

Implement regular expression matching with support for '.' and '*'.

```
'.' Matches any single character.
'*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "a*") → true
isMatch("aa", ".*") → true
isMatch("ab", ".*") → true
isMatch("aab", "c*a*b") → true
```

**Ideas**

1. DP
2. 方法就是注释里面的状态转移方程


**Tag:**  DP POCKET GEMS

```
// 1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
// 2, If p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1];
// 3, If p.charAt(j) == '*': 
//   here are two sub conditions:
//               1   if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2]  //in this case, a* only counts as empty
//               2   if p.charAt(j-1) == s.charAt(i) or p.charAt(j-1) == '.':
//                               dp[i][j] = dp[i-1][j]    //in this case, a* counts as multiple a 
//                           or dp[i][j] = dp[i][j-1]   // in this case, a* counts as single a
//                           or dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty


public class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        // "" and "" match
        dp[0][0] = true;
        // assign all the "a*b*c*" match ""
        for(int j = 1; j < n; j++){
            if(p.charAt(j) == '*') dp[0][j+1] = dp[0][j-1];
        }
        // dp
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                int x = i+1, y = j+1;
                if(p.charAt(j) == s.charAt(i) || p.charAt(j) == '.') dp[x][y] = dp[x-1][y-1];
                if(p.charAt(j) == '*'){
                    if(j == 0) return false;
                    // check empty first
                    dp[x][y] = dp[x][y-2];
                    // then check whether it could be multiple
                    if(p.charAt(j-1) == s.charAt(i) || p.charAt(j-1) == '.') dp[x][y] = dp[x][y] || dp[x-1][y];
                }
            }
        }
        return dp[m][n];
    }
}
```
## Solutions

### 0. Template

**Description**

The description of this question

**Ideas**

Ideas, hints or keywords to the problem

**Tag:**  template

```
public class Solution {
    public void test(){
        System.out.println("Hello World!");
    }
}
```

### 1. Two Sum

**Description**
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

**Ideas**
1. Sort + Two Pointer - Time: o(nlogn); Space: o(1)
2. HashMap - Time: o(n); Space: o(n) 

**Tag:**  hashMap, two pointer

```
public class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        // if(nums == null || nums.length < 2) return res;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(nums[i])){
                res[0] = map.get(nums[i]);
                res[1] = i;
                break;
            }
            else map.put(target - nums[i], i);
        }
        return res;
    }
}
```

### 15. 3Sum

**Description**
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

**Ideas**
Sort + Two Pointer - Time: o(n^2); Space: o(1)

先sort，之后调用two sum (two pointers) 

**Tag:**  hashMap, two pointer

```
public class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(num);
        List<List<Integer>> res = new LinkedList<>(); 
        for (int i = 0; i < num.length-2; i++) {
            if (i == 0 || (i > 0 && num[i] != num[i-1])) {
                int lo = i+1, hi = num.length-1, sum = 0 - num[i];
                while (lo < hi) {
                    if (num[lo] + num[hi] == sum) {
                        res.add(Arrays.asList(num[i], num[lo], num[hi]));
                        while (lo < hi && num[lo] == num[lo+1]) lo++;
                        while (lo < hi && num[hi] == num[hi-1]) hi--;
                        lo++; hi--;
                    } else if (num[lo] + num[hi] < sum) lo++;
                    else hi--;
               }
            }
        }
        return res;
    }
}
```

### 224. Basic Calculator （加减括号）

**Description**

Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .

You may assume that the given expression is always valid.

Some examples:

```
"1 + 1" = 2
" 2-1 + 2 " = 3
"(1+(4+5+2)-3)+(6+8)" = 23
```

**Ideas**

用Stack，res = 0 当结果 sign = 1 当符号，遇到数字和），更新res；遇到加减更新sign，遇到（把当前存到stack里，reset res 和 sign

**Tag:**  stack math

```
public int calculate(String s) {
    Stack<Integer> stack = new Stack<>();
    int len = s.length(), res = 0, sign = 1;
    for(int i = 0; i < len; i++){
        if(Character.isDigit(s.charAt(i))){
            int sum = 0;
            while(i < len && Character.isDigit(s.charAt(i))){
                sum = sum * 10 + (s.charAt(i) - '0');
                i++;
            }
            res += sum * sign;
            if(i == len) break;
        }
        if(s.charAt(i) == '+') sign = 1;
        if(s.charAt(i) == '-') sign = -1;
        if(s.charAt(i) == '(') {
            stack.push(res);
            stack.push(sign);
            res = 0;
            sign = 1;
        }
        if(s.charAt(i) == ')') res = res * stack.pop() + stack.pop();
    }
    return res;
}
```
### 227. Basic Calculator II （加减乘除）

**Description**

Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . The integer division should truncate toward zero.

You may assume that the given expression is always valid.

Some examples:

```
"3+2*2" = 7
" 3/2 " = 1
" 3+5 / 2 " = 5
```

**Ideas**

利用的stack的思想，但是可以不用stack来implement

1. 遇到 + - 更新res；
2. 遇到 * / 更新mem1 // 相当于stack

**Tag:**  math string stack

```
public int calculate(String s) {
    if (s == null) return 0;
    s = s.trim().replaceAll(" +", "");
    int length = s.length();

    int res = 0;
    long preVal = 0; // initial preVal is 0
    char sign = '+'; // initial sign is +
    int i = 0;
    while (i < length) {
        long curVal = 0;
        while (i < length && (int)s.charAt(i) <= 57 && (int)s.charAt(i) >= 48) { // int
            curVal = curVal*10 + (s.charAt(i) - '0');
            i++;
        }
        if (sign == '+') {
            res += preVal;  // update res
            preVal = curVal;
        } else if (sign == '-') {
            res += preVal;  // update res
            preVal = -curVal;
        } else if (sign == '*') {
            preVal = preVal * curVal; // not update res, combine preVal & curVal and keep loop
        } else if (sign == '/') {
            preVal = preVal / curVal; // not update res, combine preVal & curVal and keep loop
        }
        if (i < length) { // getting new sign
            sign = s.charAt(i);
            i++;
        }
    }
    // final update
    res += preVal;
    return res;
}
```

### 467. Unique Substrings in Wraparound String 

**Description**

Consider the string s to be the infinite wraparound string of "abcdefghijklmnopqrstuvwxyz", so s will look like this: "...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".

Now we have another string p. Your job is to find out how many unique non-empty substrings of p are present in s. In particular, your input is the string p and you need to output the number of different non-empty substrings of p in the string s.

Note: p consists of only lowercase English letters and the size of p might be over 10000.

Some examples:

```
Input: "a"
Output: 1

Explanation: Only the substring "a" of string "a" is in the string s.

Input: "cac"
Output: 2
Explanation: There are two substrings "a", "c" of string "cac" in the string s.

Input: "zab"
Output: 6
Explanation: There are six substrings "z", "a", "b", "za", "ab", "zab" of string "zab" in the string s.

```

**Ideas**

1. 用一个size为26的DP数组来存以某字母开头的最长substring大小
2. for loop的时候，在每一段连续串结束之后，更新dp数组
3. 由于每一个字母开头的最长substring大小，就是以这个字母开头的所有子串数量，所以res = sum(dp)

**Tag:**  math string stack

```
public class Solution {
    public int findSubstringInWraproundString(String p) {
        if(p == null || p.length() == 0) return 0;
        // the number of the len starting from a to z
        int[] dp = new int[26];
        int preLen = 1;
        char last = p.charAt(0);
        dp[last - 'a'] = 1;
        // important i <= p.length(): make sure that the final one calulated
        for(int i = 1; i <= p.length(); i++){
            // ab, za
            while(i < p.length() && (p.charAt(i) - p.charAt(i-1) == 1 || p.charAt(i) - p.charAt(i-1) == -25)){
                // preLen plus 1
                preLen++;
                i++;
            }                            
            // update dp
            for(int j = 0; j < preLen; j++){
                int idx = (last - 'a' + j) % 26;
                dp[idx] = Math.max(dp[idx], preLen - j);
            }
            // update curt
            if(i < p.length()) {
                last = p.charAt(i);
                preLen = 1;
            }
        }
        int res = 0;
        for(int num : dp) res += num;
        return res;
    }
}
```
## Welcome to GitHub Pages

[View website](https://chuanglan.github.io/Leetcode/)

[View raw](https://github.com/ChuangLan/Leetcode/blob/master/index.md)

[Edit raw](https://github.com/ChuangLan/Leetcode/edit/master/index.md)

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
