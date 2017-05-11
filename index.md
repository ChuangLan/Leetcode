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
[Edit raw](https://github.com/ChuangLan/Leetcode/edit/master/index.md)
