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
3. 如果有 ^ 操作，还要再存一个数字mem2

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
