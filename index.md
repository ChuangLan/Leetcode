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

### 1. Two Sum

**Description**
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

**Ideas**
1. Sort + Two Pointer - Time: o(nlogn); Space: o(1)
2. HashMap - Time: o(n); Space: o(n) 


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

