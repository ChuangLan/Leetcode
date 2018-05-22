[Edit raw](https://github.com/ChuangLan/Leetcode/edit/master/index.md)

# Google
### Talk: Binary Indexed Tree

[Binary Indexed Tree](https://www.topcoder.com/community/data-science/data-science-tutorials/binary-indexed-trees/)

Get the last digit 1 in a binary integer. (Isolating the last digit)

NOTE: Instead of “the last non-zero digit,” it will write only “the last digit.”

There are times when we need to get just the last digit from a binary number, so we need an efficient way to do that. Let num be the integer whose last digit we want to isolate. In binary notation num can be represented as a1b, where a represents binary digits before the last digit and b represents zeroes after the last digit.

Integer -num is equal to (a1b)¯ + 1 = a¯0b¯ + 1. b consists of all zeroes, so b¯ consists of all ones. Finally we have

-num = (a1b)¯ + 1 = a¯0b¯ + 1 = a¯0(0…0)¯ + 1 = a¯0(1…1) + 1 = a¯1(0…0) = a¯1b.
Now, we can easily isolate the last digit, using bitwise operator AND (in C++, Java it is &) with num and -num:           
      a1b
&    a¯1b
    ——————–
= (0…0)1(0…0)

# Pocket Gems
## TODO: 210 (topological sort), NAN String setCharAt()

### NAN. Talk: HashMap collision

Hash tables deal with collisions in one of two ways.

Option 1: By having each bucket contain a linked list of elements that are hashed to that bucket. This is why a bad hash function can make lookups in hash tables very slow. 形成单链表

Option 2: If the hash table entries are all full then the hash table can increase the number of buckets that it has and then redistribute all the elements in the table. The hash function returns an integer and the hash table has to take the result of the hash function and mod it against the size of the table that way it can be sure it will get to bucket. So by increasing the size, it will rehash and run the modulo calculations which if you are lucky might send the objects to different buckets. 扩容重新计算

Java uses both option 1 and 2 in its hash table implementations.

### 377. Combination Sum IV (unique array，多次使用，返回combination个数)

**Description**

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

Example:

```
nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.
```
**Ideas**
1. DP, int array define the number from 1 to target
2. dp[i] = dp[i-num] num is each one in nums array

**Tag:** DP POCKET GEMS

```
public class Solution {
    public int combinationSum4(int[] nums, int target) {
        //Dp
        //bottom up
        if(target < 1) return 0;
        int[] dp = new int[target+1];
        dp[0] = 1;
        for(int t = 1; t <= target; t++){
            for(int num: nums){
                if(num > t) continue;
                dp[t] += dp[t-num];
            }
        }
        return dp[target];
    }
    
    // public int combinationSum4(int[] nums, int target) {
    //     //Resursion version
    //     //time limit exceeded
    //     int sum = 0;
    //     for(int num: nums){
    //         if(target - num > 0) sum += combinationSum4(nums, target - num);
    //         else if(target - num == 0) sum += 1;
    //     }
    //     return sum;
    // }
}

```

### 216. Combination Sum III (k个数，1-9，和为n)

**Description**

Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.


Example 1:

Input: k = 3, n = 7

Output:[[1,2,4]]

Example 2:

Input: k = 3, n = 9

Output:

[[1,2,6], [1,3,5], [2,3,4]]

**Ideas**
1. Add number within ascending sequence
2. 9 + 8 + 7 + ... 9-k+1 is the largest one, if n > max, just return;

**Tag:** backtracking POCKET GEMS

```
public class Solution {
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        helper(res, new ArrayList<Integer>(), k, n);
        return res;
    }
    private void helper(List<List<Integer>> res, List<Integer> curt, int k, int n){
        if(19*k - k*k < 2*n) return;
        int last = (curt.isEmpty())? 0: curt.get(curt.size()-1);
        if(k == 0) {
            res.add(new ArrayList<Integer>(curt));
            return;
        }
        for(int i = last+1; i <= 9 && ((2*i+k-1)*k <= 2*n); i++){
            curt.add(i);
            helper(res, curt, k-1, n-i);
            curt.remove(curt.size()-1);
        }
    }
}

```


### 40. Combination Sum II (数组有重复，但只可以使用多次）

**Description**

Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8, 
A solution set is: 
```
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

```

**Ideas**
1. Sort first
2. Backtracking with for loop
3. The start is idx+1, since no multi-use
4. Remember to offset the duplicates if the first one not used

**Tag:** backtracking POCKET GEMS

```
public class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backtracking(res, temp, candidates, -1, target);
        return res;
    }
    
    private void backtracking(List<List<Integer>> res, List<Integer> temp, int[] candidates, int idx, int target){
        if(target <= 0){
            if(target == 0 && temp.size() != 0) res.add(new ArrayList<Integer>(temp));
            return;
        }
        for(int i = idx+1; i < candidates.length; i++){
            temp.add(candidates[i]);
            backtracking(res, temp, candidates, i, target - candidates[i]);
            temp.remove(temp.size()-1);
            while(i+1 < candidates.length && candidates[i] == candidates[i+1]) i++;
        }
    }
}

```

### 39. Combination Sum (数组没有重复，但可以使用多次）

**Description**

Given a set of candidate numbers (C) (without duplicates) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [2, 3, 6, 7] and target 7, 
A solution set is: 
```
[
  [7],
  [2, 2, 3]
]
```

**Ideas**
1. Backtracking
2. Remember, when target <= 0 is the exit, not just target == 0
3. The start of for loop is idx itself
4. No need to sort

**Tag:** backtracking POCKET GEMS

```
public class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // Arrays.sort(candidates); // no need to sort
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backtracking(res, temp, candidates, 0, target);
        return res;
    }
    
    private void backtracking(List<List<Integer>> res, List<Integer> temp, int[] candidates, int idx, int target){
        if(target <= 0){
            if(target == 0 && temp.size() != 0) res.add(new ArrayList<Integer>(temp));
            return;
        }
        for(int i = idx; i < candidates.length; i++){
            temp.add(candidates[i]);
            backtracking(res, temp, candidates, i, target - candidates[i]);
            temp.remove(temp.size()-1);
        }
    }
}
```


### 41. First Missing Positive

**Description**

Given an unsorted integer array, find the first missing positive integer.

For example,
Given [1,2,0] return 3,
and [3,4,-1,1] return 2.

Your algorithm should run in O(n) time and uses constant space.

**Ideas**
1. The key here is to use swapping to keep constant space and also make use of the length of the array.
2. It means there can be at most n positive integers. 
3. So each time we encounter an valid integer.
4. Find its correct position and swap. Otherwise we continue.

保证在没遇到invalid input之前，第i个都是没问题的

**Tag:** two pointers POCKET GEMS

```
public class Solution {
    public int firstMissingPositive(int[] A) {
        int i = 0;
        while(i < A.length){
            if(A[i] == i+1 || A[i] <= 0 || A[i] > A.length) i++;
            else if(A[A[i]-1] != A[i]) swap(A, i, A[i]-1);
            else i++;
        }
        i = 0;
        while(i < A.length && A[i] == i+1) i++;
        return i+1;
    }
    
    private void swap(int[] A, int i, int j){
        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}
```

### 75. Sort Colors

**Description**

Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note:
You are not suppose to use the library's sort function for this problem.

**Ideas**
1. Use left and right pointers to record the position of next 0 and 2
2. Iterate the loop, when meet 0, swap with the left; meet 2, swap with the right; meet 1, do nothing
3. Make sure that after each time you have swapped the numbers, do i-- to check the curt again

**Tag:** two pointers POCKET GEMS

```
public class Solution {
    public void sortColors(int[] nums) {
        // int[] colors = new int[3];
        // for(int num : nums) colors[num]++;
        // for(int i = 0; i < nums.length; i++){
        //     if(i < colors[0]) nums[i] = 0;
        //     else if(i < colors[0] + colors[1]) nums[i] = 1;
        //     else nums[i] = 2;
        // }
        // 0, 3
        int left = 0, right = nums.length - 1;
        for(int i = 0; i <= right; i++){
            if(nums[i] == 2 && right != i){
                int swap = nums[i];
                nums[i--] = nums[right];
                nums[right--] = swap;
            }
            else if(nums[i] == 0 && left != i){
                int swap = nums[i];
                nums[i--] = nums[left];
                nums[left++] = swap;
            }
        }
        // if(nums[left] == 0) left++;
        // if(nums[right] == 3) right--;
        // // 1, 2
        // for(int i = left; i <= right; i++){
        //     if(nums[i] == 2 && right != i){
        //         int swap = nums[i];
        //         nums[i--] = nums[right];
        //         nums[right--] = swap;
        //     }
        //     else if(nums[i] == 1 && left != i){
        //         int swap = nums[i];
        //         nums[i--] = nums[left];
        //         nums[left++] = swap;
        //     }
        // }
    }
}

```


### 287. Find the Duplicate Number

**Description**

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.
Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.

**Ideas**
1. 0 -> 2 -> 1 -> 3 -> 1 -> 4;
2. This is just like to find a cycle on the linkedList;
3. 0 would always be a valid head, it never goes back to 0 anyway. 

**Tag:** two pointers binary search POCKET GEMS

```
public class Solution {
    public int findDuplicate(int[] nums) {
        // O(1) space and readOnly: No sort
        // Not only once: no xor
        // the number is from 1 to n which means that nums[0] will be a safe entry
        // since nums[0] != 0 and it will go next whatever
        // then after that it will the same question as cycle on linkedList
        
        //Binary search
        // if(nums == null || nums.length < 2) return -1;
        // int start = 1;
        // int end = nums.length - 1;
        // while(start < end){
        //     int mid = start + ((end - start) >> 1);
        //     int count = 0;
        //     for(int num: nums){
        //         if(num <= mid) count++;
        //     }
        //     if(count > mid) end = mid;
        //     else start = mid + 1;
        // }
        // return start;
        
        // Two Pointers
        
        if(nums == null || nums.length < 2){
            return -1;
        } 
        int slow = nums[0];
        int fast = nums[nums[0]];
        while(slow != fast){
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        fast = 0;
        while(slow != fast){
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
}

```

### 4. Median of Two Sorted Arrays

**Description**

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

Example 1:
```
nums1 = [1, 3]
nums2 = [2]

The median is 2.0
```
Example 2:
```
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5 
```

**Ideas**
1. Find median in two sorted array means find the kth number in two sorted array, where k is (m+n)/2 (or avg);
2. Do binary search: since each time we can discard half of the array:
3. Get the k/2th point of each array, if one is small, means that the segment before this should be discarded, as well as the segment after that
4. Remember to use k = 1 as start to protect from dead loop with no offset each recursion
5. Always remember to check the bound when cal the median in each recursion

**Tag:** binary search POCKET GEMS

```
public class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if(nums1 == null || nums2 == null) return -1;
        int m = nums1.length, n = nums2.length;
        int left = (m + n + 1) >> 1;
        int right = (m + n + 2) >> 1;
        return 0.5 * (getKth(nums1, 0, nums2, 0, left) + getKth(nums1, 0, nums2, 0, right));
    }
    
    private int getKth(int[] nums1, int s1, int[] nums2, int s2, int k){
        if(s1 > nums1.length-1) return nums2[s2 + k - 1];
        if(s2 > nums2.length-1) return nums1[s1 + k - 1];
        if(k == 1) return Math.min(nums1[s1], nums2[s2]); // important not k == 0: to protect from no offset
        int m1 = (s1 + k/2 - 1 < nums1.length) ? nums1[s1 + k/2 - 1] : Integer.MAX_VALUE;
        int m2 = (s2 + k/2 - 1 < nums2.length) ? nums2[s2 + k/2 - 1] : Integer.MAX_VALUE;
        if(m1 < m2) return getKth(nums1, s1+k/2, nums2, s2, k-k/2);
        return getKth(nums1, s1, nums2, s2+k/2, k-k/2);
    }
}
```


### NAN. JSON parse to DB

**Description**

json格式
```
[
{type:"session", data:{player_id:89757, session_id:66055, name:"ergou", date:"2016-1-1HHMMSS"}},
{type:"purchase", data:{player_id:89757, session_id:66055, state:"success", date:"2016-1-2HHMMSS"}},
{type:"battle", data:{player_id:...., session_id:...., result:....}}, 
.....
]
```
就是这类型的，长这样整体叫events，里边的每一条叫event，长的差不多，名字不一样，data也就相应的不一样。

database格式
```
class Row {
      String tableName;
      Map<String, String> map;
}  
```
就是把json的格式parse了然后变成这样就行了，tableName就是对应上边的type（session，purchase，battle这种），map里就是每个event里data里所有的key value pair。
然后给了一个class
```
class JSONValue {
       String getAsString()
       Long getAsLong()
       Map getAsMap()
       List getAsList()
       ....
}
```
就是每个key value pair的value都是一个JSONValue，整个的list也是JSONValue，data也是，可以用这些method转换成java的的类型（别的语言也行，我是java），但是特定的类型只能call特定的函数，比如session对应的是数字，所以你提取session的值的时候只能call getAsLong()，如果你call了getAsMap()或其他的就不对。而data是Map，parse data时候只能call getAsMap()这个意思。
你要实现的method接口是
```
List<Row> parseToDB(List<JSONValue> events, Config c) {

}
```
类似这样表达
arg1 就是上面的JSON，是一堆events。Config是另一个类型也是需要设计的，具体功能差不多就是告诉你哪个类型需要调用JSONValue里的哪个method才能parse对，比如告诉你session里的name是string类型，那么你parse name这个field的时候就要call getAsString()这样。这里需要OOD，我设计了一个type的Interface然后就差不多那样

**Ideas**
1. Map dataMap = eventMap.get("data").getAsMap();
2. Use Config class to get method

**Tag:** sort POCKET GEMS

```
code 略
```


### NAN. Shortest Manhattan distance

**Description**
I have a grid with certain intersections marked. I'd like to find the intersection that is closest to all the marked intersections. That is, I need to find a point such that the sum of distances from all points is minimum.

有一个网格。在这个网格上有若干个人，如何在
此网格上找出一个约会地点，使其到所有人的距离之和最短。

**Ideas**
1. Image we find a position on the 1D line
2. The result would always be the median of all these elements (a segment when there're two of them)
3. So for the manhattan distance, you can do it independently
4. Find the two median and combine them together. 

**Tag:** sort POCKET GEMS

```
code 略
```


### 297. Serialize and Deserialize Binary Tree

**Description**
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

For example, you may serialize the following tree
```
    1
   / \
  2   3
     / \
    4   5
```
as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

**Ideas**

1. BFS 方法
2. Use a queue to output level order 
3. Use a queue to create tree's children
**Tag:** Tree POCKET GEMS

```
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "";
        Queue<TreeNode> q = new LinkedList<>();
        StringBuilder res = new StringBuilder();
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node == null) {
                res.append("n ");
                continue;
            }
            res.append(node.val + " ");
            q.add(node.left);
            q.add(node.right);
        }
        return res.toString();
    }

    public TreeNode deserialize(String data) {
        if (data == "") return null;
        Queue<TreeNode> q = new LinkedList<>();
        String[] values = data.split(" ");
        TreeNode root = new TreeNode(Integer.parseInt(values[0]));
        q.add(root);
        for (int i = 1; i < values.length; i++) {
            TreeNode parent = q.poll();
            if (!values[i].equals("n")) {
                TreeNode left = new TreeNode(Integer.parseInt(values[i]));
                parent.left = left;
                q.add(left);
            }
            if (!values[++i].equals("n")) {
                TreeNode right = new TreeNode(Integer.parseInt(values[i]));
                parent.right = right;
                q.add(right);
            }
        }
        return root;
    }
}
```


### NAN. Find good number

**Description**
一个range(int a, int b) 里面找到good number.
good number的定义是，所有组成它的prime factor的个数是个prime number. 8（） d( |1 i, m! ]& |'
V& X:
比如：10 = 2 * 5， 是两个prime number，所以他是good number. 30 = 2*3*5 三个prime num
，也是good number 但是7 = 7 不是，因为1不是prime number个

**Ideas**

1. First call countPrime() to cal the boolean array for isPrime()
2. Then use this boolean array to record the count of each prime factor
3. Then check isPrime(count);

**Tag:** prime, math POCKET GEMS

```
public class Solution {
    public boolean isGoodNumber(int n) {
        int count = 0;
        boolean[] isPrime = new boolean[n+1];
        for(int i = 2; i <= n; i++){
            isPrime[i] = true;
        }
		// build the isPrime set
        for(int i = 2; i*i <= n; i++){
            if(isPrime[i]){
                for(int j = i*i; j <= n; j+=i){
                    isPrime[j] = false;
                }
            }
        }
        for(int i = 2; i <= n; i++){
	    if(isPrime[i]){
	        while(n % i == 0){
		    count++;
		    n /= i;
		}
	    }
	}
	return isPrime[count];
    }
} 
```


### 398. Random Pick Index

**Description**
Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Note:
The array size can be very large. Solution that uses too much extra space will not pass the judge.

Example:
```
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);

// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);

// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
```
**Ideas**

1. Each time there is a prob of (1/k) to update the result, and 1-(1/k) to remain.
2. k is index + 1, or say the times we iterate
3. The final prob is 1/n for each

**Tag:** Reservior Sampling POCKET GEMS

```
public class Solution {

    int[] nums;
    public Solution(int[] nums) {
        // Arrays.sort(nums);
        this.nums = nums;
    }
    
    public int pick(int target) {
        //O(n) work solution
        int k = 1;
        int res = -1;
        for(int i = 0; i < nums.length; i++){
            double r = Math.random();
            if(target == nums[i]){
                if(r <= 1.0/k) res = i;    
                k++;
            }
        }
        return res;
        
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int param_1 = obj.pick(target);
 */
 ```


### NAN String implementation

**Description**
Design a string class , with implementation of charAt() and substring(b,e), with substring()
requires O(1) time and O(1) space complexity
Followup:
A new method setCharat(index, char) is added, a substring must keep the changes of parrent
string that are made before its creation, but both the parrent string and the substring will not
affect each other after the creation of the substring, requires O(1) space complexity

**Ideas**

1. The value (charArray) doesn't change.
2. Use offest and count (start and length) to define the substring
3. Because String is immutable, you don't need to spare more space for it.

Followups:
1. 

**Tag:** String POCKET GEMS

```
//JDK 6
String(int offset, int count, char value[]) {
	this.value = value;
	this.offset = offset;
	this.count = count;
}
 
public String substring(int beginIndex, int endIndex) {
	//check boundary
	return  new String(offset + beginIndex, endIndex - beginIndex, value);
}
//JDK 7
public String(char value[], int offset, int count) {
	//check boundary
	this.value = Arrays.copyOfRange(value, offset, offset + count);
}
 
public String substring(int beginIndex, int endIndex) {
	//check boundary
	int subLen = endIndex - beginIndex;
	return new String(value, beginIndex, subLen);
}
```


### 140. Word Break II

**Description**
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. You may assume the dictionary does not contain duplicate words.

Return all such possible sentences.

For example, given
s = "catsanddog",
dict = ["cat", "cats", "and", "sand", "dog"].

A solution is ["cats and dog", "cat sand dog"].

**Ideas**

1. DP, use 2D boolean array to record whether dp[start][end] is valid
2. dp[i][i] determines that from i to end is valid
3. After fill with the 2D array, check each (start,end) recursively. 
4. Since the previous would only be valid if the leading is valid, so we can do the dfs safely.
5. Build the string with spaces. When the end reaches the length of string, return

**Tag:** DP, DFS POCKET GEMS

```
public class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        List<String> res = new ArrayList<>();
        if(s == null || wordDict == null || wordDict.size() == 0) return res;
        int ls = s.length();
        boolean[][] dp = new boolean[ls + 1][ls + 1];
        // init boolean matrix
        // dp[i][i] means there is solution from i to end, no matter what it is
        dp[ls][ls] = true;
        for(int i = ls; i >= 0 ; i--){
            for(String word : wordDict){
                int lw = word.length();
                if(lw + i <= ls && word.equals(s.substring(i, i +lw))){
                    dp[i][i+lw] = dp[i+lw][i+lw];
                    if(dp[i+lw][i+lw]) dp[i][i] = true;
                }
            }
        }
        // go to each solution
        recurse(s, res, "", dp, 0);
        return res;
    }
    
    private void recurse(String s, List<String> res, String temp, boolean[][] dp, int start){
        int ls = dp.length - 1;
        for(int end = start + 1; end <= ls; end++){
            if(dp[start][end]) {
                if(end == ls){
                    res.add(temp + s.substring(start, end));
                    return;
                }
                recurse(s, res, temp+s.substring(start, end)+" ", dp, end);
            }
        }
    }
}
```


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
