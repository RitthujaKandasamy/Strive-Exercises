"""4. Non Duplicated Challenge:
In this challenge you need to code a function that receives a list of numbers and returns the non-duplicated number:

 i. e. function_whatever([1, 1, 2, 2, 3, 5, 5, 6, 6]) outputs 3

i. e. function_whatever([1, 2, 2, 3, 3]) outputs 1"""



def non_duplication_number(nums):
    
    x = len(nums)
    for i in range(0, x):
        for j in range(0, x):
            if i!=j:
                if nums[i] == nums[j]:
                    break
        if nums[i] != nums[j]:
            print(nums[i])

non_duplication_number([1, 2, 2, 3, 6, 5, 9, 6, 10, 9])             