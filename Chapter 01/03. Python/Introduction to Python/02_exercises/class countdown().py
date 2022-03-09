import math
import matplotlib.pyplot as plt

class countdown():

    def __init__(self, nums):
       self.nums = nums
    
    
    def countdown(self):
        print("Countdown")
    for count in range(20, 0, self.nums -1):
        print(count)
        
    print ("Blast off!")




