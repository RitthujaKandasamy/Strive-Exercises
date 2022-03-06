"""1. Pyramid Challenge:
In this challenge you need to code a function that receives a number print out this pyramid:

i. e. print_pyramid(5)""" 


def print_pyramid(n):

    for i in range(0, n):
        for j in range(i + 1):
            print("*", end="")
        print("\n")    

print_pyramid(5)