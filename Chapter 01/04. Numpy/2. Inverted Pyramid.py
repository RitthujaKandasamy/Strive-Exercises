"""2. Inverted Pyramid Challenge:
In this challenge you need to code a function that receives a number print out this pyramid:

i. e. print_pyramid(5) """



def print_pyramid(n):

    for i in range(n, 0, -1):
        for j in range(0, i):
            print("*", end="")
        print("\n")    

print_pyramid(5)