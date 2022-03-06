"""3. Strive School Challenge:
In this challenge you need to code a function that receives a list of numbers and returns:

Strive if the number is divisible by 3
School if the number is divisible by 5
Strive School if the number is divisible by 3 and 5
the number itself otherwise
i. e. strive_school([1, 2, 3, 5, 15]) outputs [1, 2, Strive, School, Strive School]"""




def strive_school(Strive):
    y = []
    x = len(Strive)
    for i in range(0,x):
        if Strive[i] % (3 * 5) == 0:
            print("Strive School" )
        elif Strive[i] % 5 == 0:
            print("School", end=", ")
        elif Strive[i] % 3 == 0:
            print("Strive", end=", ")
        else:
            print(Strive[i], end=", ")
              
strive_school([7, 8, 3, 5, 15])
