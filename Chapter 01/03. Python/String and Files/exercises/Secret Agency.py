"""5. Secret Agent Challenge:
The goal of this task is to write a function that is able to find the Secret Agent (if any) in a group of people.

The function's parameters are the number of the people and a list of pairs (a list of two elements) with the following scheme:

the first element of the pair is the name of the person who has information about the person in the second element of the pair. Example: [[felix, lara], [felix, jeno], [lara, jeno]] means that  Felix has information about Jeno and Lara, then Lara has information about Jeno.

 

Rules:

A Secret Agent is the person that has information about all the other people, but no one has information about the Secret Agent.

So, in the previous example Felix has information about all the others, but no one has information about Felix -> Felix is the Secret Agent.

You have to return:

the Secret Agent (if any): [[felix, lara], [felix, jeno], [lara, jeno]] -> returns felix | [[felix, jeno], [lara, jeno], [lara, felix]] -> returns lara
0 if there isn't a Secret Agent ([[felix, lara], [lara, jeno], [jeno, felix]] -> returns 0) 
if the number of the people is less or equals to 0, or the length of the list is grater than the number of the people or if the length of the pair is not 2 -> returns -1"""




def Secret_Agent_Challenge(a):

    x = len(a[0])
    for i in range(0, x):
        for j in range(0, x):
            if i != j:
                if a[i] == a[j]:
                    print("felix")
                    

    

Secret_Agent_Challenge([["felix", "jeno"], ["lara", "jeno", "Jangiri"], ["lara", "felix"]])