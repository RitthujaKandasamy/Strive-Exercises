

import math
from re import X



class Cube:


   """
  Cube Challenge:
    Create an application that gives the data to represent 2 cubic objects in a 3-dimensional space that is capable to determine if they intersect.
       1. If the cubes intersect, determine what the volume of the shared space is.
       2. The two cubes are parallel to each other (they are not rotated in any way).
    To build each of the cubes the user must be able to provide coordinates (x,y,z) for the center of the cube and the length of the edge.

  The code should return:
    A boolean indicating it the cubes intersect and the volume of the shared space, if the cubes do not intersect then the volume of shared space should be 0.

  I.E. :
    No intersection
       cube1                   cube2            
       center: (10, 10, 0)     center: (5, 5, 0)
       size: 5                 size: 2

  Returns:
       (False, 0)    
   """


   def __init__(self, length, width, height):
        self.x = length
        self.y = width
        self.z = height

   
   def solve(self, length1, width1, height1, size1, length2, width2, height2, size2):
       self.y1 = width1
       self.z1 = height1
       self.x1 = length1
       self.a1 = size1
       self.a2 = size2
       self.x2 = length2
       self.y2 = width2
       self.z2 = height2

       # if x2 is not in the x1, then it can not intersect
       # so we are using (not) to check
       
       if not ((((self.x2 - (self.a2/2)) > (self.x1 - (self.a1/2))) and ((self.x2 - (self.a2/2)) < (self.x1 + (self.a1/2)))) or (((self.x2 + (self.a2/2)) > (self.x1 - (self.a1/2))) and ((self.x2 - (self.a2/2)) < (self.x1 + (self.a1/2))))):
             return False
       else:
             return True


    


   