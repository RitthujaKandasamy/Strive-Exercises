

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


   def __init__(self, x1, y1, z1, a1, a2):
        self.widt1 = y1
        self.height1 = z1
        self.length1 = x1
        self.size1 = a1
        self.size2 = a2


   def intersect_not(self, x2, y2, z2):
        self.length2 = x2
        self.width2 = y2
        self.height2 = z2

        # cube intersect or not formula

        if (x2 - a1) > (x1 - )