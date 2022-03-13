

import math
from re import X
from turtle import width, window_width



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


   def __init__(self, length1, width1, height1, size1, length2, width2, height2, size2):
      self.y1 = width1
      self.z1 = height1
      self.x1 = length1
      self.a1 = size1
      self.a2 = size2
      self.x2 = length2
      self.y2 = width2
      self.z2 = height2

   
   def solve(self):
       
       # if x2 is not in the x1, then it can not intersect
       # so we are using (not) to check

       if not ((((self.x2 - (self.a2/2)) > (self.x1 - (self.a1/2))) and ((self.x2 - (self.a2/2)) < (self.x1 + (self.a1/2)))) or (((self.x2 + (self.a2/2)) > (self.x1 - (self.a1/2))) and ((self.x2 + (self.a2/2)) < (self.x1 + (self.a1/2))))):
             return False
       else:
             return True



   def volume(self):
       a3 = self.x1 - (self.a1/2)
       a4 = self.x1 + (self.a1/2)
       b1 = self.x2 - (self.a2/2)
       b2 = self.x2 + (self.a2/2)
       

       if (b2 > a3) and (b2 < a4) and (b1 < a3) and (b1 < a4):
           length = (a4 - a3) - (a4 - b2)
           print(length)
       elif (a4 > b1) and (a3 < b1) and (b2 > a4) and (a3 < b2):
           length3 = (a4 - b1)
           print(length3)
       elif (b1 > a3) and (b2 < a4) and (a4 > b1) and (a3 < b2):
           length4 = (b2 - b1)
           print(length4)    
         


   def volume1(self):
       a5 = self.y1 - (self.a1/2)
       a6 = self.y1 + (self.a1/2)
       b3 = self.y2 - (self.a2/2)
       b4 = self.y2 + (self.a2/2)

       if (b4 > a5) and (b4 < a6) and (b3 < a5) and (b3 < a6):
           width = (a6 - a5) - (a6 - b4)
           print(width)
       elif (a6 > b3) and (a5 < b3) and (b4 > a6) and (a5 < b4):
           width3 = (a6 - b3)
           print(width3)
       elif (b3 > a5) and (b4 < a6) and (a6 > b3) and (a5 < b4):
           width4 = (b4 - b3)
           print(width4)    


   def volume2(self):
       a7 = self.z1 - (self.a1/2)
       a8 = self.z1 + (self.a1/2)
       b5 = self.z2 - (self.a2/2)
       b6 = self.z2 + (self.a2/2)

       if (b6 > a7) and (b6 < a8) and (b5 < a7) and (b5 < a8):
           height = (a8 - a7) - (a8 - b6)
           print(height)
       elif (a8 > b5) and (a7 < b5) and (b6 > a8) and (a7 < b6):
           height3 = (a8 - b5)
           print(height3)
       elif (b5 > a7) and (b6 < a8) and (a8 > b5) and (a7 < b6):
           height4 = (b6 - b5)
           print(height4)    


   #def total_volume(self):
       #volume 

cube = Cube(10, 10, 0, 5, 8, 9, 0, 2)
print(cube.solve())       
print(cube.volume())
print(cube.volume1())
print(cube.volume2())
#print(cube.volume()*cube.volume1()*cube.volume2())