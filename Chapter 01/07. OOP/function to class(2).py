


class Pyramid:

    """
    Pyramid challenge

    Attribute:
        height(integer) 
    """
    
    def __init__(self, height):
        self.height = height

    def draw_pyramid(self):
        for i in range(0, self.height +1):
              print(i * "#")
        print("\n")    

    
pyramid = Pyramid(5)

pyramid.height

pyramid.draw_pyramid()