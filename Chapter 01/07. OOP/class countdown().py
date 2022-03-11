
class Countdown:
    """
    Create a countdown function that starts at a certain count, 
        and counts down to zero. Instead of zero, print "Blast off!".

        Attributes: start, end, between
    """

    def __init__(self, start, end, between):
       self.start = start
       self.end = end
       self.between = between
       pass
    
    
    def countdown_blast(self):
      for count in range(self.start, self.end, self.between):
         print(count)
      print ("Blast off!")

c = Countdown(30, 10, -1)

c.countdown_blast()

