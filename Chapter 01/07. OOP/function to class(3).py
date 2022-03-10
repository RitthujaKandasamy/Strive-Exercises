

class Primenumber:

    """
        List of all prime numbers.

        Attributes: fnum, snum, tnum
    """

    def __init__(self, fnum, snum):
        self.first_number = fnum
        self.second_number = snum
        pass

    def prime_numbers(self):
        for num in range(self.first_number, self.second_number):
            for i in range(self.first_number, num):
                if num % i == 0:
                    break 
            else:
                print(num)



prime = Primenumber(2, 30)

prime.prime_numbers()