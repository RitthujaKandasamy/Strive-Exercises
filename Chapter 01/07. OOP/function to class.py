

import math

class Book:

    """
    The cover price of a book is 24.95 EUR, but bookstores get a 40 percent discount. 
        Shipping costs 3 EUR for the first copy and 75 cents for each additional copy. 
            Calculate the total wholesale costs for 60 copies.

        Attributes: discount, shipping, copy, price 
    """      

    def __init__(self, discount, shipping, copy, price):
        self.shipping_cost = shipping
        self.discount = discount
        self.additional_copy = copy 
        self.cover_price = price
        pass
        

    def first_copy(self):
        self.first_book_copy = int((self.shipping_cost + self.cover_price) / self.discount)
        return self.first_book_copy 

    def copies(self):
        self.number_of_copy = int(((self.additional_copy + self.cover_price) * 59) / self.discount)
        return self.number_of_copy

    def total_copies(self):
        total = int(self.first_book_copy + self.number_of_copy)
        print(total)  

book = Book(1.40, 3, 0.75, 24.95) 

book.first_copy()
book.copies()
book.total_copies()
