

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
        self.first_copy = int(self.shipping_cost + self.cover_price) / self.discount
        return self.first_copy

    def copies(self):
        self.copies = int((self.additional_copy + self.cover_price) * 59) / self.discount
        return self.copies

    def total_copies(self, total):
        self.total = int(self.first_copy + self.copies)
        print(self.total)  

book = Book(1.40, 3, 0.75, 24.95) 
book.shipping_cost
book.discount
book.additional_copy
book.cover_price  

book.total_copies()