

import math

class Book:

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

    def total_copies(self):
        self.total_copies = int(self.first_copy + self.copies)
        print(self.total_copies)  

book = Book(1.40, 3, 0.75, 24.95) 
book.shipping_cost
book.discount()
book.additional_copy
book.cover_price         