

class Food:

    """  
    Create a class called food that has following attributes.

       Attributes: name, flavor, price, country_of_origin, nutritional_content
    """
    

    def __init__(self, name, flavor, price, nutritional_content, country_of_origin):
        self.name = name
        self.nutritional_content = nutritional_content
        self.flavor = flavor
        self.country_of_origin = country_of_origin
        self.price = price
        self.__ratings_stars = []
        pass

    def calculate_food_cost(self, beginning_inventory, Purchases, Ending_Inventory, Food_sales):
        self.percentage = (beginning_inventory +  Purchases -  Ending_Inventory) / Food_sales
        return self.percentage

    def calculate_discount(self, discount):
        self.discount = discount
        self.discount_first = int(self.price * (1 - self.discount))
        return self.discount_first

    def calculate_time(self, prepare, person):
         self.person = person
         self.prepare = prepare
         self.total_time = self.person * self.prepare
         return self.total_time

    def add_rating(self, stars):
        self.__ratings_stars.apppend(stars)

    def get_ratings_average(self):
        return sum(self.__ratings.stars) / len(self.__ratings_stars)

    def get_total_ratings(self):
        return len(self.__ratings_stars)

    def __str__(self):
        return f'Food Name: {self.name}\nFood Flavours: {self.flavor}\nFood Nutritional Content: {self.nutritional_content}\nFood Price: {self.price}\nFood Country of Origin: {self.country_of_origin}'



class Slow_food:

    """
       Create a class called slow food that has following attributes.

       Attributes: name, price, nutritional_content
    """