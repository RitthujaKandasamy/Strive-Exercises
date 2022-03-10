

from turtle import distance


class Food:

    """  
    Create a class called food that has following attributes.

       Attributes: name, price
    """
    

    def __init__(self, name, price):
        self.name = name
        self.price = price
        self.__quality_of_the_food = []
    

    def calculate_food_cost(self, beginning_inventory, Purchases, Ending_Inventory, Food_sales):
        percentage = (beginning_inventory +  Purchases -  Ending_Inventory) / Food_sales
        return percentage

    def calculate_discount(self, discount):
        discount_first = self.price * (1 - discount)
        return discount_first

    def calculate_time(self, prepare, person):
         total_time = person * prepare
         return total_time

    def add_quality(self, food_quality):
        self.__quality_of_the_food.append(food_quality)

    def get_quality_average(self):
        ratings = sum(self.__quality_of_the_food) / len(self.__quality_of_the_food)
        return ratings

    def get_total_ratings(self):
        return len(self.__quality_of_the_food)

    def __str__(self):
        return f'Food Name: {self.name}\nFood Price: {self.price}$'



class Slow_food(Food):

    """
       Create a class called slow food that has following attributes.

       Attributes: name, price, nutritional_content, flavor, country_of_origin
    """

    def __init__(self, name, price, nutritional_content, flavor, country_of_origin):
        super().__init__(name, price)
        self.country_of_origin = country_of_origin
        self.nutritional_content = nutritional_content
        self.flavor = flavor
        

    def food_nutrition(self):

        # nutritionalvalue are calculated in calories

        if self.nutritional_content >= 1000:
            print('5 Stars')
        elif self.nutritional_content <= 1900:
            print('3 Stars')
        else:
            print('No Stars')

  
    def food_order(self, order):
        self.__extra_order = order
        if self.__extra_order < 2:
            raise ValueError("Food order must be greater than 2, we need to take more than 2 order daily")


    def __str__(self):
        return f'Food Name: {self.name}\nFood Price: {self.price}$\nFood Nutritional Content: {self.nutritional_content}\nFood Flavours: {self.flavor}\nFood Country of Origin: {self.country_of_origin}'



class Fast_food(Food):

    """
       Create a class called slow food that has following attributes.

       Attributes: name, price, country_of_origin, flavor, takeaway_or_dine_in
    """

    def __init__(self, name, price, country_of_origin, flavor, takeaway_or_dine_in):
        super().__init__(name, price)
        self.country_of_origin = country_of_origin
        self.flavor = flavor
        self.takeaway_or_dine_in = takeaway_or_dine_in
        self.__discount = 0.5


    def food_delivery(self, distance):
         
        # delivery are calculated in kilometers
        #distance = input()
        #print('Enter your distance from house to our shop: ' + distance)

        if distance >= 5:
            print('Food Delivery is free')
        else:
            print('Food Delivery cost 10 Euro')
        

    def special_discount(self):
        discount_offer = self.price / self.__discount
        return discount_offer
        
        
    def __str__(self):
        return f'Food Name: {self.name}\nFood Price: {self.price}$\nFood Country of Origin: {self.country_of_origin}\nFood Flavours: {self.flavor}'



food = Food('Local', 34)
print(food.__str__())
print(food.calculate_food_cost(45, 789, 24, 65))
print(food.calculate_discount(0.5))
print(food.calculate_time(40, 4))


food1 = Slow_food('Dosa', 67, 678, 'Spicy', 'India')
print(food1.__str__())
print(food1.food_nutrition())
print(food1.food_order(5))
food1.add_quality(5)
food1.add_quality(8)
food1.add_quality(6)
food1.add_quality(10)
food1.add_quality(7)


food2 = Fast_food('Pasta', 45, 'Italy', 'Spicy', 'Takeaway')
print(food2.__str__())
print(food2.food_delivery(4))
print(food2.special_discount())
food2.add_quality(7)
food2.add_quality(5)
food2.add_quality(7)
food2.add_quality(9)


all_food = [food1, food2]
print('****************************')


for foodie in all_food:
    print(foodie)
    print('****************************')
    print(f'Reviewed by {foodie.get_total_ratings()} customers,'f'with and average rating of{foodie.get_quality_average()}')
    print('###############################')