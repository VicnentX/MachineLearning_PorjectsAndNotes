class Cat:
    """
    define a Cat class
    """
    # initiate
    def __init__(self, new_name, new_age):
        self.name = new_name
        self.age = new_age

    def __str__(self):
        return f"{self.name} is {self.age} year old"

    # method
    def eat(self):
        print("eating fish")

    def drink(self):
        print("drinking cocacola")

    def introduce(self):
        print(f"{self.name} is {self.age} year old")


tom = Cat("tom", 12)
black_captain = Cat("black_captain", 25)
print(tom)
print(black_captain)
print(tom.introduce())
