class Greeter(object):

    # constructor
    def __init__(self, name):
        self.name = name # create an instance variable

    # instance method
    def greet(self, loud=False):
        if loud:
            print(f"Hello, {self.name.upper()}")
        else:
            print(f"Hello, {self.name}")


g = Greeter("Fred")
g.greet()
g.greet(loud=True)
