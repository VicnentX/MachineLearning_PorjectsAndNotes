# Python functions are defined using the def keyword. For example:


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


for x in [-1, 0, 1]:
    print(sign(x))

# Prints "negative", "zero", "positive"

# We will often define functions to take optional keyword arguments, like this:


def hello(name, loud=False):
    if loud:
        print(f"HELLO, {name.upper()}!")
    else:
        print('Hello, %s' % name)


hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
# There is a lot more information about Python functions in the documentation.