# gausic sum


def gausi(i):
    if i == 1:
        return 1
    else:
        return gausi(i - 1) + i


print(gausi(10))
