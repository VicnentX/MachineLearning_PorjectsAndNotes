def quicksort(arr):
    if len(arr) <= 1:
        return arr

    mid = arr[len(arr) // 2]
    left = [i for i in arr if i < mid]
    middle = [i for i in arr if i == mid]
    right = [i for i in arr if i > mid]

    return quicksort(left) + middle + quicksort(right)

print(quicksort([1,2,4,5,6,7,88,7,6,5555,4,3,43,546,3]))

