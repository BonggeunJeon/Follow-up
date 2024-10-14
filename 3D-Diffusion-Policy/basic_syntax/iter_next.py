"""
    "iter()" function returns an "iterator" from an iterable (like a list or tuple).
    
    You can traverse through all the elements of the collection, one at a time.
"""

my_list = [1, 2, 3, 4, 5]
iterator = iter(my_list) # An object that can be used to iterate over "my_list"

""" 
    "next()" function retrieves the next item from the iterator.
"""
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
#print(next(iterator)) # It will raise StopIteration because the list has been exhausted.

"""
    "iter()" is used to convert a list or tuple into an iterator
    "next()" is used to get the next item from that iterator.
"""


