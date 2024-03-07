from time import time


def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper


def adjacency_4(x, y):
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
