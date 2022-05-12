import time


def function_timer(func):
    def inner(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()

        print('timing:{}: {}'.format(func.__name__, toc - tic))

        return result

    return inner
