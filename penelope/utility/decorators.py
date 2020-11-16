import functools
import logging


def try_catch(func, exceptions=None, suppress=False):
    """
    Surrounds the function with a try-except block
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except exceptions as e:
            logging.exception('Exception occurred: [{}]'.format(e))
            if not suppress:
                raise

    return wrapper
