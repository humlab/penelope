import functools
import logging


def try_catch(func, exceptions=None, suppress=False, nice=False):
    """
    Surrounds the function with a try-except block
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except exceptions as e:
            if nice:
                logging.error(str(e))
            else:
                logging.exception('[{}]'.format(e))
            if not suppress:
                raise

    return wrapper
