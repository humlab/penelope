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

class ExpectException:
    def __init__(self, exception_class):
        self.exception_class = exception_class

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not isinstance(exc_val,
                                                   self.exception_class):
            raise self.exception_class(str(exc_val)) from exc_val
        return False
