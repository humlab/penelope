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


def suppress_error(func, exceptions=(Exception,), suppress=True, nice=True):
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
        if exc_type is not None and not isinstance(exc_val, self.exception_class):
            raise self.exception_class(str(exc_val)) from exc_val
        return False


def deprecated(obj):
    if isinstance(obj, (type,)):
        return _deprecated_class(cls=obj)
    return _deprecated_function(f=obj)


def _deprecated_function(f):
    def _deprecated(*args, **kwargs):
        logging.warning(f"Method '{f.__name__}' is deprecated and will be removed in future release")
        return f(*args, **kwargs)

    return _deprecated


def _deprecated_class(cls):
    class Deprecated(cls):
        def __init__(self, *args, **kwargs):
            logging.warning(f"Class '{cls.__name__}' is deprecated and will be removed in future release")
            super().__init__(*args, **kwargs)

    return Deprecated
