import inspect
import numpy as np



def __nice_args(args, kwargs):
    """ Helper function to make args look nice. Keeps

    Clean up *args and **kwargs from inspect to keep track for logging.
    Currently implemented: removes "self" objects, only logs objects that are
        in GOOD_TYPES (see code).

    Parameters
    ----------    
    args: dict
        arguments.
    kwargs: dict
        keyword arguments.

    Returns
    -------
    str

    """

    # will only log arguments with these types
    GOOD_TYPES = [str, int, float, np.float64, np.int64, np.int32, np.ndarray,
                  list, bool, np.nan]

    # keep tracks of nice args
    nice_args = []

    # go through all arguments and check for condition
    for i, arg in enumerate(args):
        nice_arg = False

        # don't want the "self" object (first object) to be logged
        if i > 0:
            nice_arg = True
        # add other conditions below
        if type(arg) in GOOD_TYPES or arg is None:
            nice_arg = True

        if nice_arg:
            nice_args.append(arg)

    # go through kwargs dict and make them strings
    for arg, val in kwargs.items():
        if type(val) in GOOD_TYPES or val is None:
            kwargs_txt = f"{arg}={val}"
            nice_args.append(kwargs_txt)

    return nice_args


def __func_logger(func):
    """ Helper function to class_logger, keeps track of their name/args/kwargs.

    Parameters
    ----------
    func: function
        function to be logged.

    Returns
    -------
    function: decorator function to be evaluated

    """
       
    # decorator to be called for logging
    def decorator(*args, **kwargs):
        # evaluate the function
        result = func(*args, **kwargs)

        # get only nice args for logging
        nice_args = __nice_args(args, kwargs)

        # shorter version
        LOGS = f"PROCESSING LOG: function {func.__name__} with args {nice_args}"
        # longer version
        # LOGS = f"{func.__code__}: function {func.__name__} with args {nice_args}"
        print(LOGS)

        return result

    return decorator

def time_logger(class_definition):
    """ Decorator to log every method in class.

    Parameters
    ----------
    class_definition: class
        The class to log.

    Returns
    -------
    None
    
    """
    for name, method in inspect.getmembers(class_definition):
        if (not inspect.ismethod(method) and not inspect.isfunction(method)) or inspect.isbuiltin(method):
            continue
        func_decorator = __func_logger(method)
        setattr(class_definition, name, func_decorator)

    # class_definition.function_call_logs += LOGS
    return class_definition