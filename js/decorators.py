import functools
import pickle

def pickler(func):
    '''
    Decorator that automatically pickles a method return value if the do_pickle argument is set to true
    :param func: input function
    :return: output function
    '''

    @functools.wraps(func)
    def pickler(*args, **kwargs):
        doPickle = kwargs.get('do_pickle')
        if doPickle is not None:
            del kwargs['do_pickle']
        rval = func(*args, **kwargs)
        if doPickle:
            rval = pickle.dumps(rval)
        return rval
    return pickler

