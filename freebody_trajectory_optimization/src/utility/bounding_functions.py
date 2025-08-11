import numpy as np

def smax(val1, val2, k):
    """
    Smooth maximum using Log-Sum-Exp. This expression is mumerically stable and produces a value 
    slightly larger than max(val1,val2), depending on k.
    """
    m = np.maximum(k * val1, k * val2)
    return (1.0 / k) * (m + np.log(np.exp(k * val1 - m) + np.exp(k * val2 - m)))
def dsmax__dval1(val1, val2, k):
    """
    Calculates the partial derivative of smax(val1, val2, k) with respect to val1.
    """
    return np.exp(k * (val1 - smax(val1, val2, k)))
def dsmax__dval2(val1, val2, k):
    """
    Calculates the partial derivative of smax(val1, val2, k) with respect to val2.
    """
    return dsmax_dval1(val2, val1, k)
def smin(val1, val2, k):
    """
    Smooth minimum using Log-Sum-Exp. This expression is mumerically stable and produces a value 
    slightly smaller than min(val1,val2), depending on k.
    """
    m = np.maximum(-k * val1, -k * val2)
    return (-1.0 / k) * (m + np.log(np.exp(-k * val1 - m) + np.exp(-k * val2 - m)))
def dsmin__dval1(val1, val2, k):
    """
    Calculates the partial derivative of smin with respect to val1.
    """
    return np.exp(-k * (val1 - smin(val1, val2, k)))
def dsmin__dval2(val1, val2, k):
    """
    Calculates the partial derivative of smin with respect to val2.
    """
    return dsmin__dval1(val2, val1, k)
def             bounded_smooth_func(funcval, min_bound, max_bound, k):
    """
    General function that is smoothly bounded between a min and max. 
    """
    max_bound_funcval     = smin(          funcval, max_bound, k) # max bound
    min_max_bound_funcval = smax(max_bound_funcval, min_bound, k) # min bound
    return min_max_bound_funcval
def derivative__bounded_smooth_func(funcval, min_bound, max_bound, k):
    """
    Derivative of a general function that is smoothly bounded between a min and max. 
    """
    return dsmax__dval1( smin(funcval, max_bound, k), min_bound, k ) * dsmin__dval1(funcval, max_bound, k)
def             bounded_nonsmooth_func(funcval, min_bound, max_bound):
    """
    General function that is nonsmoothly bounded between a min and max. 
    """
    return np.clip(funcval, min_bound, max_bound)
def derivative__bounded_nonsmooth_func(funcval, min_bound, max_bound):
    """
    Derivative of a general function that is nonsmoothly bounded between a min and max. 
    """
    in_bounds = (min_bound < funcval) & (funcval < max_bound)
    return np.where(in_bounds, 1.0, 0.0)